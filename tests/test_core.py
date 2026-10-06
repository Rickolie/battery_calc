"""Tests for the acceptance criteria in Specs.txt section 10."""
import io
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from battery_calc.analysis import Analysis, Inputs, Options  # noqa: E402
from battery_calc.battery import (FULL_DISCHARGE, GRID_CHARGE, SELF, Battery, perfect_foresight,  # noqa: E402
                                  plan_dynamic, plan_forecast, simulate)
from battery_calc.breakeven import breakeven, lifetime_kwh, wear_cost  # noqa: E402
from battery_calc.config import Connection, load_config  # noqa: E402
from battery_calc.contracts import Contract, parse_vast  # noqa: E402
from battery_calc.costs import Period, compute_cost, marginal_values  # noqa: E402
from battery_calc.p1 import FLAG_SYNTH, build_profile_year, load_p1  # noqa: E402
from battery_calc.prices import load_price_history, replay_prices  # noqa: E402
from battery_calc.taxes import Taxes  # noqa: E402
from synth import write_year  # noqa: E402

CFG = load_config(os.path.join(ROOT, "config.yaml"))
P1 = os.path.join(ROOT, "data", "Energy_house.csv")


def make_battery(**kw):
    d = dict(id="b", brand="Test", model="5kWh", usable_kwh=5.0, max_charge_w=2000, max_discharge_w=2000,
             rte=0.9, standby_w=5, cycle_life=6000, eol_capacity=0.7, warranty_years=10, price_nl=2000)
    d.update(kw)
    return Battery(**d)


@pytest.fixture(scope="module")
def profile():
    df, rep = load_p1(P1, CFG)
    return build_profile_year(df, rep, CFG), rep


def synthetic_p1(drop_month=None) -> io.StringIO:
    idx = pd.date_range("2025-01-01", "2026-01-01", freq="15min", tz="Europe/Amsterdam", inclusive="left")
    h = idx.hour + idx.minute / 60
    imp = np.where((h < 8) | (h > 18), 0.15, 0.02)
    exp = np.where((h > 10) & (h < 16), 0.2 * (1 + np.sin(idx.dayofyear / 365 * np.pi)), 0.0)
    df = pd.DataFrame({"time": idx.tz_localize(None).strftime("%Y-%m-%d %H:%M"),
                       "Import T1 kWh": 1000 + np.cumsum(imp) * 0.5, "Import T2 kWh": 500 + np.cumsum(imp) * 0.5,
                       "Export T1 kWh": 100 + np.cumsum(exp) * 0.5, "Export T2 kWh": 50 + np.cumsum(exp) * 0.5,
                       "L1 max W": 0, "L2 max W": 0, "L3 max W": 0})
    if drop_month:
        df = df[~df["time"].str.startswith(drop_month)]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------- data loading

def test_p1_dst_and_counts(profile):
    pr, rep = profile
    assert rep.duplicates == 0 and not rep.resets
    # 23 h and 25 h days in the profile year
    assert len(pr.loc["2026-03-29"]) == 92
    assert len(pr.loc["2025-10-26"]) == 100
    assert len(pr) == pd.date_range(rep.window[0], rep.window[1], freq="15min", inclusive="left").size


def test_synthetic_october_flagged_and_real_check(profile):
    pr, rep = profile
    assert "2025-10" in rep.missing_months
    octo = pr.loc["2025-10"]
    assert (octo["flag"] == FLAG_SYNTH).mean() > 0.9
    assert (pr.loc["2026-01", "flag"] != FLAG_SYNTH).all()
    assert octo["imp"].sum() > 0 and octo["exp"].sum() > 0


def test_missing_month_upload_is_filled():
    df, rep = load_p1(synthetic_p1(drop_month="2025-06"), CFG, label="upload")
    assert rep.long_gaps, "the missing month must be reported as a gap"
    pr = build_profile_year(df, rep, CFG)
    assert "2025-06" in rep.missing_months
    june = pr.loc["2025-06"]
    assert (june["flag"] == FLAG_SYNTH).all()
    assert june["exp"].sum() > 0


def test_p1_gap_interpolation_and_reset():
    buf = synthetic_p1()
    df = pd.read_csv(buf)
    df = df.drop(index=[100, 101])                       # 30-min gap -> interpolated
    df.loc[5000:, "Import T1 kWh"] -= 900                 # counter reset
    b = io.StringIO()
    df.to_csv(b, index=False)
    b.seek(0)
    d, rep = load_p1(b, CFG)
    assert len(rep.interpolated_gaps) == 1
    assert len(rep.resets) == 1
    assert (d["imp"].dropna() >= 0).all()


def test_too_little_data_stops():
    buf = synthetic_p1()
    df = pd.read_csv(buf).head(96 * 10)
    b = io.StringIO()
    df.to_csv(b, index=False)
    b.seek(0)
    d, rep = load_p1(b, CFG)
    with pytest.raises(ValueError, match="one month"):
        build_profile_year(d, rep, CFG)


def test_price_loader_formats(tmp_path):
    for y, layout in ((2023, "comma_kwh"), (2024, "semicolon_mwh"), (2025, "incl_only")):
        write_year(str(tmp_path), y, layout)
    cfg = dict(CFG)
    cfg["paths"] = dict(CFG["paths"], price_glob=str(tmp_path / "jeroen_punt_nl_dynamische_stroomprijzen_jaar_*.csv"))
    s, rep = load_price_history(cfg, Taxes(CFG))
    assert rep.years == [2023, 2024, 2025]
    # All normalised to €/kWh kale, quarter-hourly.
    assert 0.0 < s.mean() < 0.3
    assert s.index.to_series().diff().dropna().dt.total_seconds().max() == 900
    for y in rep.years:
        assert rep.missing_hours[y] == 0


def test_replay_matches_weekday_and_quarter(profile):
    pr, _ = profile
    idx = pd.date_range("2024-01-01", "2025-01-01", freq="15min", tz="Europe/Amsterdam", inclusive="left")
    prices = pd.Series(0.1, index=idx)
    t, pos, vals = replay_prices(pr.index, prices, 2024)
    src = pr.index[pos]
    assert (src.weekday == t.weekday).all()
    assert ((src.hour * 4 + src.minute // 15) == (t.hour * 4 + t.minute // 15)).mean() > 0.99
    assert len(t) == len(idx)            # leap year 2024: every quarter-hour gets a profile value


# ---------------------------------------------------------------- contracts

def test_vast_parse():
    with open(os.path.join(ROOT, "data", "vast.txt"), encoding="utf-8") as f:
        c, facts = parse_vast(f.read())
    assert c.price_t1 == pytest.approx(0.12405) and c.price_t2 == pytest.approx(0.13445)
    assert c.fixed_eur_year == pytest.approx(135.0)
    assert c.feed_in_cost_year(4500) == pytest.approx(421.0)
    assert c.feed_in_cost_year(20000) == pytest.approx(1259.0)
    assert facts["grid_eur_year_excl_vat"] == pytest.approx(393.25)
    assert facts["energy_tax_incl_vat_implied"] / 1.21 == pytest.approx(0.09161, abs=1e-5)


def test_saldering_nets_yearly_totals():
    taxes = Taxes(CFG)
    c = Contract("f", "S", "P", "fixed", price_single=0.1, feed_in=0.05)
    imp = np.array([1.0, 0.0, 1.0, 0.0])
    exp = np.array([0.0, 1.5, 0.0, 0.0])
    per = Period(imp, exp, np.zeros(4, bool), None, 365, 2026)
    sal = compute_cost(per, c, CFG["regimes"]["saldering"], taxes, "3x25", include_fixed=False)
    assert sal.energy_tax == pytest.approx(0.5 * taxes.energy_tax(2026) * 1.21)
    assert sal.supply == pytest.approx(0.5 * 0.1 * 1.21)
    no = compute_cost(per, c, CFG["regimes"]["nosal_min50"], taxes, "3x25", include_fixed=False)
    assert no.energy_tax == pytest.approx(2.0 * taxes.energy_tax(2027) * 1.21)
    # 50% minimum of the kale price applies after 2027
    assert no.feed_in_income == pytest.approx(1.5 * 0.05)


def test_feed_in_costs_never_make_export_negative():
    taxes = Taxes(CFG)
    c = Contract("f", "S", "P", "fixed", price_single=0.1, feed_in=0.0, feed_in_cost_kwh=0.05)
    per = Period(np.zeros(4), np.ones(4), np.zeros(4, bool), None, 365, 2027)
    r = compute_cost(per, c, CFG["regimes"]["nosal_2030"], taxes, "3x25", include_fixed=False)
    assert r.feed_in_costs <= max(r.feed_in_income, 0) + 1e-12


# ---------------------------------------------------------------- battery

def random_net(n=96 * 28, seed=1):
    rng = np.random.default_rng(seed)
    h = (np.arange(n) % 96) / 4
    return 0.15 - 0.4 * np.exp(-((h - 13) ** 2) / 8) + rng.normal(0, 0.05, n)


def test_energy_balance_closes_every_interval():
    b = make_battery()
    net = random_net()
    r = simulate(net, SELF, b, 4600)
    sb = b.standby_w / 4000
    assert np.allclose(r.imp - r.exp, net + sb + r.batt_ac, atol=1e-12)
    u = 0.25 + 0.1 * np.sin(np.arange(len(net)) / 96 * 2 * np.pi)
    lp = perfect_foresight(net, u, u - 0.1, b, 4600)
    assert np.allclose(lp.imp - lp.exp, net + sb + lp.batt_ac, atol=1e-7)


def test_soc_and_power_limits():
    b = make_battery(max_charge_w=800, max_discharge_w=800)
    net = random_net()
    r = simulate(net, SELF, b, 4600, keep_trace=True)
    assert r.soc.min() >= -1e-12 and r.soc.max() <= b.usable_kwh + 1e-9
    assert np.abs(r.batt_ac).max() <= 0.8 * 0.25 + 1e-12


def test_zero_spread_gives_zero_savings_under_dynamic():
    taxes = Taxes(CFG)
    b = make_battery(rte=1.0, standby_w=0.0)
    c = Contract("d", "S", "Dyn", "dynamic", markup=0.0, feed_in_markup=0.0)
    net = random_net()
    imp, exp = np.maximum(net, 0), np.maximum(-net, 0)
    idx = pd.date_range("2025-03-01", periods=len(net), freq="15min", tz="Europe/Amsterdam")
    spot = np.full(len(net), 0.1)
    per = Period(imp, exp, np.zeros(len(net), bool), spot, len(net) / 96, 2026)
    reg = CFG["regimes"]["saldering"]
    u, s = marginal_values(per, c, reg, taxes, net_importer=True)
    modes = plan_dynamic(idx, u, s, b, 4600, wear=0.0, sell=True, export=exp)
    r = simulate(net, modes, b, 4600, degrade=False)
    base = compute_cost(per, c, reg, taxes, "3x25", include_fixed=False).total
    withb = compute_cost(per.with_flows(r.imp, r.exp), c, reg, taxes, "3x25", include_fixed=False).total
    assert base - withb == pytest.approx(0.0, abs=1e-9)


def test_perfect_foresight_bounds_heuristics():
    b = make_battery()
    net = random_net()
    idx = pd.date_range("2025-06-01", periods=len(net), freq="15min", tz="Europe/Amsterdam")
    u = 0.25 + 0.1 * np.sin(np.arange(len(net)) / 96 * 2 * np.pi)
    s = u - 0.15
    cost = lambda r: float((u * r.imp - s * r.exp).sum())
    lp = perfect_foresight(net, u, s, b, 4600)
    sc = simulate(net, SELF, b, 4600, degrade=False)
    dyn = simulate(net, plan_dynamic(idx, u, s, b, 4600, wear=0.0, export=np.maximum(-net, 0)), b, 4600,
                   degrade=False)
    assert cost(lp) <= cost(sc) + 1e-6
    assert cost(lp) <= cost(dyn) + 1e-6
    fc = simulate(net, plan_forecast(idx, net, u, s, b, 4600, wear=0.0), b, 4600, degrade=False)
    assert cost(lp) <= cost(fc) + 1e-6
    # restricted bounds: fewer freedoms never earn more
    solar_only = perfect_foresight(net, u, s, b, 4600, grid_charge=False, sell=False)
    assert cost(lp) <= cost(solar_only) + 1e-6
    assert np.all(solar_only.batt_ac <= np.maximum(-net, 0) + 1e-7)   # charges only from surplus


def test_german_prices_pick_the_cheaper_country():
    from battery_calc.battery import BEST_PRICE
    b = make_battery(price_nl=1199.0)
    b.price_de = 990.0
    assert b.best_price(50.0) == (1040.0, "DE")
    assert b.price_variants(50.0, best=True)[BEST_PRICE] == 1040.0
    b.price_de = 1180.0
    assert b.best_price(50.0) == (1199.0, "NL")          # DE + travel is dearer: stay in NL
    assert BEST_PRICE not in b.price_variants(50.0)       # only with German prices on


def test_curtail_only_cuts_negative_value_export():
    from battery_calc.battery import curtail
    b = make_battery()
    net = random_net()
    s = np.where(np.arange(len(net)) % 7 == 0, -0.05, 0.08)
    r = simulate(net, SELF, b, 4600, degrade=False)
    c = curtail(r, s)
    assert np.all(c.exp[s < 0] == 0) and np.allclose(c.exp[s >= 0], r.exp[s >= 0])
    assert np.allclose(c.curtailed + c.exp, r.exp)
    value = lambda x: float((s * x.exp).sum())  # noqa: E731
    assert value(c) >= value(r)                 # never earns less on export
    assert np.allclose(c.imp, r.imp)            # curtailing never adds import


def test_forecast_optimiser_uses_known_prices_and_beats_self_consumption():
    """A repeating day with a cheap night and an expensive evening: the
    optimiser charges at night and sells/covers the evening, and with a
    perfect forecast it matches perfect foresight closely."""
    b = make_battery(standby_w=0.0)
    days = 14
    idx = pd.date_range("2025-03-01", periods=96 * days, freq="15min", tz="UTC")
    h = np.asarray(idx.hour)
    net = np.where((h >= 17) & (h < 22), 0.4, 0.05)
    u = np.where(h < 6, 0.10, np.where((h >= 17) & (h < 22), 0.45, 0.25))
    s = u - 0.12
    cost = lambda r: float((u * r.imp - s * r.exp).sum())  # noqa: E731
    modes = plan_forecast(idx, net, u, s, b, 4600, wear=0.0)
    fc = simulate(net, modes, b, 4600, degrade=False)
    sc = simulate(net, SELF, b, 4600, degrade=False)
    lp = perfect_foresight(net, u, s, b, 4600)
    assert (modes[h < 6] == GRID_CHARGE).any()
    assert not (modes == FULL_DISCHARGE)[h < 6].any()
    assert cost(fc) < cost(sc) - 1.0
    assert cost(fc) - cost(lp) <= 0.05 * (cost(sc) - cost(lp))     # within 5% of the bound's gain


def test_breakeven_lossless_infinite_life_equals_charge_price():
    b = make_battery(rte=1.0, cycle_life=math.inf, warranty_years=math.inf)
    w, life, limit = wear_cost(b, 2000, 250)
    assert w == 0.0
    assert breakeven(0.17, b.rte, w) == pytest.approx(0.17)


def test_lifetime_limit_reported():
    assert lifetime_kwh(make_battery(cycle_life=1000), 250)[1] == "cycles"
    assert lifetime_kwh(make_battery(warranty_mwh=5), 250)[1] == "warranty throughput"
    assert lifetime_kwh(make_battery(), 100)[1] == "calendar life"


def test_degradation_reduces_capacity():
    b = make_battery(cycle_life=100)
    r = simulate(random_net(96 * 200), SELF, b, 4600)
    assert r.end_capacity < b.usable_kwh
    assert r.end_capacity >= b.usable_kwh * b.eol_capacity - 1e-9


def test_connection_caps():
    c1 = Connection.parse("1x35", 0.2)
    c3 = Connection.parse("3x25", 0.2)
    assert c3.battery_cap_w(1) == pytest.approx(25 * 230 * 0.8)
    assert c1.battery_cap_w(1) == pytest.approx(35 * 230 * 0.8)
    assert c1.battery_cap_w(3) is None
    assert c3.battery_cap_w(3) == pytest.approx(3 * 25 * 230 * 0.8)


def test_connection_changes_run(tmp_path):
    bat = tmp_path / "b.csv"
    src = pd.read_csv(os.path.join(ROOT, "data", "online", "batteries.csv"), dtype=str)
    three = src.iloc[[0]].copy()
    three["id"], three["phases"], three["max_charge_w"], three["max_discharge_w"] = "big3", "3", "15000", "15000"
    pd.concat([src, three]).to_csv(bat, index=False)
    res = {}
    for conn in ("1x35", "3x25"):
        an = Analysis(CFG, Inputs(batteries_csv=str(bat)), Options(connection=conn, plots=False), log=None)
        an.no_write = True
        an.step_load()
        an.step_current_contract()
        an.step_batteries()
        res[conn] = an.sections[-1].tables["battery_specs"].set_index("id")
    assert res["1x35"].loc["big3", "in_scope"] == "no"
    assert res["3x25"].loc["big3", "in_scope"] == "yes"
    assert res["3x25"].loc["big3", "power_cap_w"] == pytest.approx(3 * 25 * 230 * 0.8)
    assert res["1x35"].loc["marstek_v3_5", "power_cap_w"] == 2500


def test_full_run_both_regimes_and_analyses(tmp_path):
    for y in (2023, 2024, 2025):
        write_year(str(tmp_path), y, "comma_kwh")
    bat = tmp_path / "b.csv"
    src = pd.read_csv(os.path.join(ROOT, "data", "online", "batteries.csv"), dtype=str).iloc[[2]]
    src["price_nl_incl_vat"] = "1200"
    src.to_csv(bat, index=False)
    cfg = dict(CFG)
    cfg["paths"] = dict(CFG["paths"], price_glob=str(tmp_path / "jeroen_*.csv"), results_dir=str(tmp_path / "res"))
    cfg["strategies"] = dict(CFG["strategies"], enabled=["self_consumption", "dynamic"])
    an = Analysis(cfg, Inputs(batteries_csv=str(bat)), Options(plots=False), log=None)
    secs = {s.id: s for s in an.run()}
    keys = list(secs["contracts"].tables)
    for scn in ("2026 rules", "2027-2029 rules", "2030+ rules"):
        assert any(k.startswith("Last 3 years") and scn in k for k in keys)
        assert any(k.startswith("Full history") and scn in k for k in keys)
    rk = secs["payback"].tables["payback_ranked"]
    assert set(rk["analysis"]) == {"headline", "full"}
    assert (rk["payback_years"] > 0).all()
    assert os.path.exists(tmp_path / "res" / "breakeven.csv")
    # every column shown in a report table has a plain-language description
    from battery_calc.columns import describe
    from battery_calc.report import shown_columns
    missing = [(name, c) for s in secs.values() for name, df in s.tables.items()
               for c in shown_columns(df) if describe(c, name) is None]
    assert not missing, "\n".join(map(str, missing))
    # every printed table has a one-line finding, and sections render collapsed
    from battery_calc.report import section_html
    from battery_calc.summaries import summarize
    no_finding = [name for s in secs.values() for name, df in s.tables.items()
                  if name not in s.csv_only and not df.empty and not summarize(name, df)]
    assert not no_finding, no_finding
    h = section_html(secs["payback"])
    assert "<details class='sec'>" in h and "<details class='tblw'>" in h


def test_hbc_extreme_pair_matching():
    from battery_calc.battery import CHARGE_PV, GRID_CHARGE, plan_hbc
    idx = pd.date_range("2025-01-06", periods=96, freq="15min", tz="Europe/Amsterdam")
    price = np.full(96, 0.20)
    price[8:12] = 0.05          # cheap night
    price[72:76] = 0.40         # evening peak
    m = plan_hbc(idx, price, {"min_delta": 0.06})
    assert (m[8:12] == GRID_CHARGE).all() and (m[72:76] == SELF).all()
    assert (m[20:60] == CHARGE_PV).all()          # neutral default
    capped = plan_hbc(idx, price, {"min_delta": 0.06, "cheapest_hrs": 0.5})
    assert (capped == GRID_CHARGE).sum() == 2 and (capped[72:76] == SELF).all()
    flat = plan_hbc(idx, np.full(96, 0.2), {"min_delta": 0.06})
    assert (flat == CHARGE_PV).all()


# ---------------------------------------------------------------- additions (Specs.txt section 11)

def test_min_price_difference():
    from battery_calc.extras import min_price_difference
    assert min_price_difference(0.0, 0.85, 0.13) == pytest.approx(0.13)
    assert min_price_difference(0.20, 1.0, 0.0) == pytest.approx(0.0)
    # break-even minus charge price
    assert min_price_difference(0.2, 0.85, 0.1) == pytest.approx(0.2 / 0.85 + 0.1 - 0.2)


def test_earnings_split_closes_to_saving():
    from battery_calc.extras import earnings_split
    b = make_battery()
    net = random_net()
    r = simulate(net, SELF, b, 4600)
    u = np.full(len(net), 0.3)
    s = np.full(len(net), 0.07)
    saving = float(((np.maximum(net, 0) - r.imp) * u - (np.maximum(-net, 0) - r.exp) * s).sum())
    parts = earnings_split(net, r.batt_ac, b.standby_w / 4000, u, s, saving)
    assert sum(parts.values()) == pytest.approx(saving)
    assert parts["avoided_import"] > 0 and parts["solar_feed_in_given_up"] < 0
    assert abs(parts["standby_and_other"]) < 0.05 * abs(saving) + 5   # only standby remains


def test_black_friday_variants():
    b = make_battery(price_nl=1210.0)
    v = b.price_variants(50.0, {"discount_nl": 0.1, "discount_de": 0.2, "vat": 0.21})
    assert v["NL Black Friday (est.)"] == pytest.approx(1089.0)
    assert v["DE Black Friday (est.)"] == pytest.approx(1000 * 0.8 + 50)
    b.price_bf_nl = 999.0
    assert b.price_variants(0.0, {})["NL Black Friday"] == pytest.approx(999.0)
    off = b.price_variants(0.0, {"estimates": False})
    assert "NL Black Friday" in off and not any("(est.)" in k for k in off)


def test_kiln_days_battery_helps_and_bigger_kiln_fewer_days(profile):
    from battery_calc.extras import kiln_days
    pr, _ = profile
    june = pr.loc["2026-06"]
    b = make_battery(usable_kwh=5.0, max_charge_w=2500, max_discharge_w=2500, standby_w=0)
    soc = np.full(len(june), 2.5)
    nb = kiln_days(june, None, 0, None, [1.5, 4.5], 8, 0.7, [8, 9, 10], 0.05)
    wb = kiln_days(june, b, 4600, soc, [1.5, 4.5], 8, 0.7, [8, 9, 10], 0.05)
    small_nb, big_nb = (nb[nb.kiln_kw == p].free.sum() for p in (1.5, 4.5))
    small_wb = wb[wb.kiln_kw == 1.5].free.sum()
    assert small_nb >= big_nb
    assert small_wb >= small_nb


def test_power_profile_shares(profile):
    from battery_calc.extras import power_profile
    pr, _ = profile
    real = (pr["flag"] != "synthetic").values
    stats, cover, _, _, _ = power_profile(pr, real)
    shares = cover["share_of_surplus_it_can_store"].values
    assert np.all(np.diff(shares) >= 0) and shares[-1] <= 1.0 + 1e-9


def test_variant_match_rules():
    from battery_calc.scrape import _variant_match
    t = "SolarFlow 2400 AC+ (2.4 kWh) / 1*AB3000L (2.88 kWh)"
    assert _variant_match(t, "1*ab3000l&!smart meter")
    assert not _variant_match("SolarFlow 2400 AC+ + Smart Meter D0 / 1*AB3000L", "1*ab3000l&!smart meter")
    assert _variant_match("STREAM AC 5000", "=stream ac 5000")
    assert not _variant_match("2 × STREAM AC 5000", "=stream ac 5000")
    assert _variant_match("SolarVault 3 Pro Max AC 5,04kWh", "5.04kwh")


def test_blackfriday_scrape_records_deal_and_reference(tmp_path, monkeypatch):
    import datetime as dt
    import json as js
    from battery_calc import scrape
    csv = tmp_path / "b.csv"
    pd.DataFrame([{"id": "x", "brand": "B", "model": "M", "price_nl_incl_vat": "", "price_nl_lowest_incl_vat": "",
                   "shop_urls": "shopify:https://shop.example/products/bat|=base",
                   "shop_urls_de": "shopify:https://shop.de/products/bat|=base"}]).to_csv(csv, index=False)
    prices = {"nl": 1000_00, "de": 800_00}

    def fake_fetch(url, timeout=30):
        cents = prices["de"] if "shop.de" in url else prices["nl"]
        return js.dumps({"variants": [{"title": "Base", "price": cents, "compare_at_price": None, "available": True},
                                      {"title": "Base + extra", "price": 1, "available": True}]})
    monkeypatch.setattr(scrape, "fetch", fake_fetch)
    cfg = dict(CFG)
    cfg["paths"] = dict(CFG["paths"], batteries_file=str(csv))
    scrape.scrape_battery_prices(cfg, today=dt.date(2026, 11, 10))          # before the window
    prices.update(nl=850_00, de=700_00)
    scrape.scrape_battery_prices(cfg, today=dt.date(2026, 11, 27))          # Black Friday
    prices.update(nl=900_00, de=750_00)
    scrape.scrape_battery_prices(cfg, today=dt.date(2026, 11, 28))          # price goes up again
    r = pd.read_csv(csv).iloc[0]
    assert r.price_nl_ref_incl_vat == 1000 and r.price_de_ref_excl_vat == 800
    assert r.price_blackfriday_nl_incl_vat == 850 and r.price_blackfriday_de_excl_vat == 700   # lowest kept
    assert r.price_blackfriday_nl_date == "2026-11-27"
    assert r.price_nl_incl_vat == 900
    scrape.scrape_battery_prices(cfg, today=dt.date(2027, 11, 1))           # next season starts clean
    assert pd.isna(pd.read_csv(csv).iloc[0].price_blackfriday_nl_incl_vat)
