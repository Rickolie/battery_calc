"""One-line findings per table, shown next to each (collapsed) table on the
website and in the report, so the main result is visible without opening it."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _ref(df) -> str:
    """The reference price variant: the cheaper of NL/DE when German prices are on."""
    return "Cheapest NL/DE" if (df.price_variant == "Cheapest NL/DE").any() else "NL current"


def _e(v) -> str:
    return f"€{v:,.0f}"


def _best(df, col, low=True):
    d = df.dropna(subset=[col])
    d = d[np.isfinite(d[col].astype(float))]
    if d.empty:
        return None
    return d.loc[d[col].astype(float).idxmin() if low else d[col].astype(float).idxmax()]


def _contracts(df):
    r = _best(df, "expected_eur_year")
    cur = df[df.contract.str.startswith("Current")]
    s = f"Cheapest: {r.contract} at about {_e(r.expected_eur_year)} a year"
    if len(cur):
        s += f", {_e(cur.iloc[0].expected_eur_year - r.expected_eur_year)} less than your current contract"
    dyn = df[df.type == "dynamic"]
    fix = df[(df.type != "dynamic") & ~df.contract.str.startswith("Current")]
    if len(dyn) and len(fix):
        bf = _best(fix, "expected_eur_year")
        s += f"; cheapest fixed: {bf.contract} ({_e(bf.expected_eur_year)})"
    return s + "."


def _by_year_group(df):
    g = df.groupby("group").expected.mean().sort_values()
    return (f"Contract costs per price period: cheapest on average in {g.index[0]} prices, most expensive in "
            f"{g.index[-1]} prices.")


def _feed_in(df):
    last = df.iloc[-1]
    return (f"In {int(last.year)} your exported solar earned on average {last.feed_in_income_per_kwh * 100:.1f} ct/kWh "
            f"(average spot {last.avg_spot_eur_kwh * 100:.1f} ct), with {last.export_at_negative_price_pct:.0f}% "
            "of the export at negative prices.")


def _current(df):
    parts = [f"{r.scenario}: {_e(r.total)}" for r in df.itertuples()]
    return "Your current contract per year: " + "; ".join(parts) + "."


def _monthly(df):
    imp, exp = df.import_kwh.sum(), df.export_kwh.sum()
    syn = df.index[df.synthetic_share > 0.5].tolist() if "synthetic_share" in df else []
    s = f"Profile year: {imp:,.0f} kWh imported and {exp:,.0f} kWh exported"
    if syn:
        s += f"; mostly synthetic month(s): {', '.join(map(str, syn))}"
    return s + "."


def _price_overview(df):
    last = df.iloc[-1]
    hi = _best(df, "avg_daily_spread_eur_kwh", low=False)
    return (f"Latest year {int(last.year)}: average day-ahead price {last.avg_price_eur_kwh * 100:.1f} ct/kWh, "
            f"{int(last.negative_price_hours)} negative-price hours; the largest daily spread was in {int(hi.year)}.")


def _import_export(df):
    d = df.set_index("flow")
    return (f"Import peaks at {d.loc['import', 'max_kw']:.1f} kW (99% of the time below {d.loc['import', 'p99_kw']:.1f} kW); "
            f"export peaks at {d.loc['export', 'max_kw']:.1f} kW.")


def _power_vs_energy(df):
    r = df.iloc[0]
    return (f"A {r.battery_power_kw:g} kW battery could store {r.share_of_surplus_it_can_store:.0%} of the surplus and "
            f"cover {r.share_of_import_it_can_cover:.0%} of the import; more power raises both.")


def _phase(df):
    r = _best(df, "max_kw", low=False)
    return f"Highest peak on {r.phase.replace(' max W', '')}: {r.max_kw:.1f} kW."


def _specs(df):
    d = df[df.get("in_scope", "yes") == "yes"] if "in_scope" in df else df
    r = _best(d, "eur_per_kwh_usable")
    return (f"{len(d)} batteries in scope; cheapest per usable kWh: {r.battery} at "
            f"€{r.eur_per_kwh_usable:,.0f}/kWh.")


def _breakeven(df):
    d = df[df.price_variant == _ref(df)] if "price_variant" in df else df
    r = _best(d, "wear_eur_kwh")
    return (f"Lowest wear cost: {r.battery} at {r.wear_eur_kwh * 100:.1f} ct per kWh delivered; discharging only pays "
            "when the price is at least charge price ÷ RTE + wear.")


def _min_diff(df):
    r = _best(df, "at_charge_0.10")
    return (f"Smallest price difference worth charging (charging at 10 ct): {r['at_charge_0.10'] * 100:.1f} ct with "
            f"{r.battery}.")


def _savings(df):
    col = next((c for c in df.columns if c.startswith("saving_2027")), None)
    if col is None:
        return None
    d = df[df.analysis == "headline"] if "analysis" in df else df
    d = d[d.strategy != "perfect_foresight"]          # only achievable strategies
    r = _best(d, col, low=False)
    return f"Largest saving from 2027: {r.battery} with {r.strategy} on {r.contract}, {_e(r[col])} a year."


def _ranked(df):
    if "yearly_cost_with_battery_2027" in df:          # compact table: one row per battery × contract
        c = _best(df, "yearly_cost_with_battery_2027")
        p = _best(df, "payback_years")
        return (f"Lowest yearly cost: {c.battery} on {c.contract}, {_e(c.yearly_cost_with_battery_2027)} a year; "
                f"fastest payback: {p.battery} on {p.contract} ({p.payback_years:.1f} years).")
    d = df[(df.analysis == "headline") & (df.price_variant == _ref(df)) & (df.strategy != "perfect_foresight")]
    r = _best(d if len(d) else df, "payback_years")
    where = " (bought in Germany)" if r.get("bought_in") == "DE" else ""
    return (f"Fastest payback at today's price{where}: {r.battery} with {r.strategy} on {r.contract}, "
            f"{r.payback_years:.1f} years (net present value {_e(r.npv_eur)}).")


def _scale(df):
    b = df[df.best_for_battery.astype(bool)]
    common = b.scale.mode().iloc[0] if len(b) else None
    return (f"The best break-even scale is most often {common:g}× (below 1 = trade on smaller price differences)."
            if common is not None else None)


def _sensitivity(df):
    base = df[df.variant == "base"]
    worst = _best(df[df.variant != "base"], "saving_eur_year_2027_rules")
    if worst is None or base.empty:
        return None
    b = base[(base.battery == worst.battery) & (base.contract == worst.contract)]
    ref = b.iloc[0].saving_eur_year_2027_rules if len(b) else base.iloc[0].saving_eur_year_2027_rules
    return (f"Biggest risk: '{worst.variant}' lowers the saving of {worst.battery} from {_e(ref)} to "
            f"{_e(worst.saving_eur_year_2027_rules)} a year.")


def _blackfriday(df):
    r = df.iloc[0]
    found = df.deal_found.astype(str).str.len().gt(0).sum() if "deal_found" in df else 0
    s = f"Best Black Friday payback: {r.battery}, {r.best_payback:.1f} years"
    return s + (f"; real deals found for {found} batteries." if found else "; no real deals recorded yet (estimates).")


def _earnings(df):
    y = df[df.strategy != "perfect_foresight"].groupby("strategy").net_saving.sum().sort_values(ascending=False)
    return f"Over the battery's life {y.index[0]} earns most: {_e(y.iloc[0])} in total."


def _gap(df):
    cls = [c for c in df.columns if c.endswith(" kWh")]
    v = df.set_index("strategy")
    parts = [f"{c}: forecast {_e(v.loc['forecast', c])} vs perfect {_e(v.loc['perfect_foresight', c])}"
             for c in cls if "forecast" in v.index and "perfect_foresight" in v.index]
    return "Per size class – " + "; ".join(parts) + ". Most of the gap is forecasting your own solar and usage."


def _earn_avg(df):
    parts = []
    for cls, g in df.groupby("size_class", sort=False):
        g = g[g.strategy != "perfect_foresight"]
        r = g.loc[g.net_saving.idxmax()]
        parts.append(f"{cls}: {r.strategy} {_e(r.net_saving)}")
    return "Best net saving in an average full year – " + "; ".join(parts) + "."


def _sanity(df):
    ok = df.balance_error_kwh.max() < 1e-6
    parts = [f"{r.size_class} full on {int(r.days_full)} days" for r in df.itertuples()]
    return ("Energy balance closes" if ok else "Energy balance error!") + "; " + ", ".join(parts) + "."


def _comparison(df):
    r = df.iloc[0]
    return (f"{r.battery}: {r.payback_years:.1f} years payback, {_e(r.saving_eur_year_2027)} a year from 2027 "
            f"with {r.strategy}.")


def _size_class(df):
    r = _best(df, "payback_years")
    n = _best(df, "npv_eur", low=False)
    s = f"Fastest payback: {r.size_class} ({r.battery}, {r.payback_years:.1f} years)"
    if n is not None and n.size_class != r.size_class:
        s += f"; most value over its life: {n.size_class} ({n.battery}, NPV {_e(n.npv_eur)})"
    return s + "."


def _combos(df):
    r = df.loc[df.yearly_cost_eur_2027.idxmin()]
    return f"Cheapest from 2027: {r.option}, about {_e(r.yearly_cost_eur_2027)} a year."


def _kiln(df):
    d = df[df.free_days > 0]
    if d.empty:
        return "No kiln size fires on surplus solar alone."
    big = d[d.free_days >= 20]
    r = (big if len(big) else d).iloc[-1]
    return f"{r.kiln_kw:g} kW fires on surplus solar on {int(r.free_days)} days a year ({r.months_with_free_days})."


def _solar(df):
    neg = df[df.cumulative_eur < 0]
    paid = df[df.cumulative_eur >= 0]
    s = f"Saved {_e(df.saving_eur.iloc[:3].sum())} in {df.year.iloc[0]}–{df.year.iloc[min(2, len(df) - 1)]}"
    if len(neg) and len(paid):
        s += f"; earned back in {int(paid.year.iloc[0])}"
    return s + f"; {_e(df.cumulative_eur.iloc[-1])} at the end of their life."


def _fixdyn(df):
    fx = df[df.role == "cheapest fixed offer"]
    dy = df[df.role == "cheapest dynamic"]
    if fx.empty or dy.empty:
        return None
    f, d = fx.iloc[0], dy.iloc[0]
    return (f"From 2027: fixed {f.contract} {_e(f.cost_2027_2029_eur)} vs dynamic {d.contract} "
            f"{_e(d.cost_2027_2029_eur)} a year – fixed costs {_e(f.cost_2027_2029_eur - d.cost_2027_2029_eur)} more.")


def _fixdyn_batt(df):
    d = df.dropna(subset=["fixed_costs_more_eur"]) if "fixed_costs_more_eur" in df else df.iloc[0:0]
    if d.empty:
        return None
    return ("With a battery, fixed still costs more per year: " + "; ".join(
        f"{r.size_class} {_e(r.fixed_costs_more_eur)}" for r in d.itertuples()) + ".")


def _kiln_month(df):
    return "Free firing days per month and kiln size (surplus solar only)."


def _energieknl(df):
    return f"Reference supply costs of {len(df)} suppliers (energieknl.nl)."


RULES = [
    (r"^(Last 3 years|Full history)( \(headline\))? – ", _contracts),
    (r"by year group", _by_year_group),
    (r"feed_in_value_per_year", _feed_in),
    (r"current_contract_cost", _current),
    (r"monthly_flows", _monthly),
    (r"price_overview", _price_overview),
    (r"import_export_power", _import_export),
    (r"battery_power_vs_energy", _power_vs_energy),
    (r"peak_per_phase", _phase),
    (r"battery_specs", _specs),
    (r"^breakeven$", _breakeven),
    (r"minimum_price_difference", _min_diff),
    (r"savings_by_combination", _savings),
    (r"payback_ranked", _ranked),
    (r"dynamic_breakeven_scale", _scale),
    (r"^sensitivity$", _sensitivity),
    (r"black_friday", _blackfriday),
    (r"earnings_per_(year|ownership_year)", _earnings),
    (r"gap_to_perfect_foresight", _gap),
    (r"earnings_average_full_year", _earn_avg),
    (r"self_consumption_check", _sanity),
    (r"battery_comparison", _comparison),
    (r"best_per_size_class", _size_class),
    (r"contract_and_battery_combinations", _combos),
    (r"kiln_free_firing_days", _kiln),
    (r"fixed_vs_dynamic_with_battery", _fixdyn_batt),
    (r"^fixed_vs_dynamic$", _fixdyn),
    (r"solar_saving_per_year", _solar),
    (r"free_days_per_month", _kiln_month),
    (r"energieknl", _energieknl),
]


def summarize(name: str, df: pd.DataFrame) -> str | None:
    """One sentence with the table's main finding; None when there is nothing to say."""
    if df is None or df.empty:
        return None
    if not isinstance(df.index, pd.RangeIndex) and df.index.name is not None:
        df = df.reset_index()
    for pat, fn in RULES:
        if re.search(pat, name):
            try:
                return fn(df)
            except Exception:       # a summary must never break the report
                return None
    return None
