"""Additions of 2026-10-05 (Specs.txt section 11): minimum price difference,
power profile, earnings split per source, and Objective 4 (pottery kiln on
free power)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .battery import DT_H, SELF, Battery, simulate

# ---------------------------------------------------------------- 11.1 minimum price difference


def min_price_difference(charge_price: float, rte: float, wear: float) -> float:
    """Smallest all-in price difference between charging and discharging that
    pays for round-trip losses and wear: break-even − charge price."""
    return charge_price * (1.0 / rte - 1.0) + wear


def min_difference_rows(batteries, wear_by_id: dict, cheap_price: float, vat: float,
                        charge_prices=(0.0, 0.10, 0.20, 0.30)) -> list[dict]:
    rows = []
    for b in batteries:
        w = wear_by_id.get(b.id)
        if w is None or (isinstance(w, float) and math.isnan(w)):
            continue
        r = {"battery": b.name, "rte": b.rte, "wear_eur_kwh": round(w, 4)}
        for cp in charge_prices:
            r[f"at_charge_{cp:.2f}"] = round(min_price_difference(cp, b.rte, w), 3)
        typ = min_price_difference(cheap_price, b.rte, w)
        r["setting_all_in_eur_kwh"] = round(typ, 3)
        # Taxes per kWh are the same in every hour, so an all-in difference is the
        # spot difference times (1 + VAT).
        r["setting_spot_eur_kwh"] = round(typ / (1 + vat), 3)
        rows.append(r)
    return rows


# ---------------------------------------------------------------- 11.2 power profile

POWER_STEPS_KW = (0.8, 1.2, 1.6, 2.4, 3.0, 3.6, 4.6)


def power_profile(profile: pd.DataFrame, real_mask: np.ndarray, raw_p1=None, phase_cols=None):
    """Import/export power from 15-min energy, the share of energy a battery
    with a given power can absorb or deliver, and per-phase peaks."""
    imp_kw = profile["imp"].values[real_mask] / DT_H
    exp_kw = profile["exp"].values[real_mask] / DT_H
    pct = lambda a, q: float(np.percentile(a[a > 0], q)) if (a > 0).any() else 0.0
    stats = pd.DataFrame([
        {"flow": "import", "hours_per_year": (imp_kw > 0).sum() * DT_H * 365 / max(real_mask.sum() / 96, 1),
         "median_kw": pct(imp_kw, 50), "p90_kw": pct(imp_kw, 90), "p99_kw": pct(imp_kw, 99), "max_kw": imp_kw.max()},
        {"flow": "export", "hours_per_year": (exp_kw > 0).sum() * DT_H * 365 / max(real_mask.sum() / 96, 1),
         "median_kw": pct(exp_kw, 50), "p90_kw": pct(exp_kw, 90), "p99_kw": pct(exp_kw, 99), "max_kw": exp_kw.max()},
    ])
    cover = []
    for p in POWER_STEPS_KW:
        cover.append({"battery_power_kw": p,
                      "share_of_surplus_it_can_store": np.minimum(exp_kw, p).sum() / max(exp_kw.sum(), 1e-9),
                      "share_of_import_it_can_cover": np.minimum(imp_kw, p).sum() / max(imp_kw.sum(), 1e-9)})
    cover = pd.DataFrame(cover)
    phases = None
    if raw_p1 is not None and phase_cols:
        cols = [c for c in phase_cols if c in raw_p1.columns]
        if cols:
            ph = raw_p1[cols].apply(pd.to_numeric, errors="coerce") / 1000.0
            phases = pd.DataFrame({"phase": cols, "p50_kw": ph.median().values, "p99_kw": ph.quantile(0.99).values,
                                   "max_kw": ph.max().values})
    return stats, cover, phases, np.sort(imp_kw)[::-1], np.sort(exp_kw)[::-1]


# ---------------------------------------------------------------- 11.4 earnings per source


def earnings_split(net: np.ndarray, batt_ac: np.ndarray, standby_kwh: float, u: np.ndarray, s: np.ndarray,
                   true_saving: float) -> dict:
    """Value every battery flow at that interval's marginal prices.
    The rest (standby, saldering netting, feed-in tiers) closes to the real saving."""
    L = net + standby_kwh
    ch = np.maximum(batt_ac, 0.0)
    dis = np.maximum(-batt_ac, 0.0)
    ch_solar = np.minimum(ch, np.maximum(-L, 0.0))
    ch_grid = ch - ch_solar
    dis_home = np.minimum(dis, np.maximum(L, 0.0))
    dis_grid = dis - dis_home
    parts = {
        "avoided_import": float((dis_home * u).sum()),
        "sold_to_grid": float((dis_grid * s).sum()),
        "solar_feed_in_given_up": -float((ch_solar * s).sum()),
        "grid_charging": -float((ch_grid * u).sum()),
    }
    parts["standby_and_other"] = true_saving - sum(parts.values())
    return parts


# ---------------------------------------------------------------- 11.6 pottery kiln


def kiln_days(profile: pd.DataFrame, b: Battery | None, cap_w: float, soc_start: np.ndarray | None,
              powers_kw, firing_hours: float, duty: float, start_hours, tolerance: float) -> pd.DataFrame:
    """For every kiln power and day: can a firing to maximum temperature run on
    solar surplus (+ battery) with at most `tolerance` of its energy from the grid?
    Returns one row per power and day with the best start hour."""
    idx = profile.index
    net = (profile["imp"] - profile["exp"]).values
    day = np.asarray(idx.normalize().asi8)
    bounds = np.flatnonzero(np.diff(day)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(idx)]])
    rows = []
    sb = (b.standby_w or 0.0) / 1000.0 * DT_H if b is not None else 0.0
    for a, z in zip(starts, ends):
        d_net = net[a:z]
        hours = idx[a:z].hour + idx[a:z].minute / 60.0
        soc0 = float(soc_start[a]) if (b is not None and soc_start is not None) else 0.0
        if b is not None:
            base = simulate(d_net, SELF, b, cap_w, soc0=soc0, degrade=False)
            base_imp = base.imp.sum()
        else:
            base_imp = np.maximum(d_net, 0).sum()
        for p in powers_kw:
            need = p * firing_hours * duty
            best = None
            for h0 in start_hours:
                on = (hours >= h0) & (hours < h0 + firing_hours)
                if on.sum() * DT_H < firing_hours - 1e-9:
                    continue                       # firing would run past midnight / DST day
                load = np.where(on, p * duty * DT_H, 0.0)
                if b is not None:
                    r = simulate(d_net + load, SELF, b, cap_w, soc0=soc0, degrade=False)
                    extra = r.imp.sum() - base_imp
                else:
                    extra = np.maximum(d_net + load, 0).sum() - base_imp
                if best is None or extra < best[1]:
                    best = (h0, extra)
            if best is None:
                continue
            rows.append({"date": idx[a].date(), "month": idx[a].month, "kiln_kw": p, "firing_kwh": need,
                         "best_start": best[0], "grid_kwh": max(best[1], 0.0),
                         "free": best[1] <= tolerance * need})
    return pd.DataFrame(rows)


def kiln_summary(days_nb: pd.DataFrame, days_b: pd.DataFrame, price: float, label: str) -> pd.DataFrame:
    out = []
    for p, g in days_nb.groupby("kiln_kw"):
        gb = days_b[days_b.kiln_kw == p] if days_b is not None and len(days_b) else None
        row = {"kiln_kw": p, "firing_kwh": g.firing_kwh.iloc[0],
               "free_days_no_battery": int(g.free.sum()),
               "avg_grid_kwh_per_firing_no_battery": g.grid_kwh.mean(),
               "avg_cost_per_firing_no_battery": g.grid_kwh.mean() * price}
        if gb is not None:
            row.update({f"free_days_{label}": int(gb.free.sum()),
                        f"avg_grid_kwh_per_firing_{label}": gb.grid_kwh.mean(),
                        f"avg_cost_per_firing_{label}": gb.grid_kwh.mean() * price,
                        "months_with_free_days": ",".join(str(m) for m in sorted(gb[gb.free].month.unique()))})
        out.append(row)
    return pd.DataFrame(out)
