"""Objective 1: break-even sell price per battery (section 6)."""
from __future__ import annotations

import math

from .battery import Battery


def lifetime_kwh(b: Battery, cycles_per_year: float, calendar_years: float | None = None) -> tuple[float, str]:
    """AC kWh delivered over life and which limit sets it."""
    cal = calendar_years if calendar_years is not None else (b.warranty_years or 10.0)
    avg = b.usable_kwh * (1 + b.eol_capacity) / 2.0
    cands = {"cycles": (b.cycle_life or math.inf) * avg,
             "calendar life": cycles_per_year * cal * avg}
    if b.warranty_mwh:
        cands["warranty throughput"] = b.warranty_mwh * 1000.0
    limit = min(cands, key=cands.get)
    return cands[limit] * math.sqrt(b.rte), limit


def wear_cost(b: Battery, price: float, cycles_per_year: float, residual: float = 0.0,
              calendar_years: float | None = None) -> tuple[float, float, str]:
    life, limit = lifetime_kwh(b, cycles_per_year, calendar_years)
    if not life or math.isinf(life):
        return 0.0, life, limit
    return max(price - residual, 0.0) / life, life, limit


def breakeven(charge_price: float, rte: float, wear: float) -> float:
    """Minimum value of a delivered kWh for a stored kWh to be worth it."""
    return charge_price / rte + wear


def breakeven_rows(batteries, defaults: dict, avg_import: float, solar_2026: float, solar_2027: float,
                   cycles: dict | None = None, de_travel: float = 0.0, bf: dict | None = None) -> list[dict]:
    rows = []
    cycles = cycles or {}
    for b in batteries:
        cpy = cycles.get(b.id, float(defaults.get("cycles_per_year", 250)))
        variants = b.price_variants(de_travel, bf)
        if not variants:
            variants = {"no price": None}
        for vname, price in variants.items():
            if price is None:
                w, life, limit = float("nan"), *lifetime_kwh(b, cpy)
            else:
                w, life, limit = wear_cost(b, price, cpy, float(defaults.get("residual_value", 0.0)))
            rows.append({
                "battery_id": b.id, "battery": b.name, "price_variant": vname,
                "purchase_eur": price, "usable_kwh": round(b.usable_kwh, 2), "rte": b.rte,
                "cycles_per_year": round(cpy, 1), "lifetime_kwh": round(life, 0), "lifetime_limit": limit,
                "wear_eur_kwh": w,
                "be_grid_0": breakeven(0.0, b.rte, w),
                "be_grid_avg": breakeven(avg_import, b.rte, w),
                "be_own_solar_2026": breakeven(solar_2026, b.rte, w),
                "be_own_solar_2027": breakeven(solar_2027, b.rte, w),
                "estimated": ",".join(b.estimated_fields),
            })
    return rows
