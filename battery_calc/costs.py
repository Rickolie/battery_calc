"""One cost engine for contracts with and without a battery (section 5).

C_t = I_t × (p_t + m + EB + grid_var) × (1 + VAT) − X_t × r_t + F_t

Saldering nets yearly totals; after 2027 import and export are settled
separately with a minimum feed-in compensation."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .contracts import Contract
from .taxes import Taxes


@dataclass
class Period:
    """Interval flows for one evaluated year (or part of one)."""
    imp: np.ndarray            # kWh per interval
    exp: np.ndarray
    is_low: np.ndarray         # bool, T1 (dal) intervals
    spot: np.ndarray | None    # €/kWh kale, None if no prices
    days: float                # length in days, for fixed charges
    tax_year: int

    def with_flows(self, imp, exp) -> "Period":
        return Period(imp, exp, self.is_low, self.spot, self.days, self.tax_year)


@dataclass
class CostResult:
    import_kwh: float
    export_kwh: float
    supply: float
    energy_tax: float
    feed_in_income: float
    feed_in_costs: float
    fixed_supplier: float
    grid: float
    tax_reduction: float
    bonus_amortised: float
    total: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_cost(period: Period, contract: Contract, regime: dict, taxes: Taxes, grid_label: str,
                 include_fixed: bool = True, extra_fee_per_year: float = 0.0) -> CostResult:
    vat = taxes.vat
    eb = taxes.energy_tax(period.tax_year)
    imp, exp, low = period.imp, period.exp, period.is_low
    I, X = float(imp.sum()), float(exp.sum())
    sal = bool(regime.get("saldering"))
    mf = float(regime.get("feed_in_min_frac", 0.0))
    fi_vat = (1 + vat) if regime.get("feed_in_incl_vat") else 1.0
    frac = period.days / 365.0

    if contract.is_dynamic:
        if period.spot is None:
            raise ValueError("Dynamic contract needs spot prices")
        p = period.spot
        buy = p + contract.markup
        sell = p + contract.feed_in_markup
        if sal:
            supply = float((imp * buy).sum())
            income = float((exp * sell).sum())
            eb_kwh = max(0.0, I - X)
            fi_vat = 1 + vat          # netted kWh are settled incl. VAT
        else:
            supply = float((imp * buy).sum())
            sell = np.maximum(sell, mf * buy) if mf > 0 else sell
            income = float((exp * sell).sum())
            eb_kwh = I
    else:
        p1, p2 = contract.kale_price(True), contract.kale_price(False)
        I1, I2 = float(imp[low].sum()), float(imp[~low].sum())
        X1, X2 = float(exp[low].sum()), float(exp[~low].sum())
        fi = contract.feed_in or 0.0
        if not sal and contract.post2027_feed_in is not None:
            fi = contract.post2027_feed_in
        if sal:
            n1, n2 = I1 - X1, I2 - X2
            if n1 + n2 <= 0:
                b1 = b2 = 0.0
                surplus = -(n1 + n2)
            elif n1 < 0:
                b1, b2, surplus = 0.0, n2 + n1, 0.0
            elif n2 < 0:
                b1, b2, surplus = n1 + n2, 0.0, 0.0
            else:
                b1, b2, surplus = n1, n2, 0.0
            supply = b1 * p1 + b2 * p2
            income = surplus * fi
            eb_kwh = max(0.0, I - X)
        else:
            supply = I1 * p1 + I2 * p2
            income = X1 * max(fi, mf * p1) + X2 * max(fi, mf * p2)
            eb_kwh = I

    supply *= (1 + vat)
    energy_tax = eb_kwh * eb * (1 + vat)
    income *= fi_vat
    annual_export = X / frac if frac > 0 else X
    fic = (contract.feed_in_cost_kwh * X
           + contract.feed_in_cost_year(annual_export, post2027=not sal) * frac) * (1 + vat)
    if not sal:
        fic = min(fic, max(income, 0.0))  # net payment for export may not go below zero
    if include_fixed:
        fixed = contract.fixed_eur_year * frac * (1 + vat)
        grid = taxes.grid_eur_year(grid_label) * frac * (1 + vat)
        red = -taxes.tax_reduction(period.tax_year) * frac * (1 + vat)
        years = max(contract.duration_months / 12.0, 1.0)
        bonus = (-contract.welcome_bonus / years + extra_fee_per_year) * frac
    else:
        fixed = grid = red = bonus = 0.0
    total = supply + energy_tax - income + fic + fixed + grid + red + bonus
    return CostResult(I, X, supply, energy_tax, income, fic, fixed, grid, red, bonus, total)


def marginal_values(period: Period, contract: Contract, regime: dict, taxes: Taxes,
                    net_importer: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per-interval value of avoiding 1 kWh import (u) and of exporting 1 kWh (s),
    in € incl. VAT. Used by battery strategies and the break-even."""
    vat = taxes.vat
    eb = taxes.energy_tax(period.tax_year)
    sal = bool(regime.get("saldering"))
    mf = float(regime.get("feed_in_min_frac", 0.0))
    fi_vat = (1 + vat) if regime.get("feed_in_incl_vat") else 1.0
    fic = contract.feed_in_cost_kwh * (1 + vat)
    n = len(period.imp)
    if contract.is_dynamic:
        p = period.spot
        buy = p + contract.markup
        u = (buy + eb) * (1 + vat)
        if sal and net_importer:
            s = (p + contract.feed_in_markup + eb) * (1 + vat) - fic
        elif sal:
            s = (p + contract.feed_in_markup) * (1 + vat) - fic
        else:
            sell = p + contract.feed_in_markup
            if mf > 0:
                sell = np.maximum(sell, mf * buy)
            s = sell * fi_vat - fic
    else:
        kale = np.where(period.is_low, contract.kale_price(True), contract.kale_price(False))
        u = (kale + eb) * (1 + vat)
        fi = contract.feed_in or 0.0
        if not sal and contract.post2027_feed_in is not None:
            fi = contract.post2027_feed_in
        if sal and net_importer:
            s = u - fic
        elif sal:
            s = np.full(n, fi * (1 + vat) - fic)
        else:
            s = np.maximum(fi, mf * kale) * fi_vat - fic
    return np.asarray(u, dtype=float), np.asarray(s, dtype=float)
