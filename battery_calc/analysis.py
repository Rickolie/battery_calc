"""Runs the workflow of section 10 and produces report sections.

Each section is handed to `on_section` as soon as it is computed, so the
CLI can print progress and the website can show results step by step."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import charts, extras
from .battery import (BEST_PRICE, GRID_CHARGE, SELF, ZERO_IMPORT, Battery, curtail, in_scope, load_batteries,
                      lp_available,
                      perfect_foresight, plan_dynamic, plan_forecast, plan_hbc, plan_timed, simulate)
from .breakeven import breakeven_rows, wear_cost
from .report import slug
from .config import Connection, resolve
from .contracts import Contract, load_contracts_csv, parse_vast, read_text
from .costs import Period, compute_cost, marginal_values
from .p1 import FLAG_SYNTH, build_profile_year, load_p1, load_pv, monthly_summary, tariff_is_low
from .prices import crosscheck, load_energieknl, load_price_history, price_overview, replay_prices
from .taxes import Taxes, regime_for_year

SCENARIO_TAX_YEAR = {"saldering": 2026, "nosal_min50": 2027, "nosal_2030": 2030}


@dataclass
class Section:
    id: str
    title: str
    md: str = ""
    tables: dict = field(default_factory=dict)
    figures: dict = field(default_factory=dict)
    interactive: dict = field(default_factory=dict)   # name -> chart spec (zoomable in HTML; PNG in Markdown)
    csv_only: set = field(default_factory=set)       # table names offered as CSV download, not printed
    summary: str = ""        # key finding next to the collapsed title (default: first table's finding)


@dataclass
class Inputs:
    """Where the data comes from. Paths or file-like objects; None = config."""
    p1: object = None
    pv: object = None
    fixed_contract_text: str | None = None
    fixed_contract: Contract | None = None
    no_current_contract: bool = False
    price_sources: dict | None = None
    contracts_csv: object = None
    batteries_csv: object = None
    energieknl: object = None
    label: str = "Rick's data"


@dataclass
class Options:
    connection: str = "3x25"
    margin: float = 0.2
    price_variant: str = "all"         # all | NL current | NL lowest-ever | DE 0% VAT (scenario)
    quick: bool = False                # headline years only, fewer variants
    plots: bool = True
    feed_in_2030: float | None = None  # override the 2030+ minimum fraction
    chosen_battery: str | None = None  # Objective 4 / earnings split; None = config or best payback
    battery_set: str | list | None = None  # all | shortlist | [ids]; None = config
    kiln: dict | None = None           # overrides for the kiln section of config.yaml


class Analysis:
    def __init__(self, cfg: dict, inputs: Inputs, opts: Options, on_section=None, log=print):
        self.cfg = cfg
        self.inp = inputs
        self.opts = opts
        self.on_section = on_section
        self.log = log or (lambda *a: None)
        self.sections: list[Section] = []
        self.taxes = Taxes(cfg)
        self.conn = Connection.parse(opts.connection, opts.margin, cfg.get("connection", {}).get("volts", 230))
        self.regimes = dict(cfg["regimes"])
        if opts.feed_in_2030 is not None:
            self.regimes["nosal_2030"] = {**self.regimes["nosal_2030"], "feed_in_min_frac": opts.feed_in_2030}
        self.scenarios = list(self.regimes)
        self.results: dict = {}
        # German prices on: every calculation uses the cheaper of today's NL and DE price.
        self.use_de = bool(cfg.get("battery_defaults", {}).get("use_german_prices", False))
        self.ref_variant = BEST_PRICE if self.use_de else "NL current"

    # ------------------------------------------------------------------ utils
    def emit(self, sec: Section):
        self.sections.append(sec)
        if self.on_section:
            self.on_section(sec)

    def path(self, key):
        return resolve(self.cfg, self.cfg["paths"].get(key))

    def fig(self, fn, *a, **k):
        if not self.opts.plots:
            return None
        try:
            return fn(*a, **k)
        except Exception as e:  # charts are never allowed to break a run
            self.log(f"chart failed: {e}")
            return None

    # ------------------------------------------------------------------ run
    def run(self) -> list[Section]:
        self.step_load()
        self.step_power()
        self.step_current_contract()
        self.step_contracts()
        self.step_batteries()
        self.step_breakeven()
        self.step_payback()          # also emits section 8 (break-even with simulated cycles)
        self.step_sanity()
        self.step_kiln()
        self.step_solar()
        self.step_extension()
        self.step_advice()
        return self.sections

    # ------------------------------------------------------------------ step 1
    def step_load(self):
        cfg = self.cfg
        p1_src = self.inp.p1 or self.path("p1_file")
        df, rep = load_p1(p1_src, cfg, label=getattr(p1_src, "name", str(p1_src)))
        profile = build_profile_year(df, rep, cfg)
        self.raw, self.p1rep, self.profile = df, rep, profile
        self.real_mask = (profile["flag"] != FLAG_SYNTH).values
        tot_i, tot_x = profile["imp"].sum(), profile["exp"].sum()
        self.net_importer = tot_i >= tot_x

        pv_src = self.inp.pv
        if pv_src is None:
            p = self.path("pv_file")
            pv_src = p if p and os.path.exists(p) else None
        self.pv = None
        if pv_src is not None:
            try:
                self.pv = load_pv(pv_src, cfg).reindex(profile.index)
            except Exception as e:
                rep.notes.append(f"PV file could not be read: {e}")

        self.prices, self.prep = load_price_history(cfg, self.taxes, cfg.get("_base_dir", "."),
                                                    self.inp.price_sources)
        knl_src = self.inp.energieknl
        if knl_src is None:
            p = self.path("energieknl_file")
            knl_src = p if p and os.path.exists(p) else None
        self.knl = load_energieknl(knl_src) if knl_src is not None else None
        self.prep.crosscheck = crosscheck(self.prices, self.knl,
                                          cfg.get("prices", {}).get("crosscheck_threshold_eur_kwh", 0.001))

        a = cfg["analysis"]
        years = self.prep.years if self.prices is not None else []
        full_lo, full_hi = a["full_history_years"]
        head_lo, head_hi = a["headline_years"]
        self.partial_years = set(self.prep.partial_years)
        self.full_years = [y for y in years if full_lo <= y <= full_hi]
        self.head_years = [y for y in years if head_lo <= y <= head_hi and y not in self.partial_years]
        if self.opts.quick:
            self.full_years = list(self.head_years)

        real_i = profile.loc[self.real_mask, "imp"].sum()
        real_x = profile.loc[self.real_mask, "exp"].sum()
        md = [f"**{self.inp.label}** · {self.conn.describe()}",
              f"- Profile year {rep.window[0]:%Y-%m-%d} to {rep.window[1]:%Y-%m-%d}: import **{tot_i:,.0f} kWh**, "
              f"export **{tot_x:,.0f} kWh** (real months only: {real_i:,.0f} / {real_x:,.0f})."]
        if rep.missing_months:
            md.append(f"- **Filled synthetically:** {', '.join(rep.missing_months)} (blend of the two weeks before "
                      "and after).")
        if rep.low_confidence:
            md.append("- **Low confidence:** more than 3 months are synthetic.")
        md.append("#### Meter file details")
        md.append(f"- P1 file `{rep.source}`: {rep.rows} rows, {rep.first:%Y-%m-%d %H:%M} to {rep.last:%Y-%m-%d %H:%M} "
                  f"(Europe/Amsterdam, DST handled).")
        md.append(f"- Duplicates removed: {rep.duplicates}; counter resets: {len(rep.resets)}; "
                  f"gaps ≤1 h interpolated: {len(rep.interpolated_gaps)}; longer gaps flagged and filled: "
                  f"{len(rep.long_gaps)}.")
        for g in rep.long_gaps[:10]:
            md.append(f"  - long gap {g[0]:%Y-%m-%d %H:%M} → {g[1]:%Y-%m-%d %H:%M}")
        if rep.tariff_rule_agreement is not None:
            md.append(f"- T1/T2 from counters matches the configured tariff-hour rule in "
                      f"{rep.tariff_rule_agreement:.1%} of intervals (rule used only for synthetic intervals).")
        for s in rep.partially_real_months:
            md.append(f"  - {s}")
        if self.pv is not None:
            gross = (profile["imp"] - profile["exp"] + self.pv.fillna(0)).sum()
            md.append(f"- PV data present: production {self.pv.sum():,.0f} kWh, gross consumption {gross:,.0f} kWh.")
        else:
            md.append("- No PV file: the analysis uses net flows only (enough for contracts and batteries; "
                      "gross consumption and the solar-forecast Charge goal are not available).")
        for n in rep.notes:
            md.append(f"- {n}")
        md.append("#### Price history")
        if self.prices is None:
            md += [f"- {n}" for n in self.prep.notes]
        else:
            md.append(f"- Files: {len(self.prep.files)}; duplicates at year boundaries removed: "
                      f"{self.prep.duplicates_removed}.")
            for f, d in self.prep.columns_used.items():
                md.append(f"  - `{f}`: {d}")
            md.append("- Missing hours per year: " + ", ".join(f"{y}: {h:.0f}" for y, h in self.prep.missing_hours.items()))
            for y, months in self.prep.partial_years.items():
                md.append(f"- {y} is a partial year (months {months[0]}–{months[-1]}); used for its months only.")
            ov = self.prices.loc[self.profile.index[0]:self.profile.index[-1]]
            if len(ov):
                md.append(f"- Overlap of meter and price data (validation run): {ov.index[0]:%Y-%m-%d} to "
                          f"{ov.index[-1]:%Y-%m-%d}.")
            else:
                md.append("- Meter and price data do not overlap; all price years are replayed.")
            for n in self.prep.notes:
                md.append(f"- {n}")
        md.append(f"- Cross-check: {self.prep.crosscheck}")
        sec = Section("data", "1. Your data", "\n".join(md))
        monthly = monthly_summary(profile)
        sec.tables["monthly_flows"] = monthly.round(3)
        f = self.fig(charts.monthly_flows, monthly)
        if f:
            sec.figures["monthly_flows"] = f
        if self.prices is not None:
            ov = price_overview(self.prices)
            self.price_ov = ov
            sec.tables["price_overview"] = ov.round(4)
            f = self.fig(charts.price_overview, ov)
            if f:
                sec.figures["price_overview"] = f
        if self.knl is not None and "leveringskost" in [c.lower() for c in self.knl.columns]:
            k = self.knl.copy()
            k["leveringskost"] = pd.to_numeric(k["leveringskost"].str.replace(",", "."), errors="coerce")
            latest = k[k["product"] == "stroom"].sort_values(["jaar", "maand"]).groupby("leverancier").tail(1)
            sec.tables["energieknl_reference"] = latest[["leverancier", "jaar", "maand", "leveringskost"]] \
                .rename(columns={"leveringskost": "leveringskost_eur_year"}).reset_index(drop=True)
        self.emit(sec)

    # ------------------------------------------------------------------ periods
    def profile_period(self, scenario: str, spot=None, mask=None) -> Period:
        if spot is None and mask is None:
            key = ("pperiod", scenario)
            if key not in self.results:
                self.results[key] = self._profile_period(scenario)
            return self.results[key]
        return self._profile_period(scenario, spot, mask)

    def _profile_period(self, scenario: str, spot=None, mask=None) -> Period:
        pr = self.profile
        sel = slice(None) if mask is None else mask
        imp = pr["imp"].values[sel]
        return Period(imp, pr["exp"].values[sel], pr["is_low"].values[sel], spot, len(imp) / 96.0,
                      SCENARIO_TAX_YEAR.get(scenario, 2027))

    def replay(self, year: int, scenario: str):
        """Returns (Period, target_index, real_mask) for a price year."""
        key = ("replay", year)
        if key not in self.results:
            r = replay_prices(self.profile.index, self.prices, year)
            if r is None:
                self.results[key] = None
            else:
                idx, pos, spot = r
                pr = self.profile
                self.results[key] = (idx, pos, spot, tariff_is_low(idx, self.cfg),
                                     pr["imp"].values[pos], pr["exp"].values[pos], self.real_mask[pos])
        v = self.results[key]
        if v is None:
            return None
        idx, pos, spot, low, imp, exp, real = v
        pkey = ("rperiod", year, scenario)
        if pkey not in self.results:
            self.results[pkey] = Period(imp, exp, low, spot, len(idx) / 96.0, SCENARIO_TAX_YEAR.get(scenario, 2027))
        return self.results[pkey], idx, real

    def periods(self, contract: Contract, scenario: str, years=None):
        """[(label, Period, index, real_mask)] for a contract and scenario."""
        if not contract.is_dynamic:
            return [("profile", self.profile_period(scenario), self.profile.index, self.real_mask)]
        out = []
        for y in (self.full_years if years is None else years):
            r = self.replay(y, scenario)
            if r is not None:
                out.append((y, r[0], r[1], r[2]))
        return out

    # ------------------------------------------------------------------ step 2
    def load_contracts(self):
        cur = None
        facts = {}
        if self.inp.fixed_contract is not None:
            cur = self.inp.fixed_contract
        elif not self.inp.no_current_contract:
            text = self.inp.fixed_contract_text
            if text is None:
                p = self.path("fixed_contract_file")
                text = read_text(p) if p and os.path.exists(p) else None
            if text:
                cur, facts = parse_vast(text)
        others = []
        src = self.inp.contracts_csv or self.path("contracts_file")
        if src is not None and (not isinstance(src, str) or os.path.exists(src)):
            others = load_contracts_csv(src)
        self.current = cur
        self.contracts = ([cur] if cur else []) + others
        self.facts = facts

    def step_current_contract(self):
        self.load_contracts()
        md = []
        sec = Section("current", "3. Current contract")
        if self.current is None:
            md.append("No current contract given: it is left out of the comparison.")
            sec.md = "\n".join(md)
            self.emit(sec)
            return
        c = self.current
        md.append(f"**{c.label}** ({c.type}, {c.start_date or '?'} → {c.end_date or '?'}).")
        md.append("`yearly cost = energy + energy tax + fixed costs + grid charges − tax reduction − feed-in income "
                  "+ feed-in costs` (incl. VAT, on your profile year)")
        md.append("#### Tariff details")
        md.append(f"- Source: `{c.source}`.")
        md.append(f"- Kale prices excl. VAT: T1/dal €{c.kale_price(True):.5f}, T2/normaal €{c.kale_price(False):.5f}, "
                  f"single €{(c.price_single or 0):.5f}; feed-in €{(c.feed_in or 0):.5f}/kWh; fixed €{c.fixed_eur_year:.2f}/yr.")
        if c.feed_in_cost_tiers:
            md.append("- Feed-in cost tiers (€/yr excl. VAT by yearly export): " +
                      ", ".join(f"{lo:g}–{'∞' if math.isinf(hi) else f'{hi:g}'}: {eur:g}" for lo, hi, eur in c.feed_in_cost_tiers))
        if "energy_tax_incl_vat_implied" in self.facts:
            implied = self.facts["energy_tax_incl_vat_implied"] / (1 + self.taxes.vat)
            md.append(f"- Energy tax implied by the tariff sheet: €{implied:.5f}/kWh excl. VAT; config 2026: "
                      f"€{self.taxes.energy_tax(2026):.5f} ({self.taxes.source(2026)}).")
        if "grid_eur_year_excl_vat" in self.facts:
            md.append(f"- Grid charges on the sheet: €{self.facts['grid_eur_year_excl_vat']:.2f}/yr excl. VAT "
                      f"({self.facts.get('grid_operator', '')}); config for {self.conn.label}: "
                      f"€{self.taxes.grid_eur_year(self.conn.label):.2f}.")
        rows = []
        for scn in self.scenarios:
            full = compute_cost(self.profile_period(scn), c, self.regimes[scn], self.taxes, self.conn.label)
            real = compute_cost(self.profile_period(scn, mask=self.real_mask), c, self.regimes[scn], self.taxes,
                                self.conn.label)
            d = {"scenario": self.regimes[scn]["label"], **{k: round(v, 2) for k, v in full.as_dict().items()},
                 "total_real_months_only": round(real.total, 2)}
            rows.append(d)
        sec.tables["current_contract_cost"] = pd.DataFrame(rows)
        md.append("#### Checks")
        md.append("- `total_real_months_only` is the cost without the synthetic month(s).")
        md.append("- **Acceptance check:** compare `total` under 2026 rules with an actual annual bill "
                  "(target: within 3%). No bill is in the repo, so this check is still open.")
        for w in self.taxes.warnings:
            md.append(f"- {w}")
        sec.md = "\n".join(md)
        self.emit(sec)

    # ------------------------------------------------------------------ step 3
    def contract_year_costs(self, c: Contract, scn: str, years=None) -> dict:
        key = ("ccost", c.id, scn, tuple(years) if years else None)
        if key in self.results:
            return self.results[key]
        out = {}
        for label, per, idx, real in self.periods(c, scn, years):
            full = compute_cost(per, c, self.regimes[scn], self.taxes, self.conn.label)
            rp = Period(per.imp[real], per.exp[real], per.is_low[real],
                        None if per.spot is None else per.spot[real], real.sum() / 96.0, per.tax_year)
            realc = compute_cost(rp, c, self.regimes[scn], self.taxes, self.conn.label)
            out[label] = (full, realc)
        self.results[key] = out
        return out

    def term_cost(self, c: Contract, expected: dict) -> float:
        """Average yearly cost over the contract term, split by the rules per year."""
        start = pd.Timestamp(self.cfg["analysis"]["purchase_date"])
        end = start + pd.DateOffset(months=max(c.duration_months, 1))
        if c.end_date and pd.Timestamp(c.end_date) > start:
            end = pd.Timestamp(c.end_date)     # running contract: remaining term only
        total, t = 0.0, start
        while t < end:
            ny = pd.Timestamp(year=t.year + 1, month=1, day=1)
            seg_end = min(ny, end)
            frac = (seg_end - t).days / 365.0
            scn = regime_for_year(self.cfg, t.year)
            total += expected.get(scn, np.nan) * frac
            t = seg_end
        years = (end - start).days / 365.0
        total -= c.welcome_bonus
        return total / years if years else np.nan

    def step_contracts(self):
        sec = Section("contracts", "4. Cheapest contract without a battery")
        md = []
        if not self.contracts:
            sec.md = "No contracts to compare."
            self.emit(sec)
            return
        dyn_ok = self.prices is not None and bool(self.full_years)
        analyses = {"Last 3 years (headline)": self.head_years, "Full history": self.full_years}
        cur_expected = {}
        all_rows = {}
        expected_by_contract = {}
        for c in self.contracts:
            if c.is_dynamic and not dyn_ok:
                continue
            expected_by_contract[c.id] = {}
            for aname, years in analyses.items():
                for scn in self.scenarios:
                    costs = self.contract_year_costs(c, scn)
                    if c.is_dynamic:
                        sel = {y: v for y, v in costs.items() if y in years and y not in self.partial_years}
                        if not sel:
                            continue
                    else:
                        sel = costs
                    # The switch bonus is one-off: yearly figures exclude it; it is counted
                    # once in the cost over the contract term.
                    tot = np.array([v[0].total - v[0].bonus_amortised for v in sel.values()])
                    real = np.array([v[1].total - v[1].bonus_amortised for v in sel.values()])
                    imp = np.array([v[0].import_kwh for v in sel.values()])
                    inc = np.array([v[0].feed_in_income for v in sel.values()])
                    row = {"contract": c.label, "type": c.type, "verified": "yes" if c.verified else "no",
                           "expected_eur_year": tot.mean(), "min": tot.min(), "max": tot.max(),
                           "switch_bonus_eur_once": c.welcome_bonus,
                           "eur_per_kwh_imported": (tot / np.maximum(imp, 1e-9)).mean(),
                           "feed_in_income": inc.mean(), "real_months_only": real.mean(),
                           "price_years": ",".join(str(k) for k in sel) if c.is_dynamic else "stated tariffs"}
                    all_rows.setdefault((aname, scn), []).append(row)
                    if aname.startswith("Last") or not c.is_dynamic:
                        expected_by_contract[c.id][scn] = tot.mean()
                    if self.current is not None and c.id == self.current.id:
                        cur_expected[scn] = tot.mean()
        for (aname, scn), rows in all_rows.items():
            df = pd.DataFrame(rows).sort_values("expected_eur_year")
            if scn in cur_expected:
                df["diff_vs_current"] = df["expected_eur_year"] - cur_expected[scn]
            for c in self.contracts:
                if c.id in expected_by_contract:
                    df.loc[df["contract"] == c.label, "avg_eur_year_over_term"] = self.term_cost(c, expected_by_contract[c.id])
            sec.tables[f"{aname} – {self.regimes[scn]['label']}"] = df.round(2).reset_index(drop=True)
        if dyn_ok:
            groups = self.cfg["analysis"].get("year_groups", {})
            grows = []
            for c in self.contracts:
                if not c.is_dynamic:
                    continue
                for scn in self.scenarios:
                    costs = self.contract_year_costs(c, scn)
                    for gname, (lo, hi) in groups.items():
                        vals = [v[0].total for y, v in costs.items() if lo <= y <= hi and y not in self.partial_years]
                        if vals:
                            grows.append({"contract": c.label, "scenario": self.regimes[scn]["label"], "group": gname,
                                          "years": f"{lo}–{hi}", "expected": np.mean(vals), "min": min(vals),
                                          "max": max(vals)})
                    for y in self.partial_years:
                        if y in costs:
                            grows.append({"contract": c.label, "scenario": self.regimes[scn]["label"],
                                          "group": f"{y} partial", "years": str(y),
                                          "expected": costs[y][0].total, "min": np.nan, "max": np.nan})
            if grows:
                sec.tables["Full history by year group"] = pd.DataFrame(grows).round(2)
            # chart: the 3 cheapest dynamic contracts per price year vs the cheapest fixed offer and the
            # current fixed contract (flat: stated tariffs), 2027 rules; one-off bonuses excluded
            scn = "nosal_min50"
            dyn = [c for c in self.contracts if c.is_dynamic]
            dyn.sort(key=lambda c: expected_by_contract.get(c.id, {}).get(scn, np.inf))
            series = {}
            for c in dyn[:3]:
                costs = self.contract_year_costs(c, scn)
                series[c.label] = pd.Series({y: v[0].total - v[0].bonus_amortised for y, v in sorted(costs.items())
                                             if y not in self.partial_years})
            fixed = [c for c in self.contracts if not c.is_dynamic and c.id in expected_by_contract
                     and expected_by_contract[c.id].get(scn) is not None]
            offers = [c for c in fixed if c is not self.current] or fixed
            best_fixed = min(offers, key=lambda c: expected_by_contract[c.id][scn]) if offers else None
            f = self.fig(charts.contract_years, series, best_fixed.label if best_fixed else None,
                         expected_by_contract[best_fixed.id].get(scn) if best_fixed else None,
                         "Yearly cost per price year (2027–2029 rules)")
            if f:
                sec.figures["contract_years"] = f
            years = sorted({int(y) for s in series.values() for y in s.index})
            if years:
                pal = ["#2a78d6", "#eb6834", "#1baf7a"]
                lines = [{"label": f"{lab} (dynamic)", "short": lab.split(" – ")[0], "color": pal[i],
                          "values": [round(float(s.get(y)), 0) if y in s.index else None for y in years]}
                         for i, (lab, s) in enumerate(series.items())]
                if best_fixed is not None:
                    v = expected_by_contract[best_fixed.id][scn]
                    lines.append({"label": f"{best_fixed.label} (cheapest fixed offer)", "short": "fixed offer",
                                  "color": "#4a3aa7", "values": [round(v, 0)] * len(years)})
                if self.current is not None and self.current is not best_fixed and self.current.id in expected_by_contract:
                    v = expected_by_contract[self.current.id].get(scn)
                    if v is not None:
                        lines.append({"label": f"{self.current.label} (current, ends 2027)",
                                      "values": [round(v, 0)] * len(years), "context": True})
                sec.interactive["contract_years"] = {
                    "kind": "lines", "unit": "€", "yLabel": "€ per year", "xFormat": "year", "x": years,
                    "title": "Yearly cost without a battery: cheapest dynamic contracts vs fixed",
                    "subtitle": "Dynamic contracts replayed on each year's day-ahead prices under the 2027–2029 "
                                "rules; fixed contracts cost the same every year (stated tariffs). One-off bonuses "
                                "excluded.",
                    "endLabels": True, "contextLabel": "current fixed contract (ends 2027)", "series": lines,
                    "height": 340}
        else:
            md.append("**Dynamic contracts are skipped:** no day-ahead price history was loaded.")
        md = ["`dynamic contract cost = Σ quarter-hours (spot + markup + energy tax) × import × 1.21 − feed-in income "
              "+ fixed + grid − tax reduction`",
              "Each dynamic contract is replayed on every price year (2013–2025) under each set of rules; "
              "fixed contracts use their stated tariffs."] + md
        unverified = [c.label for c in self.contracts if not c.verified]
        md.append("#### How to read the tables")
        md.append("- `expected_eur_year`: average over the price years in the table (last 3 years = headline).")
        md.append("- `avg_eur_year_over_term`: cost spread over the contract term from the purchase date, switching "
                  "rules on 1 January 2027, minus the one-off switch bonus. Yearly figures exclude the bonus.")
        md.append("- `real_months_only` excludes synthetic months.")
        if unverified:
            md.append(f"- Unverified contract terms (hand-entered or template): {', '.join(unverified)}.")
        self.expected_by_contract = expected_by_contract
        fvd = self.fixed_vs_dynamic_table()
        if fvd is not None:
            sec.tables = {"fixed_vs_dynamic": fvd.round(0), **sec.tables}
            line = self.fixed_vs_dynamic_line()
            if line:
                md.insert(2, line)
                fx, dy = self.cheapest_fixed(), self.cheapest_dynamic()
                sec.summary = (f"From 2027 the cheapest dynamic contract ({dy.label}) costs about "
                               f"€{self.cost27(dy):,.0f} a year; the cheapest fixed offer ({fx.label}) "
                               f"€{self.cost27(fx):,.0f}, €{self.cost27(fx) - self.cost27(dy):,.0f} more.")
        fv = self.feed_in_value_table()
        if fv is not None:
            sec.tables["feed_in_value_per_year"] = fv
            last = fv.iloc[-1]
            md.append("#### Why feed-in earns little on a dynamic contract")
            md.append(f"- You export mostly around midday, when "
                      f"solar pushes day-ahead prices down. In {int(last.year)} the average spot price was "
                      f"€{last.avg_spot_eur_kwh:.3f}/kWh, but only €{last.avg_spot_when_exporting:.3f} in the quarter-hours "
                      f"you export, and {last.export_at_negative_price_pct:.0f}% of your export fell in negative-price "
                      f"hours (those cost money). See `feed_in_value_per_year`.")
        sec.md = "\n".join(md)
        self.emit(sec)

    def feed_in_value_table(self):
        """Export-weighted spot price per price year on the cheapest dynamic contract (2027 rules)."""
        c = self.cheapest_dynamic()
        if c is None:
            return None
        rows = []
        for y in self.full_years:
            if y in self.partial_years:
                continue
            r = self.replay(y, "nosal_min50")
            if r is None:
                continue
            per = r[0]
            e, p = per.exp, per.spot
            w = e.sum()
            if w <= 0:
                continue
            res = compute_cost(per, c, self.regimes["nosal_min50"], self.taxes, self.conn.label, include_fixed=False)
            rows.append({"year": y, "export_kwh": w, "avg_spot_eur_kwh": p.mean(),
                         "avg_spot_when_exporting": (e * p).sum() / w,
                         "export_at_negative_price_pct": 100 * e[p < 0].sum() / w,
                         "feed_in_income_eur": res.feed_in_income,
                         "feed_in_income_per_kwh": res.feed_in_income / w})
        return pd.DataFrame(rows).round(3) if rows else None

    # ------------------------------------------------------------------ step 4
    def step_batteries(self):
        sec = Section("batteries", "5. Battery options")
        src = self.inp.batteries_csv or self.path("batteries_file")
        bats = load_batteries(src, self.cfg.get("battery_defaults", {})) if src else []
        bset = self.opts.battery_set if self.opts.battery_set is not None else self.cfg.get("battery_set", "all")
        if isinstance(bset, str) and bset != "all":
            bset = [x.strip() for x in bset.split(",") if x.strip()]
        if isinstance(bset, list) and "shortlist" in bset:
            bset = [x for x in bset if x != "shortlist"] + list(self.cfg.get("battery_shortlist", []))
        if isinstance(bset, list) and bset:
            bats = [b for b in bats if b.id in bset]
        rows = []
        self.batteries, self.excluded = [], []
        for b in bats:
            ok, why = in_scope(b)
            cap = self.conn.battery_cap_w(b.phases)
            note = why
            if cap is None:
                ok = False
                note = f"three-phase battery excluded on single-phase connection {self.conn.label}"
            if ok:
                self.batteries.append(b)
            else:
                self.excluded.append((b, note))
            # the price every calculation uses: NL, or the cheaper of NL/DE with German prices on
            dtravel = float(self.cfg.get("battery_defaults", {}).get("de_travel_cost_eur", 0.0))
            if self.use_de:
                ref_p, ref_src = b.best_price(dtravel)
            else:
                ref_p = None if b.price_nl is None else b.price_nl + b.extra_hardware
                ref_src = "NL" if ref_p is not None else ""
            rows.append({"id": b.id, "battery": b.name, "size_class": self.class_of(b)[0] or "other",
                         "type": b.type, "phases": b.phases,
                         "usable_kwh": round(b.usable_kwh, 2), "max_charge_w": b.max_charge_w,
                         "max_discharge_w": b.max_discharge_w,
                         "power_cap_w": None if cap is None else round(min(cap, b.max_charge_w, b.max_discharge_w)),
                         "rte": b.rte, "standby_w": b.standby_w, "cycle_life": b.cycle_life,
                         "eol_capacity": b.eol_capacity, "warranty_years": b.warranty_years,
                         "warranty_mwh": b.warranty_mwh, "price_nl": b.price_nl, "price_nl_lowest": b.price_nl_lowest,
                         "price_de": b.price_de, "extra_hw": b.extra_hardware,
                         "price_used_eur": ref_p, "bought_in": ref_src,
                         "eur_per_kwh_nominal": round(ref_p / b.nominal_kwh) if ref_p and b.nominal_kwh else None,
                         "eur_per_kwh_usable": round(ref_p / b.usable_kwh) if ref_p and b.usable_kwh else None,
                         "solar_during_outage": b.outage_solar or "unknown", "dc_solar_input_w": b.dc_solar_w,
                         "missing_fields": ", ".join(b.missing_fields), "estimated_with_defaults": ", ".join(b.estimated_fields),
                         "verified": "yes" if b.verified else "no", "in_scope": "yes" if ok else "no", "note": note})
        sec.tables["battery_specs"] = pd.DataFrame(rows)
        d = self.cfg.get("battery_defaults", {})
        counts = {}
        for b in self.batteries:
            counts[self.class_of(b)[0] or "other"] = counts.get(self.class_of(b)[0] or "other", 0) + 1
        md = [f"{len(bats)} batteries loaded, {len(self.batteries)} in scope for {self.conn.label}: "
              + ", ".join(f"{k}: {v}" for k, v in counts.items()) + ".",
              "`€ per usable kWh = price ÷ usable capacity` – the main price comparison between sizes.",
              "#### Missing specs and exclusions",
              f"- Missing specs are left empty and flagged; the simulation uses documented defaults "
              f"(RTE {d.get('rte')}, standby {d.get('standby_w')} W, cycle life {d.get('cycle_life')}, "
              f"EoL {d.get('eol_capacity')}, warranty {d.get('warranty_years')} yr) and marks the battery as estimated."]
        for b, why in self.excluded:
            md.append(f"- Excluded: {b.name} – {why}.")
        md.append("#### Solar during a power outage")
        md.append("- `solar_during_outage`: can solar power keep charging the battery when the grid (or the main "
                  "switch) is off? Only batteries with their own solar input can: DC panels on the battery's MPPT "
                  "inputs, or a micro-inverter on its backup port (Indevolt 2000 hybrid series). The existing roof "
                  "inverter on the house wiring stops when the grid is gone; keeping it running needs a battery "
                  "inverter that forms the grid for the house wiring behind an automatic transfer switch, with "
                  "frequency-shift power control of the solar inverter, installed by an electrician. None of these "
                  "plug-in batteries offers that. *not stated* = the shop only mentions a backup socket for devices.")
        nop = [b.name for b in self.batteries if not b.price_variants()]
        if nop:
            md.append(f"- No purchase price yet (payback cannot be computed): {', '.join(nop)}. "
                      "Run the scraper or fill `price_nl_incl_vat` in data/online/batteries.csv.")
        sec.md = "\n".join(md)
        self.emit(sec)

    # ------------------------------------------------------------------ step 5
    def avg_prices(self):
        """Average all-in import price and own-solar charge price (feed-in given up)."""
        c = self.current or next((x for x in self.contracts if not x.is_dynamic), None)
        if c is None and self.contracts:
            c = self.contracts[0]
        out = {}
        for scn in ("saldering", "nosal_min50"):
            pers = self.periods(c, scn, self.head_years) if c is not None else []
            if not pers:
                out[scn] = (0.25, 0.05)
                continue
            u_all, s_all = [], []
            for _, per, _, _ in pers:
                u, s = marginal_values(per, c, self.regimes[scn], self.taxes, self.net_importer)
                w = np.maximum(per.imp, 1e-6)
                u_all.append(np.average(u, weights=w))
                s_all.append(np.average(s, weights=np.maximum(per.exp, 1e-6)))
            out[scn] = (float(np.mean(u_all)), float(np.mean(s_all)))
        return out, c

    def wear_for(self, b: Battery, cycles: float | None = None) -> tuple[float, str]:
        d = self.cfg.get("battery_defaults", {})
        variants = b.price_variants(d.get("de_travel_cost_eur", 0.0), best=self.use_de)
        if not variants:
            return 0.0, "no price"
        name = self.ref_variant if self.ref_variant in variants else next(iter(variants))
        w, _, _ = wear_cost(b, variants[name], cycles or d.get("cycles_per_year", 250), d.get("residual_value", 0.0))
        return w, name

    def step_breakeven(self, cycles: dict | None = None, title_suffix: str = ""):
        avg, c = self.avg_prices()
        d = self.cfg.get("battery_defaults", {})
        rows = breakeven_rows(self.batteries, d, avg["saldering"][0], avg["saldering"][1], avg["nosal_min50"][1],
                              cycles=cycles, de_travel=d.get("de_travel_cost_eur", 0.0),
                              bf=self.cfg.get("blackfriday"), best=self.use_de)
        sid = "breakeven" if cycles is None else "breakeven_recomputed"
        sec = Section(sid, ("6. Break-even price per battery" if cycles is None
                            else "8. Break-even with simulated cycles") + title_suffix)
        df = pd.DataFrame(rows)
        if len(df):
            sec.tables["breakeven"] = df.round(4)
        md = ["`break-even price = charge price ÷ RTE + wear`",
              "`wear = purchase price ÷ kWh the battery delivers over its life`",
              "`minimum price difference = charge price × (1/RTE − 1) + wear`",
              "Discharging (or selling) a stored kWh only pays when it is worth at least the break-even price."]
        if cycles is not None:
            md.append("Same as section 6, but with each battery's cycles per year from the simulation (section 7) "
                      "instead of the default; more cycles spread the price over more kWh, so wear drops.")
        fp, dp = self.fixed_prices(), self.contract_prices(self.cheapest_dynamic())
        priced_nl = [r for r in rows if r["price_variant"] == self.ref_variant and r["purchase_eur"]
                     and not math.isnan(r["wear_eur_kwh"])]
        if fp is not None and dp is not None and priced_nl:
            r0 = min(priced_nl, key=lambda r: r["wear_eur_kwh"])
            parts = []
            for cc, u_, s_ in (fp, dp):
                be = s_ / r0["rte"] + r0["wear_eur_kwh"]
                parts.append(f"on {cc.label} a stored solar kWh replaces €{u_:.3f} of import and costs €{be:.3f} "
                             f"(feed-in given up €{s_:.3f} ÷ RTE + wear) → margin €{u_ - be:.3f}")
            md.append(f"**Fixed vs dynamic from 2027** ({r0['battery']}): " + "; ".join(parts) + ". A higher fixed "
                      "price makes each stored kWh worth more, which is why batteries pay back faster on fixed "
                      "contracts – but the fixed contract itself costs more (section 4). The dynamic margin is an "
                      "average: the forecast strategy picks hours where it is much larger.")
        md += self.breakeven_explainer(rows)
        md += ["#### Reference prices",
               f"- Contract {c.label if c else 'n/a'}: average all-in import price €{avg['saldering'][0]:.3f}/kWh; "
               f"own-solar charge price (feed-in given up) €{avg['saldering'][1]:.3f} under saldering and "
               f"€{avg['nosal_min50'][1]:.3f} from 2027.",
               "- Under saldering selling and using at home are worth the same; from 2027 the sell break-even is "
               "compared with the net feed-in price. Standby is a fixed cost (section 7).",
               "- `lifetime_limit` shows which limit (cycles, warranty throughput or calendar life) sets lifetime kWh."]
        if cycles is None:
            md.append(f"- Cycles per year: default {d.get('cycles_per_year', 250)} (simulated values in section 8).")
        sec.md = "\n".join(md)
        priced = [r for r in rows if r["purchase_eur"] is not None and not math.isnan(r["wear_eur_kwh"])
                  and r["price_variant"] == self.ref_variant]
        picks = []
        if cycles is not None and getattr(self, "focus", None):
            ids = {f["b"].id: f for f in self.focus}
            picks = [(ids[r["battery_id"]]["cls"], ids[r["battery_id"]]["color"], r) for r in priced
                     if r["battery_id"] in ids]
        else:
            for lab, lo, hi, color in self.size_classes():
                inc = [r for r in priced if lo <= self.size_of(self.battery(r["battery_id"])) < hi]
                if inc:
                    picks.append((lab, color, min(inc, key=lambda r: r["wear_eur_kwh"])))
        if picks:
            xs = [round(x * 0.02, 2) for x in range(0, 21)]
            sec.interactive["breakeven_lines"] = {
                "kind": "lines", "unit": "€", "yLabel": "break-even (€/kWh)", "x": xs,
                "xNames": [f"charging at €{x:.2f}/kWh" for x in xs], "xLabel": "charge price (€/kWh, all-in)",
                "title": "Break-even price vs charge price" + (" – best battery per size class" if cycles is not None
                                                                else " – lowest wear per size class"),
                "subtitle": "Above the line, discharging a stored kWh pays; the grey diagonal is the charge price "
                            "itself, the gap between them is the minimum price difference.",
                "endLabels": True, "markers": False, "contextLabel": "charge price (no losses, no wear)",
                "series": [{"label": "charge price", "values": xs, "context": True}]
                          + [{"label": f"{lab}: {r['battery']} (RTE {r['rte']:.0%}, wear €{r['wear_eur_kwh']:.3f})",
                              "short": lab, "color": color,
                              "values": [round(x / r["rte"] + r["wear_eur_kwh"], 4) for x in xs]}
                             for lab, color, r in picks],
                "formulas": self.breakeven_formulas(picks)}
        out_dir = self.path("results_dir")
        if out_dir and len(df) and not getattr(self, "no_write", False):
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, "breakeven.csv" if cycles is None else "breakeven_recomputed.csv"),
                      index=False)
        self.breakeven_df = df
        if len(df):
            nl = df[df.price_variant == self.ref_variant]
            wear = dict(zip(nl.battery_id, nl.wear_eur_kwh))
            cheap = self.typical_cheap_price()
            md_rows = extras.min_difference_rows(self.batteries, wear, cheap, self.taxes.vat)
            if md_rows:
                sec.tables["minimum_price_difference"] = pd.DataFrame(md_rows)
                sec.md += ("\n#### Minimum price difference: the setting for your battery app"
                           "\n- `at_charge_x` columns: the minimum difference when charging at price x."
                           f"\n- `setting_*`: the value to enter as `min_delta` for a typical cheap hour "
                           f"(€{cheap:.3f}/kWh all-in, average of each day's 4 cheapest hours, cheapest dynamic "
                           "contract, 2027 rules). `setting_all_in` if the app compares prices incl. taxes, "
                           "`setting_spot` if it compares spot (EPEX) prices (= all-in ÷ 1.21).")
        self.emit(sec)

    def breakeven_formulas(self, picks) -> list[dict]:
        """Per chart line: the break-even formula in all-in prices and as a spot-price (EPEX)
        rule with the cheapest dynamic contract's markup and energy tax – ready to use as a
        setting in a battery app or Home Assistant."""
        dyn = self.cheapest_dynamic()
        vat = self.taxes.vat
        eb = self.taxes.energy_tax(2027)
        m = dyn.markup if dyn is not None else 0.0
        cheap = self.typical_cheap_price()
        out = []
        for lab, _, r in picks:
            rte, w = r["rte"], r["wear_eur_kwh"]
            a = 1.0 / rte
            b = (m + eb) * (a - 1.0) + w / (1 + vat)
            spot_cheap = cheap / (1 + vat) - m - eb
            be = cheap * a + w
            out.append({"label": f"{lab}: {r['battery']}", "lines": [
                f"Break-even (all-in €/kWh) = charge price × {a:.4f} + {w:.4f}",
                f"Minimum price difference = charge price × {a - 1:.4f} + {w:.4f}",
                f"Spot rule (EPEX, excl. tax): discharge/sell only if spot ≥ charge spot × {a:.4f} + {b:.4f}",
                f"  (markup €{m:.4f} of {dyn.label if dyn else '–'}, energy tax €{eb:.4f}, VAT {vat:.0%})",
                f"Example: charging at €{cheap:.3f} all-in (spot €{spot_cheap:.3f}) → break-even €{be:.3f} all-in,"
                f" minimum difference €{be - cheap:.3f} all-in = €{spot_cheap * (a - 1) + b:.3f} spot (min_delta)",
                f"Where: RTE {rte:.0%} (1/RTE = {a:.4f}); wear €{w:.4f}/kWh = €{r['purchase_eur']:,.0f} ÷ "
                f"{r['lifetime_kwh']:,.0f} kWh over its life ({r['lifetime_limit']})"]})
        return out

    def breakeven_explainer(self, rows) -> list[str]:
        """Plain-language formula with a worked example (first priced battery)."""
        ex = next((r for r in rows if r["price_variant"] == self.ref_variant and r["purchase_eur"]
                   and not math.isnan(r["wear_eur_kwh"])), None)
        out = ["#### How the break-even price is calculated",
               "`break-even price = charge price / RTE + wear cost` (€ per kWh delivered, all-in).",
               "- **charge price / RTE** turns the price of a kWh *put into* the battery into the price of a kWh "
               "*coming out*: with a round-trip efficiency (RTE) of 85% you lose 15%, so each delivered kWh costs "
               "1/0.85 = 1.18 × the charge price.",
               "- **wear cost** is already per kWh delivered (purchase price ÷ kWh the battery delivers over its "
               "life), so it is added after the division – not `charge price / (RTE + wear cost)`.",
               "- **Minimum price difference worth charging** = break-even − charge price "
               "= charge price × (1/RTE − 1) + wear cost."]
        if ex:
            cp, rte, w = 0.20, ex["rte"], ex["wear_eur_kwh"]
            be = cp / rte + w
            out.append(f"- Example, {ex['battery']} (RTE {rte:.0%}, wear €{w:.3f}/kWh, €{ex['purchase_eur']:,.0f} "
                       f"over {ex['lifetime_kwh']:,.0f} kWh) charging at €{cp:.2f}: {cp:.2f} / {rte:.2f} + {w:.3f} "
                       f"= {cp / rte:.3f} + {w:.3f} = **€{be:.3f}/kWh**. Discharging only pays if that kWh is worth "
                       f"at least this much, a price difference of €{be - cp:.3f}.")
        return out

    def cheapest_dynamic(self):
        dyn = [c for c in self.contracts if c.is_dynamic]
        if not dyn or self.prices is None or not self.head_years:
            return None
        def key(c):
            v = [x[0].total - x[0].bonus_amortised for y, x in self.contract_year_costs(c, "nosal_min50").items()
                 if y in self.head_years]
            return np.mean(v) if v else np.inf
        return min(dyn, key=key)

    def cheapest_fixed(self):
        """The cheapest fixed contract you can switch to from 2027 (a new offer; the current
        fixed contract only when there is no other fixed offer)."""
        e = getattr(self, "expected_by_contract", {}) or {}
        fixed = [c for c in self.contracts if not c.is_dynamic and e.get(c.id, {}).get("nosal_min50") is not None]
        offers = [c for c in fixed if c is not self.current] or fixed
        return min(offers, key=lambda c: e[c.id]["nosal_min50"]) if offers else None

    def cost27(self, c) -> float | None:
        """Expected yearly cost from 2027 (2027–2029 rules, headline years; bonus excluded)."""
        return (getattr(self, "expected_by_contract", {}) or {}).get(c.id, {}).get("nosal_min50") if c else None

    def fixed_vs_dynamic_table(self) -> pd.DataFrame | None:
        e = getattr(self, "expected_by_contract", {}) or {}
        fx, dy = self.cheapest_fixed(), self.cheapest_dynamic()
        rows = []
        for role, c in (("cheapest dynamic", dy), ("cheapest fixed offer", fx), ("current fixed contract", self.current)):
            if c is None or c.id not in e or (role == "current fixed contract" and c is fx):
                continue
            x = e[c.id]
            rows.append({"role": role, "contract": c.label, "cost_2026_rules_eur": x.get("saldering"),
                         "cost_2027_2029_eur": x.get("nosal_min50"), "cost_2030_eur": x.get("nosal_2030"),
                         "switch_bonus_eur_once": c.welcome_bonus})
        if not rows:
            return None
        df = pd.DataFrame(rows)
        ref = df.loc[df.role == "cheapest dynamic", "cost_2027_2029_eur"]
        if len(ref):
            df["extra_vs_dynamic_2027_eur"] = df.cost_2027_2029_eur - ref.iloc[0]
        return df

    def contract_prices(self, c):
        """(contract, average all-in import price weighted by your import, average export value
        weighted by your export) under 2027–2029 rules; dynamic contracts over the last 3 price years."""
        if c is None:
            return None
        key = ("cprices", c.id)
        if key not in self.results:
            us, ss = [], []
            for _, per, _, _ in self.periods(c, "nosal_min50", self.head_years if c.is_dynamic else None):
                u, s = marginal_values(per, c, self.regimes["nosal_min50"], self.taxes, self.net_importer)
                us.append(np.average(u, weights=np.maximum(per.imp, 1e-9)))
                ss.append(np.average(s, weights=np.maximum(per.exp, 1e-9)))
            self.results[key] = (c, float(np.mean(us)), float(np.mean(ss))) if us else None
        return self.results[key]

    def fixed_prices(self):
        return self.contract_prices(self.cheapest_fixed())

    def fixed_vs_dynamic_line(self) -> str:
        fx, dy = self.cheapest_fixed(), self.cheapest_dynamic()
        a, b = self.cost27(fx), self.cost27(dy)
        if a is None or b is None:
            return ""
        return (f"**Fixed vs dynamic from 2027:** cheapest fixed offer {fx.label} €{a:,.0f} a year, cheapest "
                f"dynamic {dy.label} €{b:,.0f} – fixed costs €{a - b:,.0f} a year more (without battery, "
                "one-off bonuses excluded).")

    def typical_cheap_price(self, hours: int = 4) -> float:
        c = self.cheapest_dynamic()
        if c is None:
            return 0.20
        vals = []
        for _, per, idx, _ in self.periods(c, "nosal_min50", self.head_years):
            u, _s = marginal_values(per, c, self.regimes["nosal_min50"], self.taxes, self.net_importer)
            day = pd.Series(u, index=idx).groupby(idx.date)
            vals.append(day.apply(lambda x: np.sort(x.values)[:hours * 4].mean()).mean())
        return float(np.mean(vals)) if vals else 0.20

    # ------------------------------------------------------------------ step 1b
    def step_power(self):
        sec = Section("power", "2. Power profile")
        raw = None
        src = self.inp.p1 or self.path("p1_file")
        cols = self.cfg["p1"].get("phase_max_columns") or []
        if cols and isinstance(src, str) and os.path.exists(src):
            try:
                raw = pd.read_csv(src, usecols=lambda c: c in cols)
            except Exception:
                raw = None
        stats, cover, phases, imp_s, exp_s = extras.power_profile(self.profile, self.real_mask, raw, cols)
        sec.tables["import_export_power"] = stats.round(2)
        sec.tables["battery_power_vs_energy"] = cover.round(3)
        if phases is not None:
            sec.tables["peak_per_phase"] = phases.round(2)
        c8 = cover.set_index("battery_power_kw")
        md = ["A battery can only store surplus and cover load up to its own power.",
              f"- An 800 W socket battery can store {c8.loc[0.8, 'share_of_surplus_it_can_store']:.0%} of the surplus and "
              f"cover {c8.loc[0.8, 'share_of_import_it_can_cover']:.0%} of the import; 2.4 kW: "
              f"{c8.loc[2.4, 'share_of_surplus_it_can_store']:.0%} / {c8.loc[2.4, 'share_of_import_it_can_cover']:.0%}.",
              "#### Notes",
              "- Power from the 15-minute meter data (real months only).",
              "- 15-minute averages hide short peaks (kettle, induction hob). `peak_per_phase` uses the meter's own "
              "per-phase maximum (W) per interval, when the P1 file has those columns.",
              f"- Connection {self.conn.label}: {self.conn.phase_kw:.2f} kW per phase; batteries are capped at "
              f"{self.conn.battery_phase_cap_w / 1000:.2f} kW per phase."]
        sec.md = "\n".join(md)
        f = self.fig(charts.power_duration, imp_s, exp_s, [0.8, 2.4])
        if f:
            sec.figures["power_duration"] = f
        self.emit(sec)

    # ------------------------------------------------------------------ step 6
    def step_sanity(self):
        sec = Section("sanity", "9. Battery over the year – check per size")
        focus = getattr(self, "focus", None) or []
        bats = [(f["cls"], f["color"], f["b"]) for f in focus]
        if not bats and self.batteries:
            b = min(self.batteries, key=lambda x: (len(x.missing_fields), x.usable_kwh))
            bats = [(self.class_of(b)[0] or "battery", self.CLASS_COLORS[0], b)]
        if not bats:
            sec.md = "No battery in scope."
            self.emit(sec)
            return
        net = self.profile["imp"].values - self.profile["exp"].values
        idx = self.profile.index
        rows, soc_series, chosen_run = [], [], None
        for cls, color, b in bats:
            cap = self.conn.battery_cap_w(b.phases)
            r = simulate(net, SELF, b, cap, keep_trace=True)
            sb = (b.standby_w or 0) / 4000.0
            balance = np.abs((r.imp - r.exp) - (net + sb + r.batt_ac)).max()
            daily = extras.daily_battery(idx, net, r.batt_ac, r.soc, b.usable_kwh, sb)
            rows.append({"size_class": cls, "battery": b.name, "usable_kwh": b.usable_kwh,
                         "efc_per_year": r.efc, "charged_kwh": r.charged_ac, "delivered_kwh": r.discharged_ac,
                         "import_before_kwh": self.profile["imp"].sum(), "import_after_kwh": r.imp.sum(),
                         "export_before_kwh": self.profile["exp"].sum(), "export_after_kwh": r.exp.sum(),
                         "days_full": int(daily.full.sum()),
                         "days_below_half": int((daily.max_soc_kwh < 0.5 * b.usable_kwh).sum()),
                         "end_capacity_kwh": r.end_capacity, "balance_error_kwh": balance})
            soc_series.append(extras.series(f"{cls}: {b.name} – state of charge (kWh)", r.soc, color))
            if chosen_run is None or (getattr(self, "chosen", None) is not None and b.id == self.chosen.id):
                chosen_run = (cls, b, r)
        df = pd.DataFrame(rows)
        sec.tables["self_consumption_check"] = df.round(2)
        worst = df.balance_error_kwh.max()
        md = ["Each size class's best battery on your profile year with plain self-consumption: does the "
              "simulation behave?",
              "`import − export = house net load + standby + battery flow` must hold every 15 minutes "
              f"(largest error: {worst:.1e} kWh).",
              "`full cycles = energy delivered ÷ usable capacity`"]
        md += [f"- **{r.size_class}:** full on {r.days_full} days, below half full on {r.days_below_half} days, "
               f"{r.efc_per_year:.0f} cycles, import {r.import_before_kwh:,.0f} → {r.import_after_kwh:,.0f} kWh"
               for r in df.itertuples()]
        sec.md = "\n".join(md)
        sec.interactive["soc_year_15min"] = {
            "title": "State of charge per size class – whole year, every 15 minutes",
            "y": "kWh", "height": 320, "x": extras.ts(idx), "series": soc_series}
        cls, b, r = chosen_run
        sec.interactive["power_year_15min"] = {
            "title": f"{cls}: {b.name} – house and battery power, every 15 minutes",
            "y": "kW", "height": 300, "x": extras.ts(idx), "group": "Power flows of the recommended battery",
            "series": [extras.series("House net power without battery (kW, + import / − export)", net * 4, "#2a78d6",
                                     width=0.8),
                       extras.series("Battery power (kW, + charging / − discharging)", r.batt_ac * 4, "#eb6834",
                                     width=0.8)]}
        self.sanity = {"battery": b.id, "efc": r.efc, "balance_error": worst}
        self.emit(sec)

    # ------------------------------------------------------------------ step 7
    def strategies_for(self, c: Contract) -> list[str]:
        enabled = self.cfg.get("strategies", {}).get("enabled", [])
        out = [s for s in enabled if s in ("self_consumption", "timed")]
        if c.is_dynamic:
            presets = self.cfg.get("strategies", {}).get("hbc_presets", {}) or {}
            out += [s for s in enabled
                    if s in ("dynamic", "dynamic_sell", "perfect_foresight") or s in presets
                    or (s == "forecast" and lp_available())]
            # Curtailment only matters where export can be worth less than nothing (dynamic prices).
            if "curtail" in enabled:
                bases = self.cfg.get("strategies", {}).get("curtail", {}).get("bases", ["self_consumption", "forecast"])
                out += [f"{s}_curtail" for s in bases if s in out]
        return out

    @staticmethod
    def base_strategy(strategy: str) -> str:
        return strategy[:-len("_curtail")] if strategy.endswith("_curtail") else strategy

    def sim_strategy(self, strategy, b, cap, per: Period, idx, c, scn, wear, windows=None, scale=1.0,
                     rte=None, standby=None, keep_trace=False):
        net = per.imp - per.exp
        bb = b
        if rte is not None or standby is not None:
            from dataclasses import replace
            bb = replace(b, rte=rte if rte is not None else b.rte,
                         standby_w=standby if standby is not None else b.standby_w)
        if strategy.endswith("_curtail"):
            r = self.sim_strategy(self.base_strategy(strategy), b, cap, per, idx, c, scn, wear, windows, scale,
                                  rte, standby, keep_trace)
            u, s = marginal_values(per, c, self.regimes[scn], self.taxes, self.net_importer)
            return curtail(r, s, float(self.cfg.get("strategies", {}).get("curtail", {}).get("below_eur_kwh", 0.0)))
        if strategy == "self_consumption":
            return simulate(net, SELF, bb, cap, keep_trace=keep_trace)
        u, s = marginal_values(per, c, self.regimes[scn], self.taxes, self.net_importer)
        if strategy == "timed":
            modes = plan_timed(idx, windows or [])
            return simulate(net, modes, bb, cap, keep_trace=keep_trace)
        if strategy in ("dynamic", "dynamic_sell"):
            modes = plan_dynamic(idx, u, s, bb, cap, wear, scale, strategy == "dynamic_sell",
                                 self.cfg.get("strategies", {}).get("dynamic", {}), export=per.exp,
                                 solar_forecast=self.pv_for(idx), imports=per.imp)
            return simulate(net, modes, bb, cap, keep_trace=keep_trace)
        presets = self.cfg.get("strategies", {}).get("hbc_presets", {}) or {}
        if strategy in presets:
            return simulate(net, plan_hbc(idx, u, presets[strategy]), bb, cap, keep_trace=keep_trace)
        if strategy == "forecast":
            modes = plan_forecast(idx, net, u, s, bb, cap, wear, self.cfg.get("strategies", {}).get("forecast", {}))
            return simulate(net, modes, bb, cap, keep_trace=keep_trace)
        if strategy == "perfect_foresight":
            lv = self.cfg.get("strategies", {}).get("perfect_foresight", {}).get("soc_levels", 21)
            return perfect_foresight(net, u, s, bb, cap, 0.0, lv)
        raise ValueError(strategy)

    def pv_for(self, idx):
        """Solar forecast for the Charge goal: only with PV data (section 8)."""
        if self.pv is None:
            return None
        if len(idx) == len(self.profile.index) and idx[0] == self.profile.index[0]:
            # PV production minus a share for own use is unknown; use PV × export share.
            share = self.profile["exp"].sum() / max(self.pv.sum(), 1e-9)
            return self.pv.fillna(0).values * share
        return None

    def saving(self, res, per: Period, c, scn) -> float:
        base = self.base_cost(per, c, scn)
        withb = compute_cost(per.with_flows(res.imp, res.exp), c, self.regimes[scn], self.taxes,
                             self.conn.label, include_fixed=False).total
        return base - withb

    def base_cost(self, per, c, scn):
        # Periods are cached in self.results, so their ids are stable.
        key = ("base", id(per), c.id, scn)
        if key not in self.results:
            self.results[key] = compute_cost(per, c, self.regimes[scn], self.taxes, self.conn.label,
                                             include_fixed=False).total
        return self.results[key]

    def search_timed(self, b, cap, c, wear):
        tcfg = self.cfg.get("strategies", {}).get("timed", {})
        wh = int(tcfg.get("window_hours", 3))
        scn = "nosal_min50"
        pers = self.periods(c, scn, self.head_years[-1:] if c.is_dynamic else None)
        if not pers:
            return []
        _, per, idx, _ = pers[-1]
        cands = [[]]
        for cs in tcfg.get("charge_starts", [1, 3, 12, 13]):
            for ds in tcfg.get("discharge_starts", [7, 17, 18, 19]):
                ce, de = (cs + wh) % 24, (ds + wh) % 24
                if cs < ds < cs + wh or ds < cs < ds + wh:
                    continue
                cands.append([(cs, ce, GRID_CHARGE), (ds, de, ZERO_IMPORT)])
        best, best_s = [], -np.inf
        for w in cands:
            r = self.sim_strategy("timed", b, cap, per, idx, c, scn, wear, windows=w)
            sv = self.saving(r, per, c, scn)
            if sv > best_s + 1e-6:
                best, best_s = w, sv
        return best

    def step_payback(self):
        sec = Section("payback", "7. Payback per battery size and strategy")
        if not self.batteries:
            sec.md = "No battery in scope."
            self.emit(sec)
            return
        contracts = [c for c in self.contracts if not c.is_dynamic or (self.prices is not None and self.full_years)]
        # Dynamic contracts differ only in markup and fixed costs, so battery runs use the
        # cheapest few (2027-2029 rules, last 3 years, without switch bonus).
        top_n = int(self.cfg.get("analysis", {}).get("payback_top_dynamic_contracts", 3))
        dyn = [c for c in contracts if c.is_dynamic]
        if len(dyn) > top_n:
            def key(c):
                costs = self.contract_year_costs(c, "nosal_min50")
                v = [x[0].total - x[0].bonus_amortised for y, x in costs.items() if y in self.head_years]
                return np.mean(v) if v else np.inf
            keep = {c.id for c in sorted(dyn, key=key)[:top_n]}
            contracts = [c for c in contracts if not c.is_dynamic or c.id in keep]
            self.log(f"  battery runs on the {top_n} cheapest dynamic contracts: "
                     + ", ".join(c.label for c in contracts if c.is_dynamic))
        if not contracts:
            sec.md = "No contract to simulate against."
            self.emit(sec)
            return
        d = self.cfg.get("battery_defaults", {})
        records = []          # one per battery/contract/strategy/scenario/period
        meta = {}
        for b in self.batteries:
            cap = self.conn.battery_cap_w(b.phases)
            wear, wear_src = self.wear_for(b)
            for c in contracts:
                windows = self.search_timed(b, cap, c, wear) if "timed" in self.strategies_for(c) else []
                meta[(b.id, c.id)] = {"windows": windows, "wear": wear, "wear_src": wear_src}
                for strat in self.strategies_for(c):
                    if strat == "timed" and not windows:
                        continue      # no window beats self-consumption: identical results
                    for scn in self.scenarios:
                        lp = self.base_strategy(strat) in ("perfect_foresight", "forecast")   # LP: headline years
                        years = self.head_years if lp else None
                        if lp and self.opts.quick:
                            years = self.head_years[-1:]
                        for label, per, idx, _ in self.periods(c, scn, years):
                            r = self.sim_strategy(strat, b, cap, per, idx, c, scn, wear, windows=windows)
                            sv = self.saving(r, per, c, scn)
                            days = per.days
                            records.append({"battery_id": b.id, "contract_id": c.id, "strategy": strat,
                                            "scenario": scn, "price_year": label,
                                            "saving": sv * 365.0 / days if label in self.partial_years else sv,
                                            "efc": r.efc * 365.0 / days, "delivered_kwh": r.discharged_ac * 365.0 / days,
                                            "partial": label in self.partial_years})
                self.log(f"  simulated {b.name} × {c.label}")
        rec = pd.DataFrame(records)
        self.records = rec
        self._meta = meta
        summary = self.payback_table(rec, meta, contracts)
        self.payback_summary = summary
        self.focus = self.pick_focus(summary)
        self.chosen = self.pick_battery(summary)
        adv = self.advice_contract()
        md = ["`saving = yearly cost without battery − yearly cost with battery` (same contract, rules and price year)",
              "`payback = years until the savings add up to the price` (2026 rules until the end of 2026, then "
              "2027–2029 rules, then 2030+ rules; capacity fades with use)",
              f"Comparison basis: the best battery per size class ({', '.join(f['cls'] for f in self.focus) or '–'}) "
              f"on {adv.label if adv else 'the best contract'}."]
        if self.focus:
            sec.summary = "Best per size class – " + "; ".join(
                f"{f['cls']}: {self.short_name(f['b'])}, {f['row']['payback_years']:.1f} yr, "
                f"€{f['row']['_saving27']:,.0f}/yr" for f in self.focus) + "."
            md += [f"- **{f['cls']}:** {f['b'].name} – `{f['strategy']}`, payback {f['row']['payback_years']:.1f} "
                   f"years, €{f['row']['_saving27']:,.0f} a year from 2027" for f in self.focus]
        md += ["#### Strategies explained",
               "- `self_consumption`: store solar surplus, use it when the house imports. Works in every battery app.",
               "- `timed`: fixed daily charge/discharge windows, searched on the last price year.",
               "- `dynamic` / `dynamic_sell`: per day, pair the cheapest and the most expensive hours when the "
               "difference beats the break-even (section 6), keep stored energy for the priciest load; `_sell` may "
               "also sell. Needs custom control.",
               "- `forecast`: every day at 13:00, optimise over the prices known until tomorrow night and a "
               "usage/solar forecast from the last 3 days (like EMHASS in Home Assistant). Needs custom control. "
               "Last 3 price years only.",
               "- `hbc_default` / `hbc_pv_first`: Home Battery Control's Dynamic strategy as-is.",
               "- `…_curtail` (`self_consumption_curtail`, `forecast_curtail`): the same strategy, but once the "
               "battery is full (or at its power limit) the solar inverter scales back to zero export whenever "
               "exporting has a negative value. `curtailed export = remaining export in quarter-hours where the "
               "export price < 0`. Needs an inverter Home Assistant can control. Curtailing would also save money "
               "without a battery, so part of this gain is not the battery's.",
               "- `perfect_foresight`: knows all prices and usage in advance – the upper limit, not achievable. "
               "Last 3 price years only.",
               "#### Ranges and assumptions",
               "- Dynamic contracts: the payback min–max is the spread over the price years; partial years are "
               "left out.",
               f"- Purchase date {self.cfg['analysis']['purchase_date']}; net present value at "
               f"{float(self.cfg['analysis'].get('discount_rate', 0.03)):.0%} a year."]
        if self.prices is None:
            md.append("**No price history loaded:** only fixed-contract combinations were simulated.")
        if not lp_available():
            md.append("**Optimiser not available in this browser** (scipy could not load its maths library): the "
                      "`forecast` strategies and the gap analysis are skipped, and `perfect_foresight` uses a slower "
                      "approximation. Try another browser (Chrome, Edge or Firefox on a computer) or the command line "
                      "for the full results.")
        estimated = [b.name for b in self.batteries if b.estimated]
        if estimated:
            md.append(f"- Estimated specs (defaults used): {', '.join(estimated)}.")
        windows = [f"{self.bname(bid)} × {self.cname(cid)}: " + (", ".join(
            f"{'charge' if m == GRID_CHARGE else 'discharge'} {a:02d}–{z:02d}" for a, z, m in v['windows'])
            or "none beats self-consumption") for (bid, cid), v in meta.items() if v["windows"] is not None]
        if windows:
            md += ["#### Timed windows found (2027 rules)"] + [f"- {w}" for w in windows]
        sec.tables["savings_by_combination"] = summary["savings"].round(2)
        if len(summary["ranked"]):
            sec.tables["payback_ranked"] = self.payback_compact(summary["ranked"]).round(2)
            sec.tables["payback_all_variants"] = summary["ranked"].round(2)
            sec.csv_only.add("payback_all_variants")
            fvd = self.fixed_vs_dynamic_batteries(summary["ranked"])
            if fvd is not None:
                sec.tables = {"fixed_vs_dynamic_with_battery": fvd.round(1), **sec.tables}
                md.insert(3, self.fixed_vs_dynamic_battery_text(fvd))
            for name, spec in (("payback_vs_size", self.payback_scatter_spec(summary["ranked"])),
                               ("cumulative_return", self.cumulative_spec(summary["ranked"]))):
                if spec:
                    sec.interactive[name] = spec
        else:
            md.append("**No payback ranking:** no battery has a purchase price yet. Savings per year are shown above.")
        sv = self.scale_variants(meta, contracts)
        if sv is not None:
            sec.tables["dynamic_breakeven_scale"] = sv.round(2)
        sens = self.sensitivity(summary, meta, contracts)
        if sens is not None:
            sec.tables["sensitivity"] = sens.round(2)
        ys = self.savings_year_spec(rec)
        if ys:
            sec.interactive["savings_per_price_year"] = ys
        bft = self.blackfriday_table(summary)
        if bft is not None:
            sec.tables["black_friday_quick_decision"] = bft
            bfc = self.cfg.get("blackfriday", {}) or {}
            w = bfc.get("window", ["11-20", "12-01"])
            found = bft["nl_deal_eur"].notna().any() or bft["de_deal_eur"].notna().any()
            est = bool(bfc.get("estimates", True))
            md += [f"#### Black Friday quick decision (window {w[0]} – {w[1]})",
                   "- Per battery: best contract and strategy, today's price, the real deal the scraper found "
                   "(NL incl. VAT, DE 0% VAT from German manufacturer shops) and its discount against the last "
                   "normal price."
                   + (f" Estimate: −{float(bfc.get('discount_nl', 0.15)):.0%} NL / "
                      f"−{float(bfc.get('discount_de', 0.15)):.0%} DE." if est else " Estimates are switched off."),
                   "- " + ("Real deals recorded." if found else "No real deals recorded yet: the scraper checks every "
                           "4 hours during the window.")]
        ef = self.earnings_focus(meta)
        if ef is not None:
            avg, specs, per_year, ey, ec = ef
            sec.tables["earnings_average_full_year"] = avg.round(0)
            sec.tables["earnings_per_ownership_year"] = per_year.round(0)
            sec.csv_only.add("earnings_per_ownership_year")
            for k, v in specs.items():
                v["group"] = "Earnings per strategy – average full year, per size class"
                sec.interactive[k] = v
            md += ["#### Earnings per strategy",
                   "`net saving = avoided import + sold to grid − feed-in given up − grid charging − standby/other`",
                   f"- Average over the full years of ownership ({ey} prices on {ec.label}); the partial purchase "
                   "year (2026) and the partial last year are left out.",
                   "- `timed` shows large bars because it cycles from the grid every day: big avoided import, "
                   "big grid-charging cost, the net is what counts."]
        gap = self.strategy_gap(meta)
        if gap is not None and len(gap):
            sec.tables["gap_to_perfect_foresight"] = gap.round(0)
            md.append(self.gap_text(gap))
        sec.md = "\n".join(md)
        self.emit(sec)
        # Recompute Objective 1 with simulated cycles (best strategy per battery).
        cyc = {}
        if len(rec):
            best = rec[rec["scenario"] == "nosal_min50"].groupby(["battery_id", "contract_id", "strategy"])["saving"].mean()
            for bid in rec["battery_id"].unique():
                sub = best.loc[bid]
                c_id, strat = sub.idxmax()
                cyc[bid] = rec[(rec.battery_id == bid) & (rec.contract_id == c_id) & (rec.strategy == strat)
                               & (rec.scenario == "nosal_min50")]["efc"].mean()
        if cyc:
            self.step_breakeven(cycles=cyc)

    def blackfriday_table(self, summary):
        rk = summary.get("ranked") if summary else None
        if rk is None or rk.empty:
            return None
        h = rk[(rk.analysis == "headline") & (rk.strategy != "perfect_foresight")]   # only achievable strategies
        now = h[h.price_variant == self.ref_variant]
        if now.empty:
            return None
        best = now.sort_values("payback_years").drop_duplicates("_bid")
        rows = []
        for r in best.to_dict("records"):
            b = next(x for x in self.batteries if x.id == r["_bid"])
            same = h[(h._bid == r["_bid"]) & (h._cid == r["_cid"]) & (h.strategy == r["strategy"])]
            pb = dict(zip(same.price_variant, same.payback_years))
            pr = dict(zip(same.price_variant, same.price_eur))
            disc = lambda deal, ref: (1 - deal / ref) if (deal and ref) else None
            row = {"size_class": self.class_of(b)[0] or "other",
                   "battery": r["battery"], "usable_kwh": r["usable_kwh"], "contract": r["contract"],
                   "strategy": r["strategy"],
                   "nl_now_eur": pr.get("NL current"), "payback_now": pb.get(self.ref_variant),
                   "nl_deal_eur": b.price_bf_nl, "nl_deal_discount": disc(b.price_bf_nl, b.price_nl_ref),
                   "payback_nl_deal": pb.get("NL Black Friday"),
                   "nl_est_eur": pr.get("NL Black Friday (est.)"), "payback_nl_est": pb.get("NL Black Friday (est.)"),
                   "de_now_eur": b.price_de, "de_deal_eur": b.price_bf_de,
                   "de_deal_discount": disc(b.price_bf_de, b.price_de_ref),
                   "payback_de_deal": pb.get("DE Black Friday"), "payback_de_est": pb.get("DE Black Friday (est.)"),
                   "deal_found": "; ".join(x for x in (b.bf_nl_info, b.bf_de_info) if x)}
            cands = [v for v in (row["payback_nl_deal"], row["payback_de_deal"]) if v is not None]
            row["best_payback"] = min(cands) if cands else min(
                v for v in (row["payback_nl_est"], row["payback_now"]) if v is not None)
            rows.append(row)
        df = pd.DataFrame(rows)
        order = {lab: i for i, (lab, *_rest) in enumerate(self.size_classes())}
        df = (df.assign(_o=df.size_class.map(order).fillna(99)).sort_values(["_o", "best_payback"])
              .drop(columns="_o").reset_index(drop=True))
        if not (self.cfg.get("blackfriday", {}) or {}).get("estimates", True):
            df = df.drop(columns=["nl_est_eur", "payback_nl_est", "payback_de_est"])
        for c in ("nl_deal_discount", "de_deal_discount"):
            df[c] = df[c].map(lambda v: f"{v:.0%}" if isinstance(v, float) and not np.isnan(v) else "")
        return df.round(2)

    # ------------------------------------------------------------------ 11.4 / 11.6 helpers
    def pick_battery(self, summary) -> Battery | None:
        want = self.opts.chosen_battery or self.cfg.get("chosen_battery")
        if want:
            for b in self.batteries:
                if b.id == want:
                    return b
            self.log(f"  chosen battery {want!r} not found; using the fastest payback")
        rk = summary.get("ranked") if summary else None
        if rk is not None and len(rk):
            r = rk[(rk.analysis == "headline") & (rk.price_variant == self.ref_variant)
                   & (rk.strategy != "perfect_foresight")]
            adv = self.advice_contract()
            if adv is not None and (r._cid == adv.id).any():
                r = r[r._cid == adv.id]
            if len(r):
                bid = r.sort_values("payback_years").iloc[0]["_bid"]
                return next(b for b in self.batteries if b.id == bid)
        return self.batteries[0] if self.batteries else None

    def life_years(self, b: Battery, efc: float) -> float:
        lim = [b.warranty_years or 10.0]
        if efc and efc > 0 and b.cycle_life:
            lim.append(b.cycle_life / efc)
        return min(min(lim), float(self.cfg["analysis"].get("horizon_years", 20)))

    # ------------------------------------------------------------------ size classes
    CLASS_COLORS = ["#2a78d6", "#4a3aa7", "#eb6834", "#1baf7a"]   # validated all-pairs (scatter) as a set
    CLASS_COLOR_BY_KWH = {5.0: "#2a78d6", 7.5: "#4a3aa7", 10.0: "#eb6834", 15.0: "#1baf7a"}
    EARN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    EARN_KEYS = [("avoided_import", "Avoided import"), ("sold_to_grid", "Sold to grid"),
                 ("solar_feed_in_given_up", "Feed-in given up (stored solar)"),
                 ("grid_charging", "Grid charging"), ("standby_and_other", "Standby / other"),
                 ("avoided_negative_export", "Curtailed export (negative prices avoided)")]

    def size_classes(self):
        """[(label, lower kWh, upper kWh, colour)]: each class runs to the midpoint between
        its neighbours' centres; the outer edges are centre ± size_class_halfwidth_kwh.
        Colours follow the class (5 blue, 7.5 violet, 10 orange, 15 aqua)."""
        a = self.cfg.get("analysis", {})
        half = float(a.get("size_class_halfwidth_kwh", 2.5))
        cs = sorted(float(c) for c in a.get("size_classes_kwh", [5, 7.5, 10, 15]))
        out = []
        for i, c in enumerate(cs):
            lo = (cs[i - 1] + c) / 2 if i else c - half
            hi = (c + cs[i + 1]) / 2 if i + 1 < len(cs) else c + half
            color = self.CLASS_COLOR_BY_KWH.get(c, self.CLASS_COLORS[i % len(self.CLASS_COLORS)])
            out.append((f"{c:g} kWh", lo, hi, color))
        return out

    def size_of(self, b: Battery) -> float:
        return b.nominal_kwh or b.usable_kwh

    def class_of(self, b: Battery):
        for label, lo, hi, color in self.size_classes():
            if lo <= self.size_of(b) < hi:
                return label, color
        return None, None

    def battery(self, bid):
        return next(x for x in self.batteries if x.id == bid)

    def achievable_ranked(self, rk: pd.DataFrame) -> pd.DataFrame:
        """Fastest-payback row per battery: headline years, today's reference price (NL, or the
        cheaper of NL/DE with German prices on), achievable
        strategies, on the advice contract."""
        h = rk[(rk.analysis == "headline") & (rk.price_variant == self.ref_variant)
               & (rk.strategy != "perfect_foresight")]
        adv = self.advice_contract()
        if adv is not None and (h._cid == adv.id).any():
            h = h[h._cid == adv.id]
        return h.sort_values("payback_years").drop_duplicates("_bid")

    def pick_focus(self, summary) -> list[dict]:
        """The best-value battery of each size class (5 / 7.5 / 10 / 15 kWh): these
        are followed through every later section. A fixed chosen battery replaces
        the winner of its own class."""
        rk = summary.get("ranked") if summary else None
        if rk is None or rk.empty:
            return []
        best = self.achievable_ranked(rk)
        want = self.opts.chosen_battery or self.cfg.get("chosen_battery")
        out = []
        for label, lo, hi, color in self.size_classes():
            sel = best[best._bid.map(lambda i: lo <= self.size_of(self.battery(i)) < hi)]
            if sel.empty:
                continue
            r = sel[sel._bid == want].iloc[0] if want and (sel._bid == want).any() else sel.iloc[0]
            out.append({"cls": label, "color": color, "b": self.battery(r["_bid"]),
                        "c": next(x for x in self.contracts if x.id == r["_cid"]), "strategy": r["strategy"],
                        "row": r})
        return out

    def price_basis(self) -> str:
        if self.use_de:
            t = float(self.cfg.get("battery_defaults", {}).get("de_travel_cost_eur", 0.0))
            return f"today's cheaper price of NL (incl. VAT) and DE (0% VAT + €{t:,.0f} travel)"
        return "today's NL price (incl. VAT)"

    def short_name(self, b: Battery) -> str:
        return re.sub(r"\s*\(.*\)$", "", b.name)

    # ------------------------------------------------------------------ payback charts
    def payback_compact(self, rk: pd.DataFrame) -> pd.DataFrame:
        """One row per battery × contract (best achievable strategy, reference price, last 3
        years), with the yearly total cost incl. the battery – so fixed and dynamic contracts
        can be compared on what you actually pay, not only on payback time."""
        h = rk[(rk.analysis == "headline") & (rk.price_variant == self.ref_variant) & (rk.strategy != "perfect_foresight")]
        best = h.sort_values("payback_years").drop_duplicates(["_bid", "_cid"])
        rows = []
        for r in best.to_dict("records"):
            c = next(x for x in self.contracts if x.id == r["_cid"])
            base = self.cost27(c)
            rows.append({"size_class": self.class_of(self.battery(r["_bid"]))[0] or "other", "battery": r["battery"],
                         "contract": r["contract"], "contract_type": c.type, "strategy": r["strategy"],
                         "price_eur": r["price_eur"], "bought_in": r.get("bought_in", "NL"),
                         "saving_eur_year_2027": r["_saving27"], "payback_years": r["payback_years"],
                         "npv_eur": r["npv_eur"],
                         "yearly_cost_with_battery_2027": (base - r["_saving27"]) if base is not None else np.nan,
                         "_bid": r["_bid"], "_cid": r["_cid"]})
        df = pd.DataFrame(rows)
        return df.sort_values(["yearly_cost_with_battery_2027", "payback_years"]).reset_index(drop=True)

    def fixed_vs_dynamic_batteries(self, rk) -> pd.DataFrame | None:
        """Per size class: the class winner on the cheapest fixed offer vs the cheapest dynamic contract."""
        fx, dy = self.cheapest_fixed(), self.cheapest_dynamic()
        if fx is None or dy is None or not getattr(self, "focus", None):
            return None
        comp = self.payback_compact(rk)
        rows = []
        for f in self.focus:
            row = {"size_class": f["cls"], "battery": f["b"].name}
            for tag, c in (("fixed", fx), ("dynamic", dy)):
                r = comp[(comp._bid == f["b"].id) & (comp._cid == c.id)]
                if r.empty:
                    continue
                r = r.iloc[0]
                row.update({f"{tag}_strategy": r.strategy, f"{tag}_saving_eur": r.saving_eur_year_2027,
                            f"{tag}_payback_years": r.payback_years,
                            f"{tag}_cost_with_battery_eur": r.yearly_cost_with_battery_2027})
            if "fixed_cost_with_battery_eur" in row and "dynamic_cost_with_battery_eur" in row:
                row["fixed_costs_more_eur"] = row["fixed_cost_with_battery_eur"] - row["dynamic_cost_with_battery_eur"]
            rows.append(row)
        return pd.DataFrame(rows) if rows else None

    def fixed_vs_dynamic_battery_text(self, df: pd.DataFrame) -> str:
        fx, dy = self.cheapest_fixed(), self.cheapest_dynamic()
        lines = [f"**Fixed vs dynamic with a battery** ({fx.label} vs {dy.label}, from 2027, per year):"]
        for r in df.itertuples():
            if not hasattr(r, "fixed_cost_with_battery_eur") or pd.isna(getattr(r, "fixed_cost_with_battery_eur", np.nan)):
                continue
            lines.append(f"- **{r.size_class}:** fixed €{r.fixed_cost_with_battery_eur:,.0f} (battery saves "
                         f"€{r.fixed_saving_eur:,.0f}, payback {r.fixed_payback_years:.1f} yr) vs dynamic "
                         f"€{r.dynamic_cost_with_battery_eur:,.0f} (saves €{r.dynamic_saving_eur:,.0f}, payback "
                         f"{r.dynamic_payback_years:.1f} yr)")
        lines.append("- A battery pays back faster on a fixed contract (exported solar is worth little there, so "
                     "every stored kWh saves the full import price), but the fixed contract itself stays more "
                     "expensive: compare the yearly cost, not only the payback.")
        return "\n".join(lines)

    def payback_scatter_spec(self, rk):
        best = self.achievable_ranked(rk)
        best = best[np.isfinite(best.payback_years.astype(float))]
        if best.empty:
            return None
        focus = {f["b"].id: f for f in getattr(self, "focus", [])}
        pts, outside = [], False
        for r in best.to_dict("records"):
            b = self.battery(r["_bid"])
            cls, color = self.class_of(b)
            outside |= cls is None
            col = color or "#a3a29d"
            pts.append({"x": round(b.usable_kwh, 2), "y": round(float(r["payback_years"]), 2), "label": b.name,
                        "color": col, "strong": b.id in focus,
                        "short": f"{cls}: {self.short_name(b)}" if b.id in focus else "",
                        "rows": [[col, f"{r['payback_years']:.1f} years", "payback"],
                                 [col, f"€{r['price_eur']:,.0f}", f"price ({r.get('bought_in') or 'NL'})"],
                                 [col, f"€{r['_saving27']:,.0f} a year", f"saving from 2027 ({r['strategy']})"],
                                 [col, f"€{r['npv_eur']:,.0f}", "net present value"]]})
        legend = [{"label": lab, "color": col} for lab, _, _, col in self.size_classes()]
        if outside:
            legend.append({"label": "outside the size classes", "color": "#a3a29d"})
        c = self.advice_contract()
        return {"kind": "scatter", "title": "Payback time per battery",
                "subtitle": f"One dot per battery: its best strategy on {c.label if c else 'the best contract'}, at "
                            f"{self.price_basis()}. Lower is better; the best of each size class is labelled.",
                "xLabel": "usable capacity (kWh)", "yLabel": "payback (years)", "yUnit": "yr",
                "points": pts, "legend": legend, "height": 340}

    def cumulative_spec(self, rk):
        best = self.achievable_ranked(rk)
        if best.empty:
            return None
        start = pd.Timestamp(self.cfg["analysis"]["purchase_date"])
        s0 = start.year + (start.dayofyear - 1) / 365.0
        grid = sorted({round(s0 + t, 3) for p in best["_path"] for t, _ in p})
        focus = {f["b"].id: f for f in getattr(self, "focus", [])}
        series, points = [], []
        for r in best.to_dict("records"):
            path = {round(s0 + t, 3): v for t, v in r["_path"]}
            vals = [round(path[x], 1) if x in path else None for x in grid]
            f = focus.get(r["_bid"])
            if f:
                series.append({"label": f"{f['cls']}: {f['b'].name} ({r['strategy']})", "short": f["cls"],
                               "color": f["color"], "values": vals})
                if np.isfinite(r["payback_years"]):
                    points.append({"x": round(s0 + r["payback_years"], 3), "y": 0, "color": f["color"],
                                   "label": f"{r['payback_years']:.1f} yr"})
            else:
                series.append({"label": self.bname(r["_bid"]), "values": vals, "context": True})
        series.sort(key=lambda s: 0 if s.get("context") else 1)
        names = [f"purchase {start.date()}"] + [f"1 Jan {round(x)}" if abs(x - round(x)) < 1e-6
                                                 else f"end of life ({x:.1f})" for x in grid[1:]]
        c = self.advice_contract()
        return {"kind": "lines", "title": "Cumulative return: when the savings have paid for the battery",
                "subtitle": f"Starts at minus {self.price_basis()} on {start.date()}, then adds each year's saving "
                            f"(best strategy on "
                            f"{c.label if c else 'the best contract'}, capacity fading). The line crosses zero in the "
                            "payback year and ends at the battery's end of life. Grey: the other batteries.",
                "x": grid, "xNames": names, "xTicks": list(range(math.ceil(s0), math.floor(grid[-1]) + 1)),
                "xFormat": "year", "unit": "€", "yLabel": "€ cumulative", "zero": True, "markers": False,
                "endLabels": True, "series": series, "points": points, "contextLabel": "other batteries",
                "height": 380}

    def savings_year_spec(self, rec):
        if rec.empty or self.prices is None or not getattr(self, "focus", None):
            return None
        yrs = rec[rec.price_year.apply(lambda x: isinstance(x, (int, np.integer))) & (rec.scenario == "nosal_min50")
                  & ~rec.partial]
        if yrs.empty:
            return None
        x = sorted(int(v) for v in yrs.price_year.unique())
        series, swapped = [], False
        for f in self.focus:
            g = yrs[(yrs.battery_id == f["b"].id) & (yrs.contract_id == f["c"].id)]
            strat = f["strategy"]
            if g[g.strategy == strat].price_year.nunique() < len(x):     # LP strategies: headline years only
                full = g.groupby("strategy").filter(lambda d: d.price_year.nunique() >= len(x))
                if len(full):
                    strat = full.groupby("strategy").saving.mean().idxmax()
                    swapped = True
            s = g[g.strategy == strat].set_index("price_year").saving
            series.append({"label": f"{f['cls']}: {f['b'].name} ({strat})", "short": f["cls"], "color": f["color"],
                           "values": [round(float(s.get(y)), 1) if y in s.index else None for y in x]})
        return {"kind": "lines", "title": "Yearly saving per price year, per size class",
                "subtitle": "The same battery and contract replayed on the prices of each year, 2027–2029 rules."
                            + (" `forecast` only runs on the last 3 years, so its battery shows its best strategy "
                               "with the full history." if swapped else ""),
                "x": x, "xFormat": "year", "unit": "€", "yLabel": "€ per year", "zero": True, "endLabels": True,
                "series": series, "height": 320}

    # ------------------------------------------------------------------ earnings per strategy
    def earnings_years(self, b: Battery, c: Contract, meta, year_label):
        """Per ownership year and strategy: the saving split into its parts, using one
        price year (2027 rules for 2027-2029, 2030 rules after)."""
        cap = self.conn.battery_cap_w(b.phases)
        m = meta.get((b.id, c.id), {"windows": [], "wear": self.wear_for(b)[0]})
        comp, efcs = {}, {}
        for strat in self.strategies_for(c):
            if strat == "timed" and not m["windows"]:
                continue
            for scn in ("nosal_min50", "nosal_2030"):
                pers = self.periods(c, scn, [year_label] if c.is_dynamic else None)
                if not pers:
                    continue
                _, per, idx, _ = pers[-1]
                r = self.sim_strategy(strat, b, cap, per, idx, c, scn, m["wear"], windows=m["windows"])
                sv = self.saving(r, per, c, scn)
                u, s_ = marginal_values(per, c, self.regimes[scn], self.taxes, self.net_importer)
                sb = (b.standby_w or 0.0) / 1000.0 * 0.25
                parts = extras.earnings_split(per.imp - per.exp, r.batt_ac, sb, u, s_, sv)
                cut = float((r.curtailed * -s_).sum()) if r.curtailed is not None else 0.0
                parts["avoided_negative_export"] = cut
                parts["standby_and_other"] -= cut
                comp[(strat, scn)] = parts
                efcs[(strat, scn)] = r.efc
        start = pd.Timestamp(self.cfg["analysis"]["purchase_date"])
        rows = []
        for strat in dict.fromkeys(k[0] for k in comp):
            efc = efcs.get((strat, "nosal_min50"), 250.0)
            life = self.life_years(b, efc)
            t, year = 0.0, start.year
            first = (pd.Timestamp(year=year + 1, month=1, day=1) - start).days / 365.0
            while t < life - 1e-9:
                frac = min(first if t == 0 else 1.0, life - t)
                scn = regime_for_year(self.cfg, year)
                parts = comp.get((strat, scn)) or comp.get((strat, "nosal_min50"))
                capf = 1.0 - (1.0 - b.eol_capacity) * min(efc * (t + frac / 2) / (b.cycle_life or 1e12), 1.0)
                row = {"strategy": strat, "year": year, "rules": self.regimes[scn]["label"],
                       "full_year": frac > 0.999}
                for k, _ in self.EARN_KEYS:
                    row[k] = parts[k] * frac * capf
                row["net_saving"] = sum(parts[k] for k, _ in self.EARN_KEYS) * frac * capf
                rows.append(row)
                t += frac
                year += 1
        return pd.DataFrame(rows)

    def earnings_focus(self, meta):
        """Average full ownership year per strategy for each size class (the partial
        purchase year and the partial last year are left out)."""
        c = self.cheapest_dynamic() or self.current
        if c is None or not getattr(self, "focus", None):
            return None
        year = self.head_years[-1] if (c.is_dynamic and self.head_years) else "profile"
        avg_rows, specs, per_year = [], [], []
        for f in self.focus:
            tbl = self.earnings_years(f["b"], c, meta, year)
            if tbl.empty:
                continue
            per_year.append(tbl.assign(size_class=f["cls"], battery=f["b"].name))
            full = tbl[tbl.full_year]
            if full.empty:
                full = tbl
            avg = full.groupby("strategy", sort=False)[[k for k, _ in self.EARN_KEYS] + ["net_saving"]].mean()
            yrs = full.groupby("strategy", sort=False).year.agg(["min", "max"])
            for strat, r in avg.iterrows():
                avg_rows.append({"size_class": f["cls"], "battery": f["b"].name, "strategy": strat,
                                 **{k: r[k] for k, _ in self.EARN_KEYS}, "net_saving": r["net_saving"],
                                 "years": f"{yrs.loc[strat, 'min']}–{yrs.loc[strat, 'max']}"})
            specs.append((f, avg))
        if not avg_rows:
            return None
        df = pd.DataFrame(avg_rows)
        lo = min(0.0, df[[k for k, _ in self.EARN_KEYS if k != "avoided_import"]].clip(upper=0).sum(axis=1).min())
        hi = df[["avoided_import", "sold_to_grid"]].clip(lower=0).sum(axis=1).max()
        charts_ = {}
        for f, avg in specs:
            charts_[f"earnings_{f['cls']}"] = {
                "kind": "bars", "stacked": True, "rotate": True, "unit": "€", "yLabel": "€ per year",
                "title": f"{f['cls']}: {f['b'].name} – average full year per strategy",
                "subtitle": f"{year} prices; 2027–2029 rules until 2029, 2030+ rules after; capacity fading included. "
                            "Gains stack upwards, costs downwards; the dot is the net saving.",
                "categories": list(avg.index), "ydomain": [lo, hi],
                "series": [{"label": lab, "color": self.EARN_COLORS[i], "values": [round(float(v), 1) for v in avg[k]]}
                           for i, (k, lab) in enumerate(self.EARN_KEYS)],
                "total": {"label": "Net saving", "values": [round(float(v), 1) for v in avg["net_saving"]]}}
        return df, charts_, pd.concat(per_year, ignore_index=True), year, c

    # ------------------------------------------------------------------ gap to perfect foresight
    GAP_ROWS = [
        ("self_consumption", "nothing: store surplus, use it when the house imports", "rec", {}),
        ("timed", "fixed windows, chosen on the last price year (in-sample)", "rec", {}),
        ("dynamic", "today's day-ahead prices + yesterday's usage", "rec", {}),
        ("dynamic_sell", "same, may sell to the grid", "rec", {}),
        ("forecast", "prices until tomorrow night + usage/solar of the last 3 days", "rec", {}),
        ("forecast_curtail", "forecast + solar export switched off at negative prices", "rec", {}),
        ("forecast, perfect usage forecast", "same optimiser, knows the real usage and solar", "oracle", {}),
        ("bound: solar only, no selling", "everything; stores only solar, covers only own load", "pf",
         {"grid_charge": False, "sell": False}),
        ("bound: solar only, may sell", "everything; stores only solar, may sell it", "pf", {"grid_charge": False}),
        ("bound: grid charging, no selling", "everything; may charge from the grid, no selling", "pf",
         {"sell": False}),
        ("perfect_foresight", "everything in advance, all freedoms", "rec", {}),
    ]

    def strategy_gap(self, meta) -> pd.DataFrame | None:
        """Where the gap to perfect foresight comes from, per size class (2027–2029 rules,
        latest full price year): the same optimiser with less knowledge, and perfect
        foresight with fewer freedoms."""
        c = self.cheapest_dynamic()
        if c is None or not getattr(self, "focus", None) or not self.head_years or not lp_available():
            return None
        scn, year = "nosal_min50", self.head_years[-1]
        pers = self.periods(c, scn, [year])
        if not pers:
            return None
        _, per, idx, _ = pers[-1]
        net = per.imp - per.exp
        u, s = marginal_values(per, c, self.regimes[scn], self.taxes, self.net_importer)
        rec = self.records
        fcfg = self.cfg.get("strategies", {}).get("forecast", {})
        out = {name: {"strategy": name, "knows": knows} for name, knows, _, _ in self.GAP_ROWS}
        for f in self.focus:
            b = f["b"]
            cap = self.conn.battery_cap_w(b.phases)
            m = meta.get((b.id, c.id), {"windows": [], "wear": self.wear_for(b)[0]})
            mine = rec[(rec.battery_id == b.id) & (rec.contract_id == c.id) & (rec.scenario == scn)
                       & (rec.price_year == year)].set_index("strategy").saving
            for name, _, kind, kw in self.GAP_ROWS:
                if kind == "rec":
                    v = mine.get(name, np.nan)
                elif kind == "oracle":
                    r = simulate(net, plan_forecast(idx, net, u, s, b, cap, m["wear"], fcfg, forecast=net), b, cap)
                    v = self.saving(r, per, c, scn)
                else:
                    v = self.saving(perfect_foresight(net, u, s, b, cap, 0.0, **kw), per, c, scn)
                out[name][f["cls"]] = v
        df = pd.DataFrame(list(out.values()))
        cls = [f["cls"] for f in self.focus]
        df = df.dropna(subset=cls, how="all").reset_index(drop=True)
        self.gap_year = year
        return df

    def gap_text(self, gap: pd.DataFrame) -> str:
        ch = getattr(self, "chosen", None)
        col = next((f["cls"] for f in self.focus if ch is not None and f["b"].id == ch.id), self.focus[0]["cls"])
        v = dict(zip(gap.strategy, gap[col]))
        g = lambda k: v.get(k, float("nan"))  # noqa: E731
        sizes = ", ".join(f"{f['cls']} €{dict(zip(gap.strategy, gap[f['cls']])).get('forecast', float('nan')):,.0f}"
                          for f in self.focus)
        lines = [f"#### Why perfect foresight earns more ({col} battery, {self.gap_year} prices)",
                 f"- **Timing** is the biggest piece: storing only solar and only covering the house, perfect "
                 f"knowledge earns €{g('bound: solar only, no selling'):,.0f} vs €{g('self_consumption'):,.0f} "
                 "for self-consumption: it absorbs solar in the cheapest (often negative) export hours and keeps "
                 "it for the most expensive hours.",
                 f"- **Selling stored solar** adds €{g('bound: solar only, may sell') - g('bound: solar only, no selling'):,.0f}; "
                 f"**grid charging** adds €{g('bound: grid charging, no selling') - g('bound: solar only, no selling'):,.0f} "
                 "(perfect foresight ignores wear).",
                 f"- **Forecasting is the gap:** with a perfect usage/solar forecast the optimiser reaches "
                 f"€{g('forecast, perfect usage forecast'):,.0f}; with last-days forecasts €{g('forecast'):,.0f}. "
                 "A solar forecast (Forecast.Solar, Solcast) in Home Assistant closes much of it.",
                 f"- **Bigger batteries gain more from `forecast`:** {sizes}."]
        rec = self.records
        if ch is not None:
            t = rec[(rec.battery_id == ch.id) & (rec.scenario == "nosal_min50") & ~rec.partial
                    & rec.price_year.isin(self.head_years)]
            p = t.pivot_table(index="price_year", columns="strategy", values="saving", aggfunc="mean")
            if {"timed", "dynamic"} <= set(p.columns):
                wins = [str(y) for y in p.index if p.loc[y, "timed"] > p.loc[y, "dynamic"]]
                lines.append(f"- **`timed` vs `dynamic`:** timed wins in {', '.join(wins) or 'no year'}; its windows "
                             f"are picked on {self.head_years[-1]} prices (in-sample) and its grid charging ignores "
                             "wear.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ 11.6 Objective 4
    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def step_kiln(self):
        sec = Section("kiln", "10. Pottery kiln on free power")
        k = dict(self.cfg.get("kiln", {}) or {})
        k.update(self.opts.kiln or {})
        powers = [float(p) for p in k.get("powers_kw", [1.5, 2.0, 3.0, 3.6])]
        one_phase = self.conn.phases == 1
        if one_phase:
            powers = [p for p in powers if p <= float(k.get("single_phase_max_kw", 3.68))]
        hours, duty = float(k.get("firing_hours", 8)), float(k.get("avg_duty", 0.7))
        starts, tol = [int(h) for h in k.get("start_hours", [7, 8, 9, 10])], float(k.get("free_tolerance", 0.05))
        dp = self.contract_prices(self.cheapest_dynamic())
        price = dp[1] if dp else self.avg_prices()[0]["nosal_min50"][0]
        # Surplus solar only: the measured export, no battery.
        nb = extras.kiln_days(self.profile, None, 0, None, powers, hours, duty, starts, tol)
        fp = self.fixed_prices()
        rows = []
        for p, g in nb.groupby("kiln_kw"):
            rows.append({"kiln_kw": p, "firing_kwh": g.firing_kwh.iloc[0], "free_days": int(g.free.sum()),
                         "avg_grid_kwh_per_firing": g.grid_kwh.mean(),
                         "avg_cost_per_firing": g.grid_kwh.mean() * price,
                         **({"avg_cost_per_firing_fixed": g.grid_kwh.mean() * fp[1]} if fp else {}),
                         "months_with_free_days": ", ".join(self.MONTHS[m - 1]
                                                            for m in sorted(g[g.free].month.unique()))})
        summ = pd.DataFrame(rows)
        sec.tables["kiln_free_firing_days"] = summ.round(2)
        by_month = nb[nb.free].groupby(["kiln_kw", "month"]).size().unstack(fill_value=0)
        by_month = by_month.reindex(columns=range(1, 13), fill_value=0)
        by_month = by_month.loc[:, by_month.sum() > 0]
        by_month.columns = [self.MONTHS[m - 1] for m in by_month.columns]
        if len(by_month.columns):
            sec.tables["free_days_per_month"] = by_month
        sec.interactive["kiln_days"] = {
            "kind": "bars", "unit": "days", "yLabel": "free firing days per year",
            "title": "Free firing days per year on surplus solar, by kiln power",
            "subtitle": f"A firing needs rated kW × {hours * duty:g} kWh ({hours:g} h at {duty:.0%} average power) with at "
                        f"most {tol:.0%} from the grid; one firing per day; no battery.",
            "categories": [f"{p:g} kW" for p in summ.kiln_kw],
            "series": [{"label": "Free firing days", "color": "#2a78d6", "values": summ.free_days.tolist()}]}
        good = summ[summ.free_days >= 20]
        if len(good):
            g = good.iloc[-1]
            sec.summary = (f"A kiln up to {g.kiln_kw:g} kW fires on surplus solar alone on {int(g.free_days)} days a "
                           f"year ({g.months_with_free_days}).")
        md = ["`firing energy = kiln kW × firing hours × average power share`",
              "`free day` = the whole firing runs on solar surplus (what you would otherwise export) with at most "
              f"{tol:.0%} from the grid",
              f"`cost per firing = grid kWh × €{price:.3f}` (average all-in import price"
              + (f" on {dp[0].label}" if dp else "") + ", 2027 rules)"
              + (f"; on the fixed offer {fp[0].label}: × €{fp[1]:.3f} (`avg_cost_per_firing_fixed`)" if fp else "")]
        md += [f"- **{r.kiln_kw:g} kW:** {r.free_days} free days" for r in summ.itertuples() if r.free_days > 0][:6]
        md += ["#### Assumptions",
               f"- A firing to maximum temperature takes {hours:g} h at {duty:.0%} of rated power on average "
               f"(config.yaml `kiln`; edit to your kiln's data sheet). Start hour chosen per day among {starts}.",
               "- Only surplus solar is used: no battery, so the battery results elsewhere are unaffected.",
               f"- Surplus solar is not entirely free: from 2027 each kWh gives up about "
               f"€{self.avg_prices()[0]['nosal_min50'][1]:.3f} of feed-in compensation.",
               "- Kilns above 3.68 kW (16 A) need a three-phase or dedicated high-current group; "
               + ("left out on this single-phase connection." if one_phase else
                  f"your {self.conn.label} connection allows them.")]
        sec.md = "\n".join(md)
        self.emit(sec)

    # ------------------------------------------------------------------ 11. solar panels
    def solar_production(self):
        """(production per interval kWh, kWp, kWp estimated?, source text) for the profile year."""
        if "solar_prod" in self.results:
            return self.results["solar_prod"]
        sc = dict(self.cfg.get("solar", {}) or {})
        exp = self.profile["exp"].values
        if self.pv is not None:
            prod = np.maximum(self.pv.fillna(0).values, exp)
            kwp = sc.get("kwp") or prod.sum() / float(sc.get("yield_kwh_per_kwp", 900))
            out = (prod, float(kwp), False, "your PV production file")
        else:
            prod, kwp, est = extras.pv_estimate(self.profile.index, exp, sc.get("kwp"),
                                                float(sc.get("yield_kwh_per_kwp", 900)),
                                                float(sc.get("self_share_if_unknown", 0.3)),
                                                float(sc.get("latitude", 52.1)), float(sc.get("longitude", 5.1)))
            src = (f"estimated: {kwp:.1f} kWp " + ("(guessed from your export – set the real kWp)" if est
                                                   else "(your setting)") + " with a clear-sky model")
            out = (prod, kwp, est, src)
        self.results["solar_prod"] = out
        return out

    def step_solar(self):
        sec = Section("solar", "11. Solar panels: yearly saving and payback")
        sc = dict(self.cfg.get("solar", {}) or {})
        pr = self.profile
        imp, exp = pr["imp"].values, pr["exp"].values
        prod, kwp, est, src = self.solar_production()
        self_used = prod - exp                                  # solar used in the house (kWh per interval)
        month_share = pd.Series(prod, index=pr.index).groupby(pr.index.month).sum()
        month_share = month_share / month_share.sum()
        install = pd.Timestamp(sc.get("install_date", "2024-07-01"))
        fixed_until = pd.Timestamp(sc.get("fixed_contract_until", "2027-01-01"))
        life = int(sc.get("lifetime_years", 25))
        deg = float(sc.get("degradation_pct_per_year", 0.5)) / 100.0
        price = sc.get("price_eur")
        price = float(price) if price not in (None, "") else None
        dyn = self.cheapest_dynamic()
        fixed = self.current

        def saving_for(year, after=None):
            """(contract label, rules label, € saved in a full year); `after` = contract from 2027."""
            when = pd.Timestamp(year=year, month=7, day=1)
            if when < fixed_until and fixed is not None:
                per = Period(imp, exp, pr["is_low"].values, None, len(imp) / 96.0, year)
                scn = "saldering"
                c = fixed
                with_ = compute_cost(per, c, self.regimes[scn], self.taxes, self.conn.label, include_fixed=False).total
                without = compute_cost(per.with_flows(imp + self_used, np.zeros_like(exp)), c, self.regimes[scn],
                                       self.taxes, self.conn.label, include_fixed=False).total
                return c.label, f"saldering ({year} energy tax)", without - with_
            c = after or dyn or fixed
            if c is None:
                return "–", "–", np.nan
            scn = regime_for_year(self.cfg, max(year, 2027))
            vals = []
            for label, per, idx, _ in self.periods(c, scn, self.head_years if c.is_dynamic else None):
                su = self_used if label == "profile" else self_used[self.results[("replay", label)][1]]
                w = compute_cost(per, c, self.regimes[scn], self.taxes, self.conn.label, include_fixed=False).total
                wo = compute_cost(per.with_flows(per.imp + su, np.zeros_like(per.exp)), c, self.regimes[scn],
                                  self.taxes, self.conn.label, include_fixed=False).total
                vals.append(wo - w)
            return c.label, self.regimes[scn]["label"], float(np.mean(vals)) if vals else np.nan

        cache, rows = {}, []
        end = install + pd.DateOffset(years=life)
        cum = -(price or 0.0)
        cum_fx = cum
        fx_alt = self.cheapest_fixed()
        if fx_alt is not None and dyn is not None and fx_alt.id == dyn.id:
            fx_alt = None
        for year in range(install.year, end.year + 1):
            a = max(install, pd.Timestamp(year=year, month=1, day=1))
            z = min(end, pd.Timestamp(year=year + 1, month=1, day=1))
            if z <= a:
                continue
            months = pd.date_range(a, z - pd.Timedelta(days=1), freq="D").month.unique()
            share = float(month_share.reindex(months).fillna(0).sum())
            key = "fixed" if pd.Timestamp(year=year, month=7, day=1) < fixed_until and fixed is not None \
                else regime_for_year(self.cfg, max(year, 2027))
            if key == "fixed":
                key = ("fixed", year)               # energy tax differs per year
            if key not in cache:
                cache[key] = saving_for(year)
            label, rules, full = cache[key]
            if isinstance(key, tuple):
                rules = f"saldering ({year} energy tax)"
            age = max(0.0, (pd.Timestamp(year=year, month=7, day=1) - install).days / 365.0)
            s = full * share * (1 - deg) ** age
            cum += s
            # the alternative: the cheapest fixed offer instead of dynamic from 2027
            if isinstance(key, tuple) or fx_alt is None:
                s_fx = s
            else:
                k2 = ("fx", key)
                if k2 not in cache:
                    cache[k2] = saving_for(year, after=fx_alt)
                s_fx = cache[k2][2] * share * (1 - deg) ** age
            cum_fx += s_fx
            rows.append({"year": year, "contract": label, "rules": rules, "share_of_year": share,
                         "production_kwh": prod.sum() * share * (1 - deg) ** age,
                         "self_used_kwh": self_used.sum() * share * (1 - deg) ** age,
                         "exported_kwh": exp.sum() * share * (1 - deg) ** age,
                         "saving_eur": s, "cumulative_eur": cum,
                         "saving_if_fixed_eur": s_fx, "cumulative_if_fixed_eur": cum_fx})
        df = pd.DataFrame(rows)
        if fx_alt is None:
            df = df.drop(columns=["saving_if_fixed_eur", "cumulative_if_fixed_eur"])
        sec.tables["solar_saving_per_year"] = df.round(2)
        payback_year = None
        if price:
            prev = -price
            for r in df.itertuples():
                if r.cumulative_eur >= 0 > prev:
                    frac = -prev / r.saving_eur if r.saving_eur else 0
                    a = max(install, pd.Timestamp(year=r.year, month=1, day=1))
                    payback_year = a.year + (a.dayofyear - 1) / 365.0 + frac * r.share_of_year
                    break
                prev = r.cumulative_eur
        today = pd.Timestamp.now(tz=None).normalize()
        so_far = df[df.year < today.year].saving_eur.sum()
        cur = df[df.year == today.year]
        if len(cur) and today > install:
            start = max(install, pd.Timestamp(year=today.year, month=1, day=1))
            months = pd.date_range(start, today - pd.Timedelta(days=1), freq="D").month.unique()
            done = float(month_share.reindex(months).fillna(0).sum())
            so_far += float(cur.saving_eur.iloc[0]) * done / max(float(cur.share_of_year.iloc[0]), 1e-9)
        # ---- the other option: panels plus the recommended battery (bought on the purchase date)
        bat = self.solar_battery_path()
        if bat is not None:
            bname, bprice, bcum, bstart = bat
            def bat_cum(x):                      # battery: cumulative saving − price at decimal year x
                return float(bcum(x)) if x >= bstart else 0.0
            starts = [max(install, pd.Timestamp(year=y, month=1, day=1)) for y in df.year]
            ends = [min(end, pd.Timestamp(year=y + 1, month=1, day=1)) for y in df.year]
            dec = lambda ts: ts.year + (ts.dayofyear - 1) / 365.0  # noqa: E731
            bsave = []
            for a, z in zip(starts, ends):
                s_ = bat_cum(dec(z)) - bat_cum(dec(a))
                if dec(a) <= bstart < dec(z):
                    s_ += bprice                 # the purchase is not a saving
                bsave.append(s_)
            df["battery_saving_eur"] = bsave
            df["total_saving_eur"] = df.saving_eur + df.battery_saving_eur
            df["cumulative_with_battery_eur"] = df.cumulative_eur + [bat_cum(dec(z)) for z in ends]
            sec.tables["solar_saving_per_year"] = df.round(2)
            pb2 = None
            prev, prev_x = -(price or 0.0), dec(install)
            for z, v in zip(ends, df.cumulative_with_battery_eur):
                if price and v >= 0 > prev:
                    pb2 = prev_x + (dec(z) - prev_x) * (-prev) / (v - prev)
                    break
                prev, prev_x = v, dec(z)
        pb_fx = None
        if "cumulative_if_fixed_eur" in df and price:
            prev, prev_x = -price, install.year + (install.dayofyear - 1) / 365.0
            for yr, v in zip(df.year, df.cumulative_if_fixed_eur):
                zx = float(min(end, pd.Timestamp(year=yr + 1, month=1, day=1)).year) if yr < end.year else \
                    end.year + (end.dayofyear - 1) / 365.0
                if v >= 0 > prev:
                    pb_fx = prev_x + (zx - prev_x) * (-prev) / (v - prev)
                    break
                prev, prev_x = v, zx
        fixed_rows = df[df.contract == (fixed.label if fixed else "")]
        dyn_rows = df[(df.contract != (fixed.label if fixed else "")) & (df.share_of_year > 0.99)]
        f_full = fixed_rows[fixed_rows.share_of_year > 0.99].saving_eur.mean() if len(fixed_rows) else np.nan
        d_full = dyn_rows.saving_eur.iloc[0] if len(dyn_rows) else np.nan
        sec.summary = (f"The panels save about €{f_full:,.0f} a year on the fixed contract (saldering) and "
                       f"€{d_full:,.0f} a year from 2027 on the dynamic contract; "
                       + (f"paid back in {payback_year:.1f}." if payback_year else
                          "set the price paid to see the payback year." if not price else
                          "not paid back within their lifetime.")
                       + (f" With the {bname} as well: paid back in {pb2:.1f}." if bat is not None and price and pb2
                          else ""))
        md = ["`saving = yearly cost without panels − yearly cost with panels` (same contract and rules; without "
              "panels you would import everything the house used)",
              "`house use = import + solar used directly`, `solar used directly = production − export`",
              f"Contracts: **{fixed.label if fixed else '–'}** until {fixed_until.date()} (saldering), then "
              f"**{dyn.label if dyn else '–'}** (2027–2029 rules, then 2030+ rules).",
              f"- Production {prod.sum():,.0f} kWh a year ({src}); used directly {self_used.sum():,.0f} kWh "
              f"({self_used.sum() / max(prod.sum(), 1e-9):.0%}), exported {exp.sum():,.0f} kWh.",
              f"- Saved so far (from {install.date()} to today): about **€{so_far:,.0f}**."]
        if price:
            md.append(f"- Price paid €{price:,.0f}: " + (f"**paid back in {payback_year:.1f}**." if payback_year
                                                          else "not paid back within the lifetime."))
        else:
            md.append("- **No price set:** enter what the panels cost (settings, or `solar.price_eur` in "
                      "config.yaml) to see the payback year; the cumulative line now starts at €0.")
        if "saving_if_fixed_eur" in df:
            d27 = df[(df.year >= 2027) & (df.share_of_year > 0.99)]
            if len(d27):
                md.append(f"- **If you take the fixed offer {fx_alt.label} from 2027 instead of {dyn.label}:** the panels save "
                          f"€{d27.saving_if_fixed_eur.iloc[0]:,.0f} a year instead of €{d27.saving_eur.iloc[0]:,.0f}"
                          + (f"; paid back in {pb_fx:.1f}" if price and pb_fx else "")
                          + ". The panels save more on an expensive fixed price, but the total bill is still higher "
                            "(section 4).")
        if bat is not None:
            extra = df[df.year >= 2027].battery_saving_eur
            src = self.chosen.best_price(float(self.cfg.get("battery_defaults", {}).get("de_travel_cost_eur", 0)))[1] \
                if self.use_de else "NL"
            md.append(f"- **With the {bname}** (bought {pd.Timestamp(self.cfg['analysis']['purchase_date']).date()} for "
                      f"€{bprice:,.0f}{' in Germany' if src == 'DE' else ''}): it adds about €{extra[extra > 0].iloc[:3].mean():,.0f} a year from 2027 on "
                      "top of the panels" + (f"; panels and battery together are paid back in **{pb2:.1f}** "
                                             f"(panels alone {payback_year:.1f})." if price and pb2 and payback_year
                                             else "."))
        else:
            md.append("- Without a battery with a price there is no 'panels + battery' comparison.")
        if est and self.pv is None:
            md.append("- **kWp unknown:** set the real system size for a better production estimate.")
        elif self.pv is None and kwp and prod.sum() > float(kwp) * float(sc.get("yield_kwh_per_kwp", 900)) * 1.05:
            raised = prod.sum() / (float(kwp) * float(sc.get("yield_kwh_per_kwp", 900))) - 1
            md.append(f"- **Yield:** to cover your measured export the panels must produce about "
                      f"{prod.sum() / float(kwp):,.0f} kWh per kWp a year, {raised:.0%} above the "
                      f"{float(sc.get('yield_kwh_per_kwp', 900)):g} kWh default – normal for a good south-facing roof "
                      "(set `solar.yield_kwh_per_kwp` if you know it, or add inverter data for exact numbers).")
        md += ["#### Why saldering makes the panels worth more until 2027",
               "- With saldering every exported kWh is netted against an imported kWh at the full price incl. "
               "energy tax and VAT, so the whole production is worth the import price.",
               "- From 2027 only solar used directly saves the full import price; exported solar earns the "
               "(low, midday) dynamic feed-in price. A battery or shifting use to sunny hours raises the saving.",
               "#### Assumptions",
               f"- Every year uses your profile year (Oct 2025 – Sep 2026) for usage and export; dynamic years use "
               f"the average of the {', '.join(map(str, self.head_years))} prices.",
               "- The install year and the last year count only the months covered, weighted by how much the panels "
               "produce in those months.",
               f"- Panels lose {deg:.1%} output a year; lifetime {life} years. Fixed costs (standing charge, grid, "
               "tax reduction) are the same with and without panels and are left out.",
               "- Energy tax per year from config.yaml (2024, 2025, 2026; later years held flat)."]
        sec.md = "\n".join(md)
        cats = [str(y) for y in df.year]
        is_fixed = df.contract == (fixed.label if fixed else "")
        sec.interactive["solar_yearly"] = {
            "kind": "bars", "stacked": True, "unit": "€", "yLabel": "€ saved per year", "rotate": True,
            "title": "What the panels save each year",
            "subtitle": "Partial first and last year count only the months covered.",
            "categories": cats,
            "series": [{"label": f"Fixed contract with saldering ({fixed.label if fixed else '–'})",
                        "color": "#2a78d6", "values": [round(v, 1) if f else 0 for v, f in zip(df.saving_eur, is_fixed)]},
                       {"label": f"Dynamic contract ({dyn.label if dyn else '–'})", "color": "#eb6834",
                        "values": [0 if f else round(v, 1) for v, f in zip(df.saving_eur, is_fixed)]}]}
        if bat is not None:
            sec.interactive["solar_yearly"]["series"].append(
                {"label": f"Extra with the {bname}", "color": "#1baf7a",
                 "values": [round(v, 1) for v in df.battery_saving_eur]})
            sec.interactive["solar_yearly"]["title"] = "What the panels (and the battery on top) save each year"
        x = [install.year + (install.dayofyear - 1) / 365.0] + [
            float(max(install, pd.Timestamp(year=y + 1, month=1, day=1)).year) for y in df.year[:-1]] + [
            end.year + (end.dayofyear - 1) / 365.0]
        vals = [-(price or 0.0)] + [round(v, 1) for v in df.cumulative_eur]
        spec = {"kind": "lines", "unit": "€", "yLabel": "€ cumulative", "zero": True, "markers": False,
                "title": "Cumulative: when the panels have earned themselves back" if price else
                         "Cumulative saving of the panels",
                "subtitle": (f"Starts at minus the price (€{price:,.0f}) on {install.date()}." if price else
                             "No price set: starts at €0.") + " Fixed contract until 2027, dynamic after.",
                "x": [round(v, 3) for v in x], "xFormat": "year",
                "xNames": [f"installed {install.date()}"] + [f"1 Jan {round(v)}" for v in x[1:-1]] + ["end of life"],
                "xTicks": list(range(install.year + 1, end.year + 1, 2)),
                "series": [{"label": "cumulative saving − price" if price else "cumulative saving",
                            "color": "#2a78d6", "values": vals}]}
        if payback_year:
            spec["points"] = [{"x": round(payback_year, 3), "y": 0, "color": "#2a78d6",
                               "label": f"paid back {payback_year:.1f}"}]
        if bat is not None:
            # insert the battery purchase so the line drops there by the battery price
            xs = list(x)
            panels = list(vals)
            if xs[0] < bstart < xs[-1] and bstart not in xs:
                i = next(k for k, v in enumerate(xs) if v > bstart)
                f = (bstart - xs[i - 1]) / (xs[i] - xs[i - 1])
                xs.insert(i, bstart)
                panels.insert(i, round(panels[i - 1] + f * (panels[i] - panels[i - 1]), 1))
            names = [f"installed {install.date()}"] + [
                f"battery bought {pd.Timestamp(self.cfg['analysis']['purchase_date']).date()}" if abs(v - bstart) < 1e-9
                else f"1 Jan {round(v)}" for v in xs[1:-1]] + ["end of life"]
            spec.update({"x": [round(v, 3) for v in xs], "xNames": names, "endLabels": True,
                         "title": "Cumulative: panels alone vs panels + battery",
                         "subtitle": spec["subtitle"] + f" The second line also buys the {bname} "
                                     f"(€{bprice:,.0f}) on {pd.Timestamp(self.cfg['analysis']['purchase_date']).date()} "
                                     "and adds its yearly saving (best strategy, capacity fading, until its end of life)."})
            spec["series"] = [dict(spec["series"][0], label="Panels only", short="panels only", values=panels),
                              {"label": f"Panels + {bname}", "short": "panels + battery", "color": "#1baf7a",
                               "values": [round(p_ + bat_cum(v), 1) for p_, v in zip(panels, xs)]}]
            if price and pb2:
                spec.setdefault("points", []).append({"x": round(pb2, 3), "y": 0, "color": "#1baf7a",
                                                      "label": f"with battery {pb2:.1f}"})
        if "cumulative_if_fixed_eur" in df:
            fx_vals = [-(price or 0.0)] + list(df.cumulative_if_fixed_eur)
            grid = spec["x"]
            spec["series"].append({"label": f"Panels only, {fx_alt.label} from 2027", "short": "panels, fixed 2027+",
                                   "color": "#4a3aa7",
                                   "values": [round(float(np.interp(v, x, fx_vals)), 1) for v in grid]})
            spec["endLabels"] = True
            if price and pb_fx:
                spec.setdefault("points", []).append({"x": round(pb_fx, 3), "y": 0, "color": "#4a3aa7",
                                                      "label": f"fixed {pb_fx:.1f}"})
        sec.interactive["solar_cumulative"] = spec
        self.emit(sec)

    def solar_battery_path(self):
        """The recommended battery's money path for the solar section:
        (name, price, cumulative(decimal year) -> € saved − price, purchase as decimal year),
        or None without a priced battery."""
        summary = getattr(self, "payback_summary", None)
        rk = summary.get("ranked") if summary else None
        b = getattr(self, "chosen", None)
        if rk is None or rk.empty or b is None:
            return None
        r = self.achievable_ranked(rk)
        r = r[r._bid == b.id]
        if r.empty:
            return None
        r = r.iloc[0]
        start = pd.Timestamp(self.cfg["analysis"]["purchase_date"])
        s0 = start.year + (start.dayofyear - 1) / 365.0
        ts = np.array([s0 + t for t, _ in r["_path"]])
        vs = np.array([v for _, v in r["_path"]])
        return b.name, float(r["price_eur"]), (lambda x: np.interp(x, ts, vs)), s0

    # ------------------------------------------------------------------ 12. extension options
    EXT_COLORS = ["#4a3aa7", "#eb6834", "#1baf7a"]

    def step_extension(self):
        sec = Section("extension", "12. Extra panels + battery: what an extension earns")
        ec = dict(self.cfg.get("extension", {}) or {})
        opts = [o for o in (ec.get("options") or []) if o and (float(o.get("kwp") or 0) > 0
                                                               or float(o.get("battery_kwh") or 0) > 0)]
        if not ec.get("enabled", True) or not opts:
            sec.md = "Switched off (settings: *Extra panels + battery*)."
            self.emit(sec)
            return
        c = self.cheapest_dynamic() or self.current
        fx = self.cheapest_fixed()
        if c is None:
            sec.md = "No contract to calculate with."
            self.emit(sec)
            return
        prod, kwp, _, src = self.solar_production()
        bc = dict(ec.get("battery", {}) or {})
        life = int(ec.get("lifetime_years", 15))
        deg = float(ec.get("pv_degradation_pct_per_year", 0.5)) / 100.0
        r_disc = float(self.cfg["analysis"].get("discount_rate", 0.03))
        start = pd.Timestamp(self.cfg["analysis"]["purchase_date"])
        rows, curves = [], []

        def run_option(o, contract, scenarios, years):
            extra_kwp, kwh = float(o.get("kwp") or 0), float(o.get("battery_kwh") or 0)
            inv = float(o.get("inverter_kw") or 0) or max(kwh * 0.5, 1.0)
            extra = prod * (extra_kwp / kwp) if kwp else np.zeros_like(prod)
            b = None
            if kwh > 0:
                b = Battery(id="ext", brand="Extension", model=str(o.get("name", "battery")), nominal_kwh=kwh,
                            usable_kwh=kwh * float(bc.get("dod", 0.9)), max_charge_w=inv * 1000,
                            max_discharge_w=inv * 1000, rte=float(bc.get("rte", 0.9)),
                            standby_w=float(bc.get("standby_w", 15)), cycle_life=float(bc.get("cycle_life", 6000)),
                            eol_capacity=float(bc.get("eol_capacity", 0.7)), warranty_years=life)
            strats = ["self_consumption"]
            if contract.is_dynamic and b is not None:
                strats += [s for s in ("forecast_curtail", "forecast") if s in self.strategies_for(contract)]
            out, used = {}, {}
            for scn in scenarios:
                vals = {s: [] for s in strats}
                for label, per, idx, _ in self.periods(contract, scn, years if contract.is_dynamic else None):
                    ex = extra if label == "profile" else extra[self.results[("replay", label)][1]]
                    base = compute_cost(per, contract, self.regimes[scn], self.taxes, self.conn.label,
                                        include_fixed=False).total
                    net = per.imp - per.exp - ex
                    p2 = per.with_flows(np.maximum(net, 0), np.maximum(-net, 0))
                    for s in strats:
                        if b is None:
                            f_imp, f_exp = p2.imp, p2.exp
                        else:
                            res = self.sim_strategy(s, b, inv * 1000, p2, idx, contract, scn,
                                                    float(bc.get("wear_eur_kwh", 0.08)))
                            f_imp, f_exp = res.imp, res.exp
                        cost = compute_cost(p2.with_flows(f_imp, f_exp), contract, self.regimes[scn], self.taxes,
                                            self.conn.label, include_fixed=False).total
                        vals[s].append(base - cost)
                mean = {s: float(np.mean(v)) for s, v in vals.items() if v}
                best = max(mean, key=mean.get)
                out[scn], used[scn] = mean[best], best
            return out, used, float(extra.sum())

        for i, o in enumerate(opts):
            price = float(o.get("price_eur") or 0)
            sav, used, extra_kwh = run_option(o, c, self.scenarios, self.head_years)
            # money path from the purchase date: rules per calendar year, panel degradation
            t, year, cum, npv, pay = 0.0, start.year, -price, -price, None
            first = (pd.Timestamp(year=year + 1, month=1, day=1) - start).days / 365.0
            xs, ys = [start.year + (start.dayofyear - 1) / 365.0], [round(-price, 1)]
            while t < life - 1e-9:
                frac = min(first if t == 0 else 1.0, life - t)
                scn = regime_for_year(self.cfg, year)
                g = sav.get(scn, sav.get("nosal_min50", 0.0)) * frac * (1 - deg) ** t
                if pay is None and cum + g >= 0 > cum and g > 0:
                    pay = t + frac * (-cum) / g
                cum += g
                npv += g / (1 + r_disc) ** (t + frac / 2)
                t += frac
                year += 1
                xs.append(round(xs[0] + t, 3))
                ys.append(round(cum, 1))
            row = {"option": o.get("name") or f"option {i + 1}", "extra_kwp": float(o.get("kwp") or 0),
                   "battery_kwh": float(o.get("battery_kwh") or 0), "inverter_kw": float(o.get("inverter_kw") or 0),
                   "price_eur": price, "extra_production_kwh": extra_kwh, "strategy": used.get("nosal_min50", ""),
                   "saving_2026_rules_eur": sav.get("saldering"), "saving_2027_eur": sav.get("nosal_min50"),
                   "saving_2030_eur": sav.get("nosal_2030"),
                   "payback_years": pay if pay is not None else np.inf, "npv_eur": npv,
                   "net_after_lifetime_eur": cum}
            if fx is not None and fx is not c:
                fsav, _, _ = run_option(o, fx, ["nosal_min50"], None)
                row["saving_2027_on_fixed_eur"] = fsav.get("nosal_min50")
            rows.append(row)
            curves.append((row["option"], xs, ys, self.EXT_COLORS[i % 3], pay))
        df = pd.DataFrame(rows)
        sec.tables["extension_options"] = df.round(2)
        best = df.loc[df.payback_years.idxmin()]
        sec.summary = (f"Fastest payback: {best.option} – €{best.price_eur:,.0f}, saves about "
                       f"€{best.saving_2027_eur:,.0f} a year from 2027, paid back in "
                       + (f"{best.payback_years:.1f} years." if np.isfinite(best.payback_years) else "never."))
        md = ["`yearly saving = cost now − cost with the extension` (same contract and rules; the extension is "
              "added to your current panels)",
              "`extra production = current production × extra kWp ÷ current kWp` (same roof, same weather)",
              f"Contract: **{c.label}** (2026 rules until the end of 2026, then 2027–2029, then 2030+); bought on "
              f"{start.date()}; lifetime {life} years."]
        for r in df.itertuples():
            pb = f"{r.payback_years:.1f} years" if np.isfinite(r.payback_years) else "not within its lifetime"
            fx_txt = (f"; on the fixed offer it would save €{r.saving_2027_on_fixed_eur:,.0f}"
                      if hasattr(r, "saving_2027_on_fixed_eur") and pd.notna(r.saving_2027_on_fixed_eur) else "")
            md.append(f"- **{r.option}** (€{r.price_eur:,.0f}): saves €{r.saving_2027_eur:,.0f} a year from 2027 "
                      f"(`{r.strategy}`), pays back in **{pb}**, net present value €{r.npv_eur:,.0f}{fx_txt}.")
        md += ["#### Assumptions",
               f"- Extra panels produce like your current ones ({src}); {deg:.1%} less output a year.",
               f"- Battery: usable = {float(bc.get('dod', 0.9)):.0%} of the stated kWh, round-trip efficiency "
               f"{float(bc.get('rte', 0.9)):.0%} (hybrid / DC-coupled), standby {float(bc.get('standby_w', 15)):g} W, "
               f"charge and discharge power = the inverter kW, wear €{float(bc.get('wear_eur_kwh', 0.08)):.2f} per kWh "
               "in the strategy's decisions.",
               "- Strategy: the best of self-consumption and the forecast optimiser (with zero-export curtailment) "
               "on the dynamic contract; self-consumption on a fixed contract.",
               "- More panels mainly add export in summer: from 2027 that earns the low midday price, so extra "
               "panels pay back slower without a battery to store their output.",
               "- The connection's export limit and the inverter's own clipping are not modelled; check them with "
               "the installer (3×25 A allows about 17 kW)."]
        sec.md = "\n".join(md)
        sec.interactive["extension_cumulative"] = {
            "kind": "lines", "unit": "€", "yLabel": "€ cumulative", "zero": True, "markers": False,
            "title": "Extension options: when each one has earned itself back",
            "subtitle": f"Starts at minus the price on {start.date()}, then adds each year's saving on {c.label}.",
            "x": sorted({x for _, xs, _, _, _ in curves for x in xs}), "xFormat": "year", "endLabels": True,
            "series": [], "points": []}
        grid = sec.interactive["extension_cumulative"]["x"]
        for name, xs, ys, color, pay in curves:
            sec.interactive["extension_cumulative"]["series"].append(
                {"label": name, "short": name, "color": color,
                 "values": [round(float(np.interp(x, xs, ys)), 1) if x <= xs[-1] + 1e-9 else None for x in grid]})
            if pay is not None:
                sec.interactive["extension_cumulative"]["points"].append(
                    {"x": round(xs[0] + pay, 3), "y": 0, "color": color, "label": f"{pay:.1f} yr"})
        self.emit(sec)

    # ------------------------------------------------------------------ advice (shown first)
    def step_advice(self):
        sec = Section("advice", "Advice – which battery, which contract, and why")
        summary = getattr(self, "payback_summary", None)
        rk = summary.get("ranked") if summary else None
        if rk is None or rk.empty or getattr(self, "chosen", None) is None:
            sec.md = "No battery with a price could be simulated, so there is no recommendation yet."
            self.emit(sec)
            return
        adv = self.advice_contract()
        ranked = self.achievable_ranked(rk)
        rec = self.records
        net = (self.profile["imp"] - self.profile["exp"]).values

        def describe(r):
            b = self.battery(r["_bid"])
            sim = simulate(net, SELF, b, self.conn.battery_cap_w(b.phases), keep_trace=True)
            daily = extras.daily_battery(self.profile.index, net, sim.batt_ac, sim.soc, b.usable_kwh,
                                         (b.standby_w or 0) / 4000.0)
            surplus = daily.solar_surplus_kwh.sum()
            return {"size_class": self.class_of(b)[0] or "other", "battery": b.name, "price_eur": r["price_eur"],
                    "eur_per_kwh_usable": r["eur_per_kwh_usable"], "usable_kwh": b.usable_kwh,
                    "bought_in": r.get("bought_in", "NL"),
                    "contract": r["contract"], "strategy": r["strategy"], "saving_eur_year_2027": r["_saving27"],
                    "payback_years": r["payback_years"], "npv_eur": r["npv_eur"],
                    "days_full": int(daily.full.sum()),
                    "days_below_half": int((daily.max_soc_kwh < 0.5 * b.usable_kwh).sum()),
                    "share_of_surplus_stored": daily.charged_from_solar_kwh.sum() / surplus if surplus else np.nan,
                    "_bid": b.id, "_cid": r["_cid"], "_model": self.short_name(b)}

        b = self.chosen
        brow = ranked[ranked._bid == b.id]
        row = pd.Series(describe((brow if len(brow) else ranked).iloc[0].to_dict()))
        classes = self.size_class_table(describe)
        if classes is not None:
            sec.tables["best_per_size_class"] = classes.round(2)
        comp = pd.DataFrame([describe(r) for r in ranked.head(6).to_dict("records")])
        sec.tables["battery_comparison"] = comp.round(2)
        sec.summary = (f"Best choice: {b.name}" + (f" on {adv.label}" if adv is not None else "")
                       + f" – pays back in {row.payback_years:.1f} years, saves about "
                       f"€{row.saving_eur_year_2027:,.0f} a year from 2027.")
        md = [f"### Recommendation: **{b.name}**" + (f" on **{adv.label}**" if adv is not None else ""),
              f"`payback {row.payback_years:.1f} years · saves €{row.saving_eur_year_2027:,.0f}/year from 2027 · "
              f"price €{row.price_eur:,.0f}{' in Germany' if row.bought_in == 'DE' else ''} "
              f"(€{row.eur_per_kwh_usable:,.0f} per usable kWh) · net present value "
              f"€{row.npv_eur:,.0f}`"]
        if self.base_strategy(row.strategy) in ("dynamic", "dynamic_sell", "forecast", "timed") \
                or row.strategy.endswith("_curtail"):
            simple = rec[(rec.battery_id == b.id) & (rec.contract_id == row._cid) & (rec.scenario == "nosal_min50")
                         & ~rec.partial & rec.price_year.isin(["profile"] + list(self.head_years))
                         & rec.strategy.isin(["self_consumption", "hbc_pv_first"])]
            alt = ""
            if len(simple):
                ss = simple.groupby("strategy").saving.mean().sort_values(ascending=False)
                alt = (f" Without it, plain `{ss.index[0]}` (battery app or Home Battery Control as-is) saves "
                       f"€{ss.iloc[0]:,.0f} a year.")
            extra = (" Curtailing also needs a solar inverter Home Assistant can scale back."
                     if row.strategy.endswith("_curtail") else "")
            md.append(f"- **Control:** strategy `{row.strategy}` needs custom control (section 7).{extra}{alt}")
        md.append(f"- **Fit:** full on {row.days_full} days a year, stores {row.share_of_surplus_stored:.0%} of your "
                  f"solar surplus; below half full on {row.days_below_half} (winter) days.")
        combo = self.combination_table(b)
        if combo is not None:
            sec.tables["contract_and_battery_combinations"] = combo
            best_c = combo.iloc[combo.yearly_cost_eur_2027.idxmin()]
            md.append(f"- **Cheapest combination from 2027:** {best_c.option}, about "
                      f"€{best_c.yearly_cost_eur_2027:,.0f} a year (today: €{combo.iloc[0].yearly_cost_eur_2027:,.0f}).")
            fx = self.cheapest_fixed()
            if fx is not None and fx is not self.current:
                fr = combo[combo.option.str.startswith(fx.label) & (combo.battery_saving_eur > 0)]
                if len(fr):
                    md.append(f"- **If you prefer a fixed price:** {fr.iloc[0].option} costs about "
                              f"€{fr.iloc[0].yearly_cost_eur_2027:,.0f} a year, €"
                              f"{fr.iloc[0].yearly_cost_eur_2027 - best_c.yearly_cost_eur_2027:,.0f} more than the "
                              "cheapest combination – the price of certainty.")
        if classes is not None and len(classes) > 1:
            md.append(self.size_class_text(classes))
        others = comp[comp._model != self.short_name(b)]
        if len(others):
            o = others.iloc[0]
            md += ["#### Runner-up of another model",
                   f"- {o.battery}: {o.payback_years:.1f} years payback, €{o.eur_per_kwh_usable:,.0f} per usable kWh."]
        self.advice_charts(sec, adv)
        sec.md = "\n".join(md) + sec.md
        self.emit(sec)

    def size_class_table(self, describe) -> pd.DataFrame | None:
        """The best-value battery per size class (see pick_focus) and what stepping up
        from the next smaller class adds."""
        rows, prev = [], None
        for f in getattr(self, "focus", []) or []:
            d = describe(f["row"].to_dict())
            row = {"size_class": f["cls"], "battery": d["battery"], "nominal_kwh": self.size_of(f["b"]),
                   "usable_kwh": d["usable_kwh"], "price_eur": d["price_eur"], "bought_in": d.get("bought_in", "NL"),
                   "eur_per_kwh_usable": d["eur_per_kwh_usable"], "strategy": d["strategy"],
                   "saving_eur_year_2027": d["saving_eur_year_2027"], "payback_years": d["payback_years"],
                   "npv_eur": d["npv_eur"], "lifetime_net_saving_eur": f["row"].get("lifetime_net_saving_eur", np.nan),
                   "days_full": d["days_full"]}
            if prev is not None:
                ds = row["saving_eur_year_2027"] - prev["saving_eur_year_2027"]
                dp = row["price_eur"] - prev["price_eur"]
                row["extra_saving_vs_smaller"] = ds
                row["extra_price_vs_smaller"] = dp
                row["payback_of_extra_years"] = (0.0 if dp <= 0 else dp / ds) if ds > 0 else np.inf
            rows.append(row)
            prev = row
        return pd.DataFrame(rows) if rows else None

    @staticmethod
    def size_class_text(classes: pd.DataFrame) -> str:
        best_npv = classes.loc[classes.npv_eur.idxmax()]
        fastest = classes.loc[classes.payback_years.idxmin()]
        lines = ["", "### Which size?",
                 "`payback of the extra = extra price ÷ extra saving per year` (stepping up one size class)"]
        for r in classes.itertuples():
            de = " in Germany" if getattr(r, "bought_in", "NL") == "DE" else ""
            s = (f"- **{r.size_class}:** {r.battery} – €{r.price_eur:,.0f}{de}, €{r.saving_eur_year_2027:,.0f}/yr "
                 f"(`{r.strategy}`), payback {r.payback_years:.1f} yr, NPV €{r.npv_eur:,.0f}")
            extra = getattr(r, "extra_saving_vs_smaller", np.nan)
            if isinstance(extra, float) and not np.isnan(extra):
                pe = r.payback_of_extra_years
                if r.extra_price_vs_smaller <= 0:
                    s += f"; costs €{-r.extra_price_vs_smaller:,.0f} less than the smaller class and saves €{extra:,.0f}/yr more"
                else:
                    s += (f"; the extra €{r.extra_price_vs_smaller:,.0f} earns €{extra:,.0f}/yr"
                          + (f" → {pe:.1f} yr" if np.isfinite(pe) else " → never"))
            lines.append(s)
        if best_npv.size_class == fastest.size_class:
            lines.append(f"- **{fastest.size_class}** wins on both payback and lifetime value.")
        else:
            lines.append(f"- Fastest payback: **{fastest.size_class}**; most money over its life: "
                         f"**{best_npv.size_class}** – pick it if you accept a longer wait.")
        return "\n".join(lines)

    def advice_contract(self):
        """The contract the advice is built on: the cheapest dynamic contract when price history exists."""
        return self.cheapest_dynamic() or self.current

    def combination_table(self, b):
        """Yearly cost from 2027 (last 3 price years) for the current contract, the cheapest new fixed
        offer and the cheapest dynamic contract, each with and without the battery."""
        e = getattr(self, "expected_by_contract", {}) or {}
        rec = self.records
        rows = []
        offers = [c for c in self.contracts if not c.is_dynamic and c is not self.current
                  and e.get(c.id, {}).get("nosal_min50") is not None]
        best_fixed = min(offers, key=lambda c: e[c.id]["nosal_min50"]) if offers else None
        for c in [x for x in (self.current, best_fixed, self.cheapest_dynamic()) if x is not None]:
            base = e.get(c.id, {}).get("nosal_min50")
            if base is None:
                continue
            sv = rec[(rec.battery_id == b.id) & (rec.contract_id == c.id) & (rec.scenario == "nosal_min50")
                     & ~rec.partial & rec.price_year.isin(["profile"] + list(self.head_years))
                     & (rec.strategy != "perfect_foresight")]
            best = sv.groupby("strategy").saving.mean().sort_values(ascending=False)
            rows.append({"option": f"{c.label}, no battery", "yearly_cost_eur_2027": base, "battery_saving_eur": 0.0,
                         "strategy": ""})
            if len(best):
                rows.append({"option": f"{c.label} + {b.name}", "yearly_cost_eur_2027": base - best.iloc[0],
                             "battery_saving_eur": best.iloc[0], "strategy": best.index[0]})
        return pd.DataFrame(rows).round(0) if rows else None

    def advice_charts(self, sec, c):
        """Per size class: day-by-day storage and saving in the latest full price year."""
        c = c or self.current or (self.contracts[0] if self.contracts else None)
        focus = getattr(self, "focus", None) or []
        if c is None or not focus:
            return
        scn = "nosal_min50"
        year = self.head_years[-1] if (c.is_dynamic and self.head_years) else "profile"
        pers = self.periods(c, scn, [year] if c.is_dynamic else None)
        if not pers:
            return
        _, per, idx, _ = pers[-1]
        net = per.imp - per.exp
        u, s_ = marginal_values(per, c, self.regimes[scn], self.taxes, self.net_importer)
        label = f"{year} prices" if year != "profile" else "your profile year"
        cum, overview, notes = [], [], []
        dx = None
        for f in focus:
            b, strat = f["b"], f["strategy"]
            cap = self.conn.battery_cap_w(b.phases)
            m = getattr(self, "_meta", {}).get((b.id, c.id), {})
            r = self.sim_strategy(strat, b, cap, per, idx, c, scn, self.wear_for(b)[0], windows=m.get("windows"),
                                  keep_trace=True)
            saving = pd.Series((per.imp - r.imp) * u - (per.exp - r.exp) * s_, index=idx).groupby(idx.date).sum()
            saving.index = pd.to_datetime(saving.index)
            soc = r.soc if r.soc is not None else np.zeros(len(net))
            daily = extras.daily_battery(idx, net, r.batt_ac, soc, b.usable_kwh, (b.standby_w or 0) / 4000.0)
            overview.append(daily.assign(saving_eur=saving.values, size_class=f["cls"], battery=b.name)
                            .reset_index(names="date"))
            dx = extras.ts(pd.DatetimeIndex(daily.index).tz_localize("Europe/Amsterdam"))
            cum.append(extras.series(f"{f['cls']}: {b.name} ({strat})", saving.cumsum().values, f["color"], width=2))
            notes.append(f"- **{f['cls']}:** €{saving.sum():,.0f} that year; best day €{saving.max():,.2f}; "
                         f"{int((saving <= 0.01).sum())} days with (almost) no saving.")
            sec.interactive[f"daily_saving_{f['cls']}"] = {
                "title": f"{f['cls']}: {b.name} – saving per day ({label}, strategy {strat})",
                "y": "€ per day", "daily": True, "height": 260, "x": dx,
                "group": "Saving per day, per size class",
                "series": [extras.series("Saving that day (€)", saving.values, f["color"], "bar")]}
            sec.interactive[f"daily_charge_{f['cls']}"] = {
                "title": f"{f['cls']}: {b.name} – solar surplus, storage and grid import per day ({label})",
                "y": "kWh per day", "daily": True, "height": 320, "x": dx,
                "group": "Solar surplus and storage per day, per size class",
                "series": [extras.series("Solar surplus (kWh)", daily.solar_surplus_kwh, "#eda100", "bar"),
                           extras.series("Stored from solar (kWh)", daily.charged_from_solar_kwh, "#1baf7a", "bar"),
                           extras.series("Charged from grid (kWh)", daily.charged_from_grid_kwh, "#e87ba4", "bar"),
                           extras.series("Grid import without battery (kWh)", daily.import_before_kwh, "#52514e"),
                           extras.series("Grid import with battery (kWh)", daily.import_after_kwh, "#2a78d6"),
                           extras.series(f"Usable capacity ({b.usable_kwh:.1f} kWh)", [b.usable_kwh] * len(daily),
                                         "#4a3aa7", dash=[6, 4])]}
        sec.interactive = {"cumulative_saving_year": {
            "title": f"Cumulative saving through the year, per size class ({label}, 2027–2029 rules, {c.label})",
            "y": "€ cumulative", "daily": True, "height": 320, "x": dx, "series": cum}, **sec.interactive}
        sec.csv_only.add("daily_overview")
        sec.tables["daily_overview"] = pd.concat(overview, ignore_index=True).assign(
            date=lambda d: pd.to_datetime(d.date).dt.strftime("%Y-%m-%d")).round(2)
        sec.md = "\n" + "\n".join(["#### Day by day in " + label + " (charts below)"] + notes)

    def bname(self, bid):
        return next((b.name for b in self.batteries if b.id == bid), bid)

    def cname(self, cid):
        return next((c.label for c in self.contracts if c.id == cid), cid)

    def payback_table(self, rec: pd.DataFrame, meta, contracts):
        d = self.cfg.get("battery_defaults", {})
        a = self.cfg["analysis"]
        sav_rows, ranked = [], []
        analyses = {"headline": self.head_years, "full": self.full_years}
        if rec.empty:
            return {"savings": pd.DataFrame(), "ranked": pd.DataFrame()}
        for (bid, cid, strat), g in rec.groupby(["battery_id", "contract_id", "strategy"]):
            b = next(x for x in self.batteries if x.id == bid)
            c = next(x for x in contracts if x.id == cid)
            for aname, years in analyses.items():
                if c.is_dynamic:
                    gg = g[g["price_year"].isin(years) & ~g["partial"]]
                else:
                    gg = g
                if gg.empty:
                    continue
                by_scn = gg.groupby("scenario")["saving"].mean().to_dict()
                efc = gg[gg.scenario == "nosal_min50"]["efc"].mean()
                delivered = gg[gg.scenario == "nosal_min50"]["delivered_kwh"].mean()
                sav_rows.append({"battery": b.name, "contract": c.label, "strategy": strat, "analysis": aname,
                                 **{f"saving_{self.regimes[s]['label']}": v for s, v in by_scn.items()},
                                 "efc_per_year": efc, "standby_eur_year_approx": (b.standby_w or 0) * 8.76 *
                                 self.avg_u_cache(), "estimated": "yes" if b.estimated else ""})
                variants = b.price_variants(d.get("de_travel_cost_eur", 0.0), self.cfg.get("blackfriday"), self.use_de)
                if self.opts.price_variant != "all":
                    variants = {k: v for k, v in variants.items() if k in (self.opts.price_variant, self.ref_variant)}
                src = b.best_price(d.get("de_travel_cost_eur", 0.0))[1]
                for vname, price in variants.items():
                    pb = payback(price, by_scn, efc, delivered, b, self.cfg)
                    per_year = []
                    if c.is_dynamic:
                        for y, gy in gg.groupby("price_year"):
                            sy = gy.groupby("scenario")["saving"].mean().to_dict()
                            per_year.append(payback(price, sy, gy[gy.scenario == "nosal_min50"]["efc"].mean(),
                                                    gy[gy.scenario == "nosal_min50"]["delivered_kwh"].mean(), b,
                                                    self.cfg)["payback_years"])
                    ranked.append({"battery": b.name, "usable_kwh": round(b.usable_kwh, 2), "contract": c.label,
                                   "strategy": strat, "analysis": aname, "price_variant": vname,
                                   "bought_in": src if vname == BEST_PRICE else ("DE" if vname.startswith("DE") else "NL"),
                                   "price_eur": price,
                                   "eur_per_kwh_usable": round(price / b.usable_kwh) if b.usable_kwh else None,
                                   "payback_years": pb["payback_years"],
                                   "payback_min": min(per_year) if per_year else pb["payback_years"],
                                   "payback_max": max(per_year) if per_year else pb["payback_years"],
                                   "npv_eur": pb["npv"], "lifetime_net_saving_eur": pb["lifetime_net"],
                                   "efc_per_year": efc, "end_of_life_year": pb["eol_year"],
                                   "life_limit": pb["limit"], "estimated": "yes" if b.estimated else "",
                                   "_bid": bid, "_cid": cid, "_path": pb["path"],
                                   "_saving27": by_scn.get("nosal_min50", np.nan)})
        rk = pd.DataFrame(ranked)
        if len(rk):
            rk = rk.sort_values(["analysis", "payback_years"], ascending=[False, True]).reset_index(drop=True)
        return {"savings": pd.DataFrame(sav_rows), "ranked": rk}

    def avg_u_cache(self):
        if not hasattr(self, "_avg_u"):
            self._avg_u = self.avg_prices()[0]["nosal_min50"][0]
        return self._avg_u

    def scale_variants(self, meta, contracts):
        dyn = [c for c in contracts if c.is_dynamic]
        if not dyn or not self.head_years:
            return None
        c = dyn[0]
        scales = self.cfg.get("strategies", {}).get("dynamic", {}).get("breakeven_scales", [1.0])
        rows = []
        scn = "nosal_min50"
        for b in self.batteries:
            cap = self.conn.battery_cap_w(b.phases)
            wear = meta[(b.id, c.id)]["wear"]
            for sc in scales:
                vals = []
                for label, per, idx, _ in self.periods(c, scn, self.head_years):
                    r = self.sim_strategy("dynamic", b, cap, per, idx, c, scn, wear, scale=sc)
                    vals.append(self.saving(r, per, c, scn))
                rows.append({"battery": b.name, "contract": c.label, "scale": sc, "saving_eur_year": np.mean(vals)})
        df = pd.DataFrame(rows)
        df["best_for_battery"] = df.groupby("battery")["saving_eur_year"].transform(lambda x: x == x.max())
        return df

    def sensitivity(self, summary, meta, contracts):
        rk = summary["ranked"]
        if rk is None or rk.empty:
            # Without prices, still show sensitivity of savings for the best combos.
            sv = summary["savings"]
            if sv is None or sv.empty:
                return None
            col = [c for c in sv.columns if c.startswith("saving_2027")]
            if not col:
                return None
            top = sv[sv.analysis == "headline"].sort_values(col[0], ascending=False).drop_duplicates("battery").head(3)
            combos = [(r.battery, r.contract, r.strategy) for r in top.itertuples()]
        elif getattr(self, "focus", None):
            combos = [(f["b"].name, f["c"].label, f["strategy"]) for f in self.focus]
        else:
            top = rk[rk.analysis == "headline"].drop_duplicates(["_bid"]).head(3)
            combos = [(self.bname(r["_bid"]), self.cname(r["_cid"]), r["strategy"]) for r in top.to_dict("records")]
        rows = []
        scn = "nosal_min50"
        for bname, cname, strat in combos:
            b = next(x for x in self.batteries if x.name == bname)
            c = next(x for x in contracts if x.label == cname)
            cap = self.conn.battery_cap_w(b.phases)
            m = meta[(b.id, c.id)]
            years = self.head_years if c.is_dynamic else None
            for vname, kw in (("base", {}), ("RTE −5 pt", {"rte": b.rte - 0.05}), ("RTE +5 pt", {"rte": min(b.rte + 0.05, 1.0)}),
                              ("standby ×2", {"standby": (b.standby_w or 0) * 2})):
                vals = []
                for label, per, idx, _ in self.periods(c, scn, years):
                    r = self.sim_strategy(strat, b, cap, per, idx, c, scn, m["wear"], windows=m["windows"], **kw)
                    vals.append(self.saving(r, per, c, scn))
                rows.append({"battery": bname, "contract": cname, "strategy": strat, "variant": vname,
                             "saving_eur_year_2027_rules": np.mean(vals)})
        return pd.DataFrame(rows) if rows else None


def payback(price: float, savings_by_scn: dict, efc_per_year: float, delivered_kwh_year: float,
            b: Battery, cfg: dict) -> dict:
    """Year-by-year savings path from the purchase date (section 8)."""
    a = cfg["analysis"]
    r = float(a.get("discount_rate", 0.03))
    horizon = int(a.get("horizon_years", 20))
    start = pd.Timestamp(a.get("purchase_date", "2026-10-02"))
    cal_life = b.warranty_years or 10.0
    life = b.cycle_life or 1e12
    thr = (b.warranty_mwh or 0) * 1000.0
    efc_per_year = 0.0 if (efc_per_year is None or np.isnan(efc_per_year)) else efc_per_year
    delivered_kwh_year = 0.0 if (delivered_kwh_year is None or np.isnan(delivered_kwh_year)) else delivered_kwh_year
    limits = {"calendar life": cal_life}
    if efc_per_year > 0:
        limits["cycles"] = life / efc_per_year
    if thr and delivered_kwh_year > 0:
        limits["warranty throughput"] = thr / delivered_kwh_year
    limit = min(limits, key=limits.get)
    life_years = min(limits[limit], horizon)
    cum, npv, t = 0.0, 0.0, 0.0
    pay = math.inf
    year = start.year
    path = [(0.0, -price)]           # (years since purchase, cumulative saving − price)
    first_frac = (pd.Timestamp(year=year + 1, month=1, day=1) - start).days / 365.0
    while t < life_years - 1e-9:
        frac = first_frac if t == 0 else 1.0
        frac = min(frac, life_years - t)
        scn = regime_for_year(cfg, year)
        s = savings_by_scn.get(scn)
        if s is None or (isinstance(s, float) and np.isnan(s)):
            s = savings_by_scn.get("nosal_min50", 0.0)
        mid_efc = efc_per_year * (t + frac / 2)
        capf = 1.0 - (1.0 - b.eol_capacity) * min(mid_efc / life, 1.0)
        gain = s * capf * frac
        if pay == math.inf and cum + gain >= price and gain > 0:
            pay = t + frac * (price - cum) / gain
        cum += gain
        npv += gain / (1 + r) ** (t + frac / 2)
        t += frac
        year += 1
        path.append((t, cum - price))
    return {"payback_years": pay, "npv": npv - price, "lifetime_net": cum - price, "path": path,
            "eol_year": f"{start.year + (start.dayofyear / 365.0) + life_years:.1f}", "limit": limit}
