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
from .battery import (GRID_CHARGE, SELF, ZERO_IMPORT, Battery, in_scope, load_batteries, perfect_foresight,
                      plan_dynamic, plan_forecast, plan_hbc, plan_timed, simulate)
from .breakeven import breakeven_rows, wear_cost
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
        self.step_sanity()
        self.step_payback()
        self.step_kiln()
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

        md = [f"**Data:** {self.inp.label}. **{self.conn.describe()}.**", ""]
        md.append(f"P1 file `{rep.source}`: {rep.rows} rows, {rep.first:%Y-%m-%d %H:%M} to {rep.last:%Y-%m-%d %H:%M} "
                  f"(Europe/Amsterdam, DST handled).")
        md.append(f"- Duplicates removed: {rep.duplicates}; counter resets: {len(rep.resets)}; "
                  f"gaps ≤1 h interpolated: {len(rep.interpolated_gaps)}; longer gaps flagged and filled: "
                  f"{len(rep.long_gaps)}.")
        for g in rep.long_gaps[:10]:
            md.append(f"  - long gap {g[0]:%Y-%m-%d %H:%M} → {g[1]:%Y-%m-%d %H:%M}")
        if rep.tariff_rule_agreement is not None:
            md.append(f"- T1/T2 from counters matches the configured tariff-hour rule in "
                      f"{rep.tariff_rule_agreement:.1%} of intervals (rule used only for synthetic intervals).")
        md.append(f"- Profile year: {rep.window[0]:%Y-%m-%d} to {rep.window[1]:%Y-%m-%d}.")
        if rep.missing_months:
            md.append(f"- **Missing months filled synthetically:** {', '.join(rep.missing_months)} "
                      "(blend of the two weeks before and after, flagged).")
        for s in rep.partially_real_months:
            md.append(f"  - {s}")
        if rep.low_confidence:
            md.append("- **Low confidence:** more than 3 months are synthetic.")
        real_i = profile.loc[self.real_mask, "imp"].sum()
        real_x = profile.loc[self.real_mask, "exp"].sum()
        md.append(f"- Yearly import {tot_i:,.0f} kWh, export {tot_x:,.0f} kWh "
                  f"(real months only: import {real_i:,.0f}, export {real_x:,.0f}).")
        if self.pv is not None:
            gross = (profile["imp"] - profile["exp"] + self.pv.fillna(0)).sum()
            md.append(f"- PV data present: production {self.pv.sum():,.0f} kWh, gross consumption {gross:,.0f} kWh.")
        else:
            md.append("- No PV file: the analysis uses net flows only (enough for contracts and batteries; "
                      "gross consumption and the solar-forecast Charge goal are not available).")
        for n in rep.notes:
            md.append(f"- {n}")
        md.append("")
        md.append("**Price history**")
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
        sec = Section("data", "1. Data quality", "\n".join(md))
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
        sec = Section("current", "2. Current contract")
        if self.current is None:
            md.append("No current contract given: it is left out of the comparison.")
            sec.md = "\n".join(md)
            self.emit(sec)
            return
        c = self.current
        md.append(f"**{c.label}** ({c.type}, {c.start_date or '?'} → {c.end_date or '?'}). Source: `{c.source}`.")
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
        md.append("")
        md.append("Yearly cost on the profile year (incl. VAT, grid charges and tax reduction). "
                  "`total_real_months_only` is the check without the synthetic month(s).")
        md.append("")
        md.append("**Acceptance check:** compare `total` under 2026 rules with an actual annual bill "
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
        sec = Section("contracts", "3. Objective 2 – best contract without a battery")
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
            # chart: top 3 dynamic vs best fixed, 2027 rules
            scn = "nosal_min50"
            dyn = [c for c in self.contracts if c.is_dynamic]
            dyn.sort(key=lambda c: expected_by_contract.get(c.id, {}).get(scn, np.inf))
            series = {}
            for c in dyn[:3]:
                costs = self.contract_year_costs(c, scn)
                series[c.label] = pd.Series({y: v[0].total for y, v in sorted(costs.items())})
            fixed = [c for c in self.contracts if not c.is_dynamic and c.id in expected_by_contract]
            best_fixed = min(fixed, key=lambda c: expected_by_contract[c.id].get(scn, np.inf)) if fixed else None
            f = self.fig(charts.contract_years, series, best_fixed.label if best_fixed else None,
                         expected_by_contract[best_fixed.id].get(scn) if best_fixed else None,
                         "Yearly cost per price year (2027–2029 rules)")
            if f:
                sec.figures["contract_years"] = f
        else:
            md.append("Dynamic contracts are skipped: no day-ahead price history was loaded.")
        unverified = [c.label for c in self.contracts if not c.verified]
        if unverified:
            md.append(f"Unverified contract terms (hand-entered or template): {', '.join(unverified)}.")
        md.append("Fixed contracts use their stated tariffs (one value per scenario); dynamic contracts are replayed "
                  "against every price year under the scenario's rules. `real_months_only` excludes synthetic months. "
                  "`avg_eur_year_over_term` spreads the cost over the contract term from the purchase date, "
                  "switching rules on 1 January 2027, and subtracts the one-off switch bonus once. Yearly figures exclude the bonus.")
        self.expected_by_contract = expected_by_contract
        fv = self.feed_in_value_table()
        if fv is not None:
            sec.tables["feed_in_value_per_year"] = fv
            last = fv.iloc[-1]
            md.append(f"**Why feed-in earns little on a dynamic contract:** you export mostly around midday, when "
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
        sec = Section("batteries", "4. Battery dataset")
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
            rows.append({"id": b.id, "battery": b.name, "type": b.type, "phases": b.phases,
                         "usable_kwh": round(b.usable_kwh, 2), "max_charge_w": b.max_charge_w,
                         "max_discharge_w": b.max_discharge_w,
                         "power_cap_w": None if cap is None else round(min(cap, b.max_charge_w, b.max_discharge_w)),
                         "rte": b.rte, "standby_w": b.standby_w, "cycle_life": b.cycle_life,
                         "eol_capacity": b.eol_capacity, "warranty_years": b.warranty_years,
                         "warranty_mwh": b.warranty_mwh, "price_nl": b.price_nl, "price_nl_lowest": b.price_nl_lowest,
                         "price_de": b.price_de, "extra_hw": b.extra_hardware,
                         "eur_per_kwh_nominal": round(b.price_nl / b.nominal_kwh) if b.price_nl and b.nominal_kwh else None,
                         "eur_per_kwh_usable": round(b.price_nl / b.usable_kwh) if b.price_nl and b.usable_kwh else None,
                         "solar_during_outage": b.outage_solar or "unknown", "dc_solar_input_w": b.dc_solar_w,
                         "missing_fields": ", ".join(b.missing_fields), "estimated_with_defaults": ", ".join(b.estimated_fields),
                         "verified": "yes" if b.verified else "no", "in_scope": "yes" if ok else "no", "note": note})
        sec.tables["battery_specs"] = pd.DataFrame(rows)
        d = self.cfg.get("battery_defaults", {})
        md = [f"{len(bats)} batteries loaded, {len(self.batteries)} in scope for {self.conn.label}.",
              f"Missing specs are left empty and flagged; the simulation uses documented defaults "
              f"(RTE {d.get('rte')}, standby {d.get('standby_w')} W, cycle life {d.get('cycle_life')}, "
              f"EoL {d.get('eol_capacity')}, warranty {d.get('warranty_years')} yr) and marks the battery as estimated."]
        for b, why in self.excluded:
            md.append(f"- Excluded: {b.name} – {why}.")
        md.append("`solar_during_outage`: can solar power keep charging the battery when the grid (or the main "
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
        variants = b.price_variants(d.get("de_travel_cost_eur", 0.0))
        if not variants:
            return 0.0, "no price"
        name = "NL current" if "NL current" in variants else next(iter(variants))
        w, _, _ = wear_cost(b, variants[name], cycles or d.get("cycles_per_year", 250), d.get("residual_value", 0.0))
        return w, name

    def step_breakeven(self, cycles: dict | None = None, title_suffix: str = ""):
        avg, c = self.avg_prices()
        d = self.cfg.get("battery_defaults", {})
        rows = breakeven_rows(self.batteries, d, avg["saldering"][0], avg["saldering"][1], avg["nosal_min50"][1],
                              cycles=cycles, de_travel=d.get("de_travel_cost_eur", 0.0),
                              bf=self.cfg.get("blackfriday"))
        sid = "breakeven" if cycles is None else "breakeven_recomputed"
        sec = Section(sid, ("5. Objective 1 – break-even price per battery" if cycles is None
                            else "8. Objective 1 recomputed with simulated cycles") + title_suffix)
        df = pd.DataFrame(rows)
        if len(df):
            sec.tables["breakeven"] = df.round(4)
        md = [f"Reference prices from {c.label if c else 'n/a'}: average all-in import price "
              f"€{avg['saldering'][0]:.3f}/kWh; own-solar charge price (feed-in given up) "
              f"€{avg['saldering'][1]:.3f} under saldering and €{avg['nosal_min50'][1]:.3f} from 2027.",
              "Break-even = charge price / RTE + wear cost. Under saldering selling and using at home are worth the "
              "same, so one break-even applies; from 2027 the sell break-even is compared with the net feed-in price. "
              "Standby is a fixed cost, reported in Objective 3.",
              "`lifetime_limit` shows which limit (cycles, warranty throughput or calendar life) sets lifetime kWh."]
        md += self.breakeven_explainer(rows)
        if cycles is None:
            md.append(f"Cycles per year: default {d.get('cycles_per_year', 250)} (replaced by simulated values in section 8).")
        sec.md = "\n".join(md)
        priced = [r for r in rows if r["purchase_eur"] is not None and not math.isnan(r["wear_eur_kwh"])]
        if priced:
            top = sorted(priced, key=lambda r: r["wear_eur_kwh"])[:5]
            f = self.fig(charts.breakeven_lines, top)
            if f:
                sec.figures["breakeven"] = f
        out_dir = self.path("results_dir")
        if out_dir and len(df) and not getattr(self, "no_write", False):
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, "breakeven.csv" if cycles is None else "breakeven_recomputed.csv"),
                      index=False)
        self.breakeven_df = df
        if len(df):
            nl = df[df.price_variant == "NL current"]
            wear = dict(zip(nl.battery_id, nl.wear_eur_kwh))
            cheap = self.typical_cheap_price()
            md_rows = extras.min_difference_rows(self.batteries, wear, cheap, self.taxes.vat)
            if md_rows:
                sec.tables["minimum_price_difference"] = pd.DataFrame(md_rows)
                sec.md += ("\n\n**Minimum price difference worth charging** = charge price × (1/RTE − 1) + wear "
                           "(€/kWh, all-in). The `at_charge_x` columns give it for charging at price x. "
                           f"`setting_*` is the value to enter in a battery app or Home Assistant (`min_delta`) for a "
                           f"typical cheap hour (€{cheap:.3f}/kWh all-in, the average of each day's 4 cheapest hours on "
                           "the cheapest dynamic contract, 2027 rules, last 3 years): `setting_all_in` if the app "
                           "compares prices incl. taxes, `setting_spot` if it compares spot (EPEX) prices "
                           "(all-in difference ÷ 1.21, because taxes per kWh are equal every hour).")
        self.emit(sec)

    def breakeven_explainer(self, rows) -> list[str]:
        """Plain-language formula with a worked example (first priced battery)."""
        ex = next((r for r in rows if r["price_variant"] == "NL current" and r["purchase_eur"]
                   and not math.isnan(r["wear_eur_kwh"])), None)
        out = ["",
               "**How the break-even price is calculated**",
               "",
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
        sec = Section("power", "1b. Power profile")
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
        md = ["Power from the 15-minute meter data (real months only). A battery can only store surplus or cover "
              "load up to its own power, so `battery_power_vs_energy` shows how much of the yearly surplus a battery "
              "of that power can store, and how much of the import it can cover.",
              f"- An 800 W socket battery can store {c8.loc[0.8, 'share_of_surplus_it_can_store']:.0%} of the surplus and "
              f"cover {c8.loc[0.8, 'share_of_import_it_can_cover']:.0%} of the import; 2.4 kW: "
              f"{c8.loc[2.4, 'share_of_surplus_it_can_store']:.0%} / {c8.loc[2.4, 'share_of_import_it_can_cover']:.0%}.",
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
        sec = Section("sanity", "6. Self-consumption sanity check")
        if not self.batteries:
            sec.md = "No battery in scope."
            self.emit(sec)
            return
        b = min(self.batteries, key=lambda x: (len(x.missing_fields), x.usable_kwh))
        cap = self.conn.battery_cap_w(b.phases)
        net = self.profile["imp"].values - self.profile["exp"].values
        r = simulate(net, SELF, b, cap, keep_trace=True)
        balance = np.abs((r.imp - r.exp) - (net + (b.standby_w or 0) / 4000.0 + r.batt_ac)).max()
        md = [f"Battery: **{b.name}** ({b.usable_kwh:.2f} kWh usable, cap {min(cap, b.max_charge_w):.0f} W), "
              f"self-consumption on the profile year.",
              f"- Equivalent full cycles: {r.efc:.0f}/yr; charged {r.charged_ac:,.0f} kWh AC, "
              f"delivered {r.discharged_ac:,.0f} kWh AC; capacity at year end {r.end_capacity:.2f} kWh.",
              f"- Import {self.profile['imp'].sum():,.0f} → {r.imp.sum():,.0f} kWh; "
              f"export {self.profile['exp'].sum():,.0f} → {r.exp.sum():,.0f} kWh.",
              f"- Energy balance (import − export = net load + standby + battery AC flow): max error {balance:.2e} kWh."]
        sec.md = "\n".join(md)
        idx = self.profile.index
        sb = (b.standby_w or 0) / 4000.0
        daily = extras.daily_battery(idx, net, r.batt_ac, r.soc, b.usable_kwh, sb)
        md.append(f"- Whole year: the battery is completely full on {int(daily.full.sum())} days and "
                  f"stays below half full on {int((daily.max_soc_kwh < 0.5 * b.usable_kwh).sum())} days (mostly winter).")
        f = self.fig(charts.soc_year, daily, b.usable_kwh, f"{b.name}: whole year, self-consumption")
        if f:
            sec.figures["soc_year_daily"] = f
        sec.interactive["soc_year_15min"] = {
            "title": f"{b.name} – state of charge and house power, every 15 minutes",
            "y": "kWh / kW", "height": 340,
            "x": extras.ts(idx),
            "series": [extras.series("State of charge (kWh)", r.soc, "#2a6f97", "area"),
                       extras.series("House net power without battery (kW, + import / − export)", net * 4, "#e07a5f",
                                     width=0.8),
                       extras.series("Battery power (kW, + charging / − discharging)", r.batt_ac * 4, "#3d405b",
                                     width=0.8)]}
        sec.md = "\n".join(md)
        self.sanity = {"battery": b.id, "efc": r.efc, "balance_error": balance}
        self.emit(sec)

    # ------------------------------------------------------------------ step 7
    def strategies_for(self, c: Contract) -> list[str]:
        enabled = self.cfg.get("strategies", {}).get("enabled", [])
        out = [s for s in enabled if s in ("self_consumption", "timed")]
        if c.is_dynamic:
            presets = self.cfg.get("strategies", {}).get("hbc_presets", {}) or {}
            out += [s for s in enabled
                    if s in ("dynamic", "dynamic_sell", "forecast", "perfect_foresight") or s in presets]
        return out

    def sim_strategy(self, strategy, b, cap, per: Period, idx, c, scn, wear, windows=None, scale=1.0,
                     rte=None, standby=None, keep_trace=False):
        net = per.imp - per.exp
        bb = b
        if rte is not None or standby is not None:
            from dataclasses import replace
            bb = replace(b, rte=rte if rte is not None else b.rte,
                         standby_w=standby if standby is not None else b.standby_w)
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
        sec = Section("payback", "7. Objective 3 – battery payback per strategy")
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
                        lp = strat in ("perfect_foresight", "forecast")     # LP-based: headline years only
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
        md = [f"Savings = cost without battery − cost with battery under the same contract, rules and price year "
              f"(standby and losses included). Payback follows the purchase date "
              f"{self.cfg['analysis']['purchase_date']} year by year: 2026 rules until the end of 2026, then "
              f"2027–2029 rules, then 2030+ rules, with capacity fading towards end of life.",
              "Dynamic and Sell use each battery's own break-even (NL current price) from Objective 1; "
              "decisions use only that day's day-ahead prices, and stored energy is reserved for the day's "
              "priciest load. **Home Battery Control cannot do this reservation today**: `dynamic` needs custom "
              "control. `hbc_*` strategies reproduce Home Battery Control's Dynamic strategy exactly "
              "(Extreme-Pair Matching, presets in config.yaml), so those results are achievable as-is. "
              "`forecast` re-plans every day at 13:00 over the prices known then (until tomorrow night) with a "
              "usage/solar forecast from the last 3 days, the way EMHASS does in Home Assistant (custom control, "
              "headline years only). `perfect_foresight` is the upper bound (headline years only).",
              "For fixed contracts there is one value (stated tariffs); for dynamic contracts the min–max is the "
              "payback across price years. Partial price years are excluded from the ranges."]
        if (self.prices is None):
            md.append("**No price history loaded:** only fixed-contract combinations were simulated.")
        estimated = [b.name for b in self.batteries if b.estimated]
        if estimated:
            md.append(f"Estimated (defaults used for missing specs): {', '.join(estimated)}.")
        windows = [f"{self.bname(bid)} × {self.cname(cid)}: " + (", ".join(
            f"{'charge' if m == GRID_CHARGE else 'discharge'} {a:02d}–{z:02d}" for a, z, m in v['windows'])
            or "no window beats self-consumption") for (bid, cid), v in meta.items() if v["windows"] is not None]
        if windows:
            md.append("Timed windows found on history (2027 rules): " + "; ".join(windows[:12]))
        sec.tables["savings_by_combination"] = summary["savings"].round(2)
        if len(summary["ranked"]):
            sec.tables["payback_ranked"] = summary["ranked"].round(2)
            f = self.fig(charts.payback_scatter, summary["ranked"].replace([np.inf], np.nan).dropna(subset=["payback_years"]))
            if f:
                sec.figures["payback_vs_capacity"] = f
        else:
            md.append("**No payback ranking:** no battery has a purchase price yet. Savings per year are shown above.")
        # scale variants
        sv = self.scale_variants(meta, contracts)
        if sv is not None:
            sec.tables["dynamic_breakeven_scale"] = sv.round(2)
        sens = self.sensitivity(summary, meta, contracts)
        if sens is not None:
            sec.tables["sensitivity"] = sens.round(2)
        ys = self.savings_year_chart(rec)
        if ys:
            sec.figures["savings_per_price_year"] = ys
        bft = self.blackfriday_table(summary)
        if bft is not None:
            sec.tables["black_friday_quick_decision"] = bft
            bfc = self.cfg.get("blackfriday", {}) or {}
            w = bfc.get("window", ["11-20", "12-01"])
            found = bft["nl_deal_eur"].notna().any() or bft["de_deal_eur"].notna().any()
            est = bool(bfc.get("estimates", True))
            md.append(f"**Black Friday quick decision** (window {w[0]} – {w[1]}, Black Friday {bfc.get('date', '')}): "
                      "per battery its best contract and strategy, today's price, the real deal price the scraper "
                      "found during the window (NL incl. VAT, DE 0% VAT from German manufacturer shops) with the "
                      "discount against the last normal price"
                      + (f", and the estimate (−{float(bfc.get('discount_nl', 0.15)):.0%} NL / "
                         f"−{float(bfc.get('discount_de', 0.15)):.0%} DE)" if est else " (estimates switched off)")
                      + ". Sorted by the best available payback. "
                      + ("" if found else "No real deals recorded yet: the scraper fills them in during the window "
                         "(every 4 hours)" + (", until then the estimate columns apply." if est else ".")))
        self.chosen = self.pick_battery(summary)
        if self.chosen is not None:
            split = self.earnings_by_strategy(self.chosen, contracts, meta)
            if split is not None:
                tbl, fig = split
                sec.tables[f"earnings_per_year_{self.chosen.id}"] = tbl.round(0)
                if fig:
                    sec.figures["earnings_per_strategy"] = fig
                md.append(f"**Earnings per strategy** for {self.chosen.name} (chosen battery): each year of "
                          "ownership split into avoided import, energy sold, feed-in given up for stored solar, "
                          "grid charging cost, and standby/other (saldering netting, feed-in tiers). Bars stack to "
                          "the net saving (black dot); later years shrink as capacity fades.")
            gap = self.strategy_gap(self.chosen, self.cheapest_dynamic(), meta)
            if gap is not None and len(gap):
                sec.tables[f"gap_to_perfect_foresight_{self.chosen.id}"] = gap.round(1)
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
        now = h[h.price_variant == "NL current"]
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
            row = {"battery": r["battery"], "usable_kwh": r["usable_kwh"], "contract": r["contract"],
                   "strategy": r["strategy"],
                   "nl_now_eur": pr.get("NL current"), "payback_now": pb.get("NL current"),
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
        df = pd.DataFrame(rows).sort_values("best_payback").reset_index(drop=True)
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
            r = rk[(rk.analysis == "headline") & (rk.price_variant == "NL current")
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

    def earnings_by_strategy(self, b: Battery, contracts, meta):
        c = self.cheapest_dynamic() or (contracts[0] if contracts else None)
        if c is None:
            return None
        if c not in contracts and c.id not in [x.id for x in contracts]:
            contracts = contracts + [c]
        cap = self.conn.battery_cap_w(b.phases)
        m = meta.get((b.id, c.id), {"windows": [], "wear": self.wear_for(b)[0]})
        per_scn = {}
        efc_by = {}
        for strat in self.strategies_for(c):
            if strat == "timed" and not m["windows"]:
                continue
            for scn in self.scenarios:
                parts, efcs = [], []
                for label, per, idx, _ in self.periods(c, scn, self.head_years if c.is_dynamic else None):
                    r = self.sim_strategy(strat, b, cap, per, idx, c, scn, m["wear"], windows=m["windows"])
                    sv = self.saving(r, per, c, scn)
                    u, s_ = marginal_values(per, c, self.regimes[scn], self.taxes, self.net_importer)
                    sb = (b.standby_w or 0.0) / 1000.0 * 0.25
                    parts.append(extras.earnings_split(per.imp - per.exp, r.batt_ac, sb, u, s_, sv))
                    efcs.append(r.efc)
                if parts:
                    per_scn[(strat, scn)] = pd.DataFrame(parts).mean().to_dict()
                    efc_by[(strat, scn)] = float(np.mean(efcs))
        if not per_scn:
            return None
        start = pd.Timestamp(self.cfg["analysis"]["purchase_date"])
        rows = []
        for strat in sorted({k[0] for k in per_scn}):
            efc = efc_by.get((strat, "nosal_min50"), 250.0)
            life = self.life_years(b, efc)
            t, year = 0.0, start.year
            first = (pd.Timestamp(year=year + 1, month=1, day=1) - start).days / 365.0
            while t < life - 1e-9:
                frac = min(first if t == 0 else 1.0, life - t)
                scn = regime_for_year(self.cfg, year)
                comp = per_scn.get((strat, scn)) or per_scn.get((strat, "nosal_min50"))
                capf = 1.0 - (1.0 - b.eol_capacity) * min(efc * (t + frac / 2) / (b.cycle_life or 1e12), 1.0)
                row = {"strategy": strat, "year": year, "rules": self.regimes[scn]["label"]}
                for k, v in comp.items():
                    row[k] = v * frac * capf
                row["net_saving"] = sum(comp.values()) * frac * capf
                rows.append(row)
                t += frac
                year += 1
        tbl = pd.DataFrame(rows)
        fig = self.fig(charts.earnings_stacked, tbl, f"{b.name} – {c.label}")
        return tbl, fig

    def strategy_gap(self, b: Battery, c: Contract, meta) -> pd.DataFrame | None:
        """Where the gap between the achievable strategies and perfect foresight
        comes from (2027–2029 rules, headline years): the same optimiser with
        less knowledge, and perfect foresight with fewer freedoms."""
        if c is None or not c.is_dynamic:
            return None
        scn = "nosal_min50"
        years = self.head_years[-1:] if self.opts.quick else self.head_years
        pers = self.periods(c, scn, years)
        if not pers:
            return None
        cap = self.conn.battery_cap_w(b.phases)
        m = meta.get((b.id, c.id), {"windows": [], "wear": self.wear_for(b)[0]})
        fit = self.head_years[-1] if self.head_years else ""
        fcfg = self.cfg.get("strategies", {}).get("forecast", {})
        rows = [
            ("self_consumption", "nothing: store surplus, use it when the house imports", "self_consumption", {}),
            ("timed", f"fixed windows, chosen on {fit} prices (that year is in-sample)", "timed", {}),
            ("dynamic", "today's day-ahead prices + yesterday's usage", "dynamic", {}),
            ("dynamic_sell", "same, may sell to the grid", "dynamic_sell", {}),
            ("forecast", "day-ahead prices until tomorrow night + usage/solar of the last days", "forecast", {}),
            ("forecast, perfect usage forecast", "same optimiser, but knows the real usage and solar in advance",
             "oracle", {}),
            ("bound: solar only, no selling", "everything in advance; stores only solar, covers only own load",
             "pf", {"grid_charge": False, "sell": False}),
            ("bound: solar only, may sell", "everything in advance; stores only solar, may sell it",
             "pf", {"grid_charge": False}),
            ("bound: grid charging, no selling", "everything in advance; may charge from the grid, no selling",
             "pf", {"sell": False}),
            ("perfect_foresight", "everything in advance, all freedoms", "pf", {}),
        ]
        out = []
        for name, knows, kind, kw in rows:
            if kind == "timed" and not m["windows"]:
                continue
            row = {"strategy": name, "knows": knows}
            vals, cyc = [], []
            for label, per, idx, _ in pers:
                net = per.imp - per.exp
                if kind == "pf":
                    u, s = marginal_values(per, c, self.regimes[scn], self.taxes, self.net_importer)
                    r = perfect_foresight(net, u, s, b, cap, 0.0, **kw)
                elif kind == "oracle":
                    u, s = marginal_values(per, c, self.regimes[scn], self.taxes, self.net_importer)
                    r = simulate(net, plan_forecast(idx, net, u, s, b, cap, m["wear"], fcfg, forecast=net), b, cap)
                else:
                    r = self.sim_strategy(kind, b, cap, per, idx, c, scn, m["wear"], windows=m["windows"])
                v = self.saving(r, per, c, scn)
                row[f"eur_{label}"] = v
                vals.append(v)
                cyc.append(r.efc)
            row["mean_eur"] = float(np.mean(vals))
            row["cycles_per_year"] = float(np.mean(cyc))
            out.append(row)
        df = pd.DataFrame(out)
        top = df.loc[df.strategy == "perfect_foresight", "mean_eur"]
        if len(top):
            df["gap_to_bound_eur"] = top.iloc[0] - df["mean_eur"]
        return df

    @staticmethod
    def gap_text(gap: pd.DataFrame) -> str:
        v = dict(zip(gap.strategy, gap.mean_eur))
        g = lambda k: v.get(k, float("nan"))  # noqa: E731
        ycols = [c for c in gap.columns if c.startswith("eur_")]
        lines = ["**Why perfect foresight earns so much more** (chosen battery, cheapest dynamic contract, "
                 "2027–2029 rules). Each row adds or removes one piece of knowledge or freedom:",
                 f"- *Timing alone* is the biggest piece: storing only solar and only covering the house, perfect "
                 f"knowledge still earns €{g('bound: solar only, no selling'):,.0f} vs €{g('self_consumption'):,.0f} "
                 "for self-consumption, with about the same number of cycles. It absorbs solar in the cheapest "
                 "(often negative) export hours instead of first thing in the morning, and keeps the energy for "
                 "the most expensive hours.",
                 f"- *Selling stored solar* adds only €{g('bound: solar only, may sell') - g('bound: solar only, no selling'):,.0f} "
                 "per year: the battery is almost empty by the next morning, so there is rarely energy left to sell "
                 "on top of the evening and night use.",
                 f"- *Grid charging* adds €{g('bound: grid charging, no selling') - g('bound: solar only, no selling'):,.0f}; "
                 "perfect foresight ignores wear, so part of that would not pay off for real.",
                 f"- With the real day-ahead prices but a **perfect usage and solar forecast**, the optimiser reaches "
                 f"€{g('forecast, perfect usage forecast'):,.0f}; with yesterday-style forecasts (`forecast`) "
                 f"€{g('forecast'):,.0f}. So the gap is mostly about predicting the household's own solar and "
                 "usage, not about prices. A good solar forecast (e.g. Forecast.Solar or Solcast in Home "
                 "Assistant) is what closes it."]
        if "timed" in v and len(ycols) > 1:
            row = gap[gap.strategy == "timed"].iloc[0]
            dyn = gap[gap.strategy == "dynamic"].iloc[0]
            wins = [c[4:] for c in ycols if row[c] > dyn[c]]
            lines.append(f"- `timed` beats `dynamic` in {', '.join(wins) if wins else 'no year'}: its windows are "
                         f"picked on {ycols[-1][4:]} prices, so that year is in-sample. Its grid charging also ignores "
                         "the battery's wear, which `dynamic` requires every charge to earn back.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ 11.6 Objective 4
    def step_kiln(self):
        sec = Section("kiln", "9. Objective 4 – pottery kiln on free power")
        k = dict(self.cfg.get("kiln", {}) or {})
        k.update(self.opts.kiln or {})
        b = getattr(self, "chosen", None)
        powers = [float(p) for p in k.get("powers_kw", [1.5, 2.0, 3.0, 3.6])]
        one_phase = self.conn.phases == 1
        if one_phase:
            powers = [p for p in powers if p <= float(k.get("single_phase_max_kw", 3.68))]
        hours, duty = float(k.get("firing_hours", 8)), float(k.get("avg_duty", 0.7))
        starts, tol = [int(h) for h in k.get("start_hours", [7, 8, 9, 10])], float(k.get("free_tolerance", 0.05))
        nb = extras.kiln_days(self.profile, None, 0, None, powers, hours, duty, starts, tol)
        wb, soc_start, label = None, None, "with_battery"
        if b is not None:
            cap = self.conn.battery_cap_w(b.phases)
            net = (self.profile["imp"] - self.profile["exp"]).values
            r = simulate(net, SELF, b, cap, keep_trace=True)
            soc_start = np.concatenate([[0.0], r.soc[:-1]])
            wb = extras.kiln_days(self.profile, b, cap, soc_start, powers, hours, duty, starts, tol)
        price = self.avg_prices()[0]["nosal_min50"][0]
        summ = extras.kiln_summary(nb, wb, price, label)
        sec.tables["kiln_free_firing_days"] = summ.round(2)
        if wb is not None and len(wb):
            by_month = wb[wb.free].groupby(["kiln_kw", "month"]).size().unstack(fill_value=0)
            sec.tables["free_days_per_month_with_battery"] = by_month
        md = [f"How many days per year a kiln can fire to maximum temperature on solar surplus"
              f"{' plus the ' + b.name if b else ''}, with at most {tol:.0%} of the firing energy from the grid. "
              f"Assumptions (config.yaml `kiln`, edit to your kiln's data sheet): a firing takes {hours:g} h and the "
              f"kiln draws on average {duty:.0%} of its rated power, so a firing needs rated kW × {hours * duty:g} kWh. "
              f"Start hour is chosen per day among {starts}. The battery starts each day at the charge it would have "
              "had with normal self-consumption.",
              f"- `avg_cost_per_firing` = grid energy × €{price:.3f}/kWh (average all-in import price, 2027 rules). "
              "Surplus solar is not entirely free: from 2027 each kWh used instead of exported gives up about "
              f"€{self.avg_prices()[0]['nosal_min50'][1]:.3f} of feed-in compensation.",
              "- Kilns above 3.68 kW (16 A) need a three-phase (or dedicated 1-phase high-current) group; "
              + ("those are left out on this single-phase connection." if one_phase else
                 f"your {self.conn.label} connection allows them."),
              "- One firing per day at most; the profile year uses the real meter data (synthetic October included)."]
        sec.md = "\n".join(md)
        f = self.fig(charts.kiln_days, summ, b.name if b else None, label)
        if f:
            sec.figures["kiln_free_days"] = f
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
        h = rk[(rk.analysis == "headline") & (rk.price_variant == "NL current") & (rk.strategy != "perfect_foresight")]
        if adv is not None and (h._cid == adv.id).any():
            h = h[h._cid == adv.id]
        ranked = h.sort_values("payback_years").drop_duplicates("_bid")
        rec = self.records

        def describe(r):
            b = next(x for x in self.batteries if x.id == r["_bid"])
            cap = self.conn.battery_cap_w(b.phases)
            net = (self.profile["imp"] - self.profile["exp"]).values
            sim = simulate(net, SELF, b, cap, keep_trace=True)
            daily = extras.daily_battery(self.profile.index, net, sim.batt_ac, sim.soc, b.usable_kwh,
                                         (b.standby_w or 0) / 4000.0)
            sv = rec[(rec.battery_id == b.id) & (rec.contract_id == r["_cid"]) & (rec.strategy == r["strategy"])]
            sv27 = sv[(sv.scenario == "nosal_min50") & ~sv.partial
                      & sv.price_year.isin(["profile"] + list(self.head_years))]
            surplus = daily.solar_surplus_kwh.sum()
            return {"battery": b.name, "price_eur": r["price_eur"], "eur_per_kwh_usable": r["eur_per_kwh_usable"],
                    "usable_kwh": b.usable_kwh, "contract": r["contract"], "strategy": r["strategy"],
                    "saving_eur_year_2027": sv27.saving.mean() if len(sv27) else np.nan,
                    "payback_years": r["payback_years"], "npv_eur": r["npv_eur"],
                    "days_full": int(daily.full.sum()),
                    "days_below_half": int((daily.max_soc_kwh < 0.5 * b.usable_kwh).sum()),
                    "share_of_surplus_stored": daily.charged_from_solar_kwh.sum() / surplus if surplus else np.nan,
                    "_bid": b.id, "_cid": r["_cid"], "_model": re.sub(r"\s*\(.*\)$", "", b.name)}

        top = ranked.head(6).to_dict("records")
        comp = pd.DataFrame([describe(r) for r in top])
        chosen_fixed = self.opts.chosen_battery or self.cfg.get("chosen_battery")
        b = self.chosen if chosen_fixed else next(x for x in self.batteries if x.id == comp.iloc[0]["_bid"])
        self.chosen = b
        if not (comp._bid == b.id).any():
            r = ranked[ranked._bid == b.id]
            if len(r):
                comp = pd.concat([pd.DataFrame([describe(r.iloc[0].to_dict())]), comp], ignore_index=True)
        row = comp[comp._bid == b.id].iloc[0] if (comp._bid == b.id).any() else comp.iloc[0]
        # a clearly bigger battery, even if it is not in the top 6
        big = ranked[ranked.usable_kwh > row.usable_kwh * 1.6]
        bigger = describe(big.iloc[0].to_dict()) if len(big) else None
        if bigger is not None and not (comp._bid == bigger["_bid"]).any():
            comp = pd.concat([comp, pd.DataFrame([bigger])], ignore_index=True)
        sec.tables["battery_comparison"] = comp.round(2)
        classes = self.size_class_table(ranked, describe)
        if classes is not None:
            sec.tables["best_per_size_class"] = classes.round(2)
        sec.summary = (f"Best choice: {b.name}" + (f" on {adv.label}" if adv is not None else "")
                       + f" – pays back in {row.payback_years:.1f} years, saves about "
                       f"€{row.saving_eur_year_2027:,.0f} a year from 2027.")
        md = [f"### Best choice: **{b.name}**" + (f" on **{adv.label}**" if adv is not None else ""), ""]
        md.append(f"- **Pays back fastest:** {row.payback_years:.1f} years at €{row.price_eur:,.0f} "
                  f"(€{row.eur_per_kwh_usable:,.0f} per usable kWh), net present value €{row.npv_eur:,.0f} over its life.")
        md.append(f"- **Saves about €{row.saving_eur_year_2027:,.0f} per year** from 2027 with strategy "
                  f"`{row.strategy}` on {row.contract}.")
        if row.strategy in ("dynamic", "dynamic_sell", "forecast", "timed"):
            simple = rec[(rec.battery_id == b.id) & (rec.contract_id == row._cid) & (rec.scenario == "nosal_min50")
                         & ~rec.partial & rec.price_year.isin(["profile"] + list(self.head_years))
                         & rec.strategy.isin(["self_consumption", "hbc_pv_first"])]
            if len(simple):
                ss = simple.groupby("strategy").saving.mean().sort_values(ascending=False)
                md[-1] += (f" `{row.strategy}` needs custom control (see section 7); with plain `{ss.index[0]}`, "
                           f"which the battery's own app or Home Battery Control can do as-is, it saves "
                           f"€{ss.iloc[0]:,.0f} per year.")
        md.append(f"- **Its size fits your surplus:** it fills completely on {row.days_full} days a year and stores "
                  f"{row.share_of_surplus_stored:.0%} of your solar surplus. In winter it stays below half full on "
                  f"{row.days_below_half} days, so a bigger battery would mainly add capacity that sits idle.")
        if bigger is not None:
            extra = bigger["saving_eur_year_2027"] - row.saving_eur_year_2027
            md.append(f"- **Why not bigger?** {bigger['battery']} ({bigger['usable_kwh']:.1f} kWh usable) saves only "
                      f"€{extra:,.0f} more per year for €{bigger['price_eur'] - row.price_eur:,.0f} more, so it pays back "
                      f"in {bigger['payback_years']:.1f} years.")
        others = comp[(comp._model != row._model)]
        if len(others):
            o = others.iloc[0]
            md.append(f"- **Runner-up:** {o.battery}, {o.payback_years:.1f} years "
                      f"(€{o.eur_per_kwh_usable:,.0f} per usable kWh).")
        if classes is not None and len(classes) > 1:
            md.append(self.size_class_text(classes))
        combo = self.combination_table(b)
        if combo is not None:
            sec.tables["contract_and_battery_combinations"] = combo
            best_c = combo.iloc[combo.yearly_cost_eur_2027.idxmin()]
            md.append(f"- **Cheapest combination from 2027:** {best_c.option} – about €{best_c.yearly_cost_eur_2027:,.0f} "
                      f"a year, €{combo.iloc[0].yearly_cost_eur_2027 - best_c.yearly_cost_eur_2027:,.0f} less than "
                      "today's situation (see the combination table).")
        dyn = adv if (adv is not None and adv.is_dynamic) else None
        md.append("")
        md.append("The table compares the top batteries; the two charts below show, for the recommended battery, how "
                  "much solar surplus there is each day and how much of it the battery stores, and what it saves each day.")
        sec.md = "\n".join(md)
        self.advice_charts(sec, b, dyn)
        self.emit(sec)

    def size_class_table(self, ranked: pd.DataFrame, describe) -> pd.DataFrame | None:
        """Best-value battery (fastest payback) per capacity class, e.g. 5/10/15 kWh ± 2.5,
        with what the step up from the previous class adds."""
        a = self.cfg.get("analysis", {})
        centres = a.get("size_classes_kwh", [5, 10, 15])
        half = float(a.get("size_class_halfwidth_kwh", 2.5))
        size = {b.id: (b.nominal_kwh or b.usable_kwh) for b in self.batteries}
        rows, prev = [], None
        for cen in centres:
            sel = ranked[ranked._bid.map(size).between(cen - half, cen + half, inclusive="left")]
            if sel.empty:
                continue
            r = sel.iloc[0].to_dict()
            d = describe(r)
            row = {"size_class": f"{cen:g} kWh (±{half:g})", "battery": d["battery"],
                   "nominal_kwh": size[r["_bid"]], "usable_kwh": d["usable_kwh"], "price_eur": d["price_eur"],
                   "eur_per_kwh_usable": d["eur_per_kwh_usable"], "strategy": d["strategy"],
                   "saving_eur_year_2027": d["saving_eur_year_2027"], "payback_years": d["payback_years"],
                   "npv_eur": d["npv_eur"], "lifetime_net_saving_eur": r.get("lifetime_net_saving_eur", np.nan),
                   "candidates": len(sel), "days_full": d["days_full"]}
            if prev is not None:
                ds = row["saving_eur_year_2027"] - prev["saving_eur_year_2027"]
                dp = row["price_eur"] - prev["price_eur"]
                row["extra_saving_vs_smaller"] = ds
                row["extra_price_vs_smaller"] = dp
                row["payback_of_extra_years"] = dp / ds if ds > 0 else np.inf
            rows.append(row)
            prev = row
        return pd.DataFrame(rows) if rows else None

    @staticmethod
    def size_class_text(classes: pd.DataFrame) -> str:
        best_npv = classes.loc[classes.npv_eur.idxmax()]
        fastest = classes.loc[classes.payback_years.idxmin()]
        lines = ["", "**Which size?** The best-value battery per size class (fastest payback within the class):"]
        for r in classes.itertuples():
            s = (f"- **{r.size_class}:** {r.battery}, €{r.price_eur:,.0f}, saves €{r.saving_eur_year_2027:,.0f}/yr "
                 f"with `{r.strategy}`, pays back in {r.payback_years:.1f} years, net present value €{r.npv_eur:,.0f}")
            extra = getattr(r, "extra_saving_vs_smaller", np.nan)
            if isinstance(extra, float) and not np.isnan(extra):
                pe = r.payback_of_extra_years
                s += (f"; the step up adds €{extra:,.0f}/yr for €{r.extra_price_vs_smaller:,.0f} more"
                      + (f" (that extra pays back in {pe:.1f} years)" if np.isfinite(pe) else " (never pays back)"))
            lines.append(s + ".")
        if best_npv.size_class == fastest.size_class:
            lines.append(f"- The {fastest.size_class} class wins on both payback and lifetime value.")
        else:
            lines.append(f"- Fastest payback: {fastest.size_class}; most money over the battery's life (net present "
                         f"value): {best_npv.size_class}. Pick the larger one only if you are happy to wait longer "
                         "for the money back.")
        return "\n".join(lines)

    def advice_contract(self):
        """The contract the advice is built on: the cheapest dynamic contract when price history exists."""
        return self.cheapest_dynamic() or self.current

    def combination_table(self, b):
        """Yearly cost from 2027 (last 3 price years) for current/cheapest contract, with and without battery."""
        e = getattr(self, "expected_by_contract", {}) or {}
        rec = self.records
        rows = []
        for c in [x for x in (self.current, self.cheapest_dynamic()) if x is not None]:
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

    def advice_charts(self, sec, b, dyn):
        cap = self.conn.battery_cap_w(b.phases)
        scn = "nosal_min50"
        c = dyn or self.current or (self.contracts[0] if self.contracts else None)
        if c is None:
            return
        rk = self.payback_summary["ranked"]
        h = rk[(rk._bid == b.id) & (rk._cid == c.id) & (rk.analysis == "headline")
               & (rk.price_variant == "NL current") & (rk.strategy != "perfect_foresight")]
        strat = h.sort_values("payback_years").iloc[0]["strategy"] if len(h) else "self_consumption"
        year = self.head_years[-1] if (c.is_dynamic and self.head_years) else "profile"
        pers = self.periods(c, scn, [year] if c.is_dynamic else None)
        if not pers:
            return
        _, per, idx, _ = pers[-1]
        wear = self.wear_for(b)[0]
        m = getattr(self, "_meta", {}).get((b.id, c.id), {})
        r = self.sim_strategy(strat, b, cap, per, idx, c, scn, wear, windows=m.get("windows"), keep_trace=True)
        net = per.imp - per.exp
        u, s_ = marginal_values(per, c, self.regimes[scn], self.taxes, self.net_importer)
        per_q = (per.imp - r.imp) * u - (per.exp - r.exp) * s_
        saving = pd.Series(per_q, index=idx).groupby(idx.date).sum()
        saving.index = pd.to_datetime(saving.index)
        soc = r.soc if r.soc is not None else np.zeros(len(net))
        daily = extras.daily_battery(idx, net, r.batt_ac, soc, b.usable_kwh, (b.standby_w or 0) / 4000.0)
        label = f"{year} prices" if year != "profile" else "your profile year"
        sec.csv_only.add("daily_overview")
        sec.tables["daily_overview"] = daily.assign(saving_eur=saving.values).reset_index(names="date").assign(
            date=lambda d: d.date.dt.strftime("%Y-%m-%d")).round(2)
        f = self.fig(charts.daily_charge, daily, b.usable_kwh, f"{b.name}: solar surplus and storage per day")
        if f:
            sec.figures["daily_charge"] = f
        dx = extras.ts(pd.DatetimeIndex(daily.index).tz_localize("Europe/Amsterdam"))
        sec.interactive["daily_charge"] = {
            "title": f"{b.name} – per day: solar surplus, stored energy and grid import ({label}, 2027 rules)",
            "y": "kWh per day", "daily": True, "height": 340, "x": dx,
            "series": [extras.series("Solar surplus (kWh)", daily.solar_surplus_kwh, "#e9b44c", "bar"),
                       extras.series("Stored from solar (kWh)", daily.charged_from_solar_kwh, "#81b29a", "bar"),
                       extras.series("Charged from grid (kWh)", daily.charged_from_grid_kwh, "#9c6644", "bar"),
                       extras.series("Grid import without battery (kWh)", daily.import_before_kwh, "#3d405b"),
                       extras.series("Grid import with battery (kWh)", daily.import_after_kwh, "#e07a5f"),
                       extras.series(f"Usable capacity ({b.usable_kwh:.1f} kWh)", [b.usable_kwh] * len(daily),
                                     "#2a6f97", dash=[6, 4])]}
        f = self.fig(charts.daily_savings, saving, f"{b.name} with {c.label}: saving per day ({label}, 2027 rules)")
        if f:
            sec.figures["daily_savings"] = f
        sec.interactive["daily_savings"] = {
            "title": f"{b.name} with {c.label} – saving per day ({label}, 2027–2029 rules, strategy {strat})",
            "y": "€ per day", "y2": "€ cumulative", "daily": True, "height": 320,
            "x": extras.ts(pd.DatetimeIndex(saving.index).tz_localize("Europe/Amsterdam")),
            "series": [extras.series("Saving that day (€)", saving.values, "#81b29a", "bar"),
                       extras.series("Cumulative saving (€)", saving.cumsum().values, "#2a6f97", scale="y2", width=2)]}
        sec.md += (f"\n\nDaily charts: {b.name} on {c.label}, strategy `{strat}`, {label} under the 2027–2029 rules. "
                   f"Total saving that year: €{saving.sum():,.0f}; best day €{saving.max():,.2f}, "
                   f"{int((saving <= 0.01).sum())} days with (almost) no saving.")

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
                variants = b.price_variants(d.get("de_travel_cost_eur", 0.0), self.cfg.get("blackfriday"))
                if self.opts.price_variant != "all":
                    variants = {k: v for k, v in variants.items() if k == self.opts.price_variant}
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
                                   "strategy": strat, "analysis": aname, "price_variant": vname, "price_eur": price,
                                   "eur_per_kwh_usable": round(price / b.usable_kwh) if b.usable_kwh else None,
                                   "payback_years": pb["payback_years"],
                                   "payback_min": min(per_year) if per_year else pb["payback_years"],
                                   "payback_max": max(per_year) if per_year else pb["payback_years"],
                                   "npv_eur": pb["npv"], "lifetime_net_saving_eur": pb["lifetime_net"],
                                   "efc_per_year": efc, "end_of_life_year": pb["eol_year"],
                                   "life_limit": pb["limit"], "estimated": "yes" if b.estimated else "",
                                   "_bid": bid, "_cid": cid})
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

    def savings_year_chart(self, rec):
        if rec.empty or self.prices is None:
            return None
        dyn = rec[rec.price_year.apply(lambda x: isinstance(x, (int, np.integer))) & (rec.scenario == "nosal_min50")]
        if dyn.empty:
            return None
        best = dyn.groupby(["battery_id", "contract_id", "strategy"])["saving"].mean().reset_index()
        best = best.sort_values("saving", ascending=False).drop_duplicates("battery_id").head(3)
        series = {}
        for r in best.itertuples():
            s = dyn[(dyn.battery_id == r.battery_id) & (dyn.contract_id == r.contract_id) & (dyn.strategy == r.strategy)]
            series[f"{self.bname(r.battery_id)} – {r.strategy}"] = s.set_index("price_year")["saving"].sort_index()
        return self.fig(charts.savings_years, series, "Yearly savings per price year (2027–2029 rules)")


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
    return {"payback_years": pay, "npv": npv - price, "lifetime_net": cum - price,
            "eol_year": f"{start.year + (start.dayofyear / 365.0) + life_years:.1f}", "limit": limit}
