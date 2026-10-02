"""Runs the workflow of section 10 and produces report sections.

Each section is handed to `on_section` as soon as it is computed, so the
CLI can print progress and the website can show results step by step."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import charts
from .battery import (GRID_CHARGE, SELF, ZERO_IMPORT, Battery, in_scope, load_batteries, perfect_foresight,
                      plan_dynamic, plan_hbc, plan_timed, simulate)
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
        self.step_current_contract()
        self.step_contracts()
        self.step_batteries()
        self.step_breakeven()
        self.step_sanity()
        self.step_payback()
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
                    tot = np.array([v[0].total for v in sel.values()])
                    real = np.array([v[1].total for v in sel.values()])
                    imp = np.array([v[0].import_kwh for v in sel.values()])
                    inc = np.array([v[0].feed_in_income for v in sel.values()])
                    row = {"contract": c.label, "type": c.type, "verified": "yes" if c.verified else "no",
                           "expected_eur_year": tot.mean(), "min": tot.min(), "max": tot.max(),
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
                  "switching rules on 1 January 2027.")
        sec.md = "\n".join(md)
        self.expected_by_contract = expected_by_contract
        self.emit(sec)

    # ------------------------------------------------------------------ step 4
    def step_batteries(self):
        sec = Section("batteries", "4. Battery dataset")
        src = self.inp.batteries_csv or self.path("batteries_file")
        bats = load_batteries(src, self.cfg.get("battery_defaults", {})) if src else []
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
                              cycles=cycles, de_travel=d.get("de_travel_cost_eur", 0.0))
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
        traces = {}
        for title, month in (("Summer week", 7), ("Winter week", 1)):
            sel = np.flatnonzero(idx.month == month)
            if len(sel) >= 7 * 96:
                s = sel[7 * 96: 14 * 96]
                traces[f"{title} ({idx[s[0]]:%Y-%m-%d})"] = (idx[s], r.soc[s], net[s])
        f = self.fig(charts.soc_weeks, traces)
        if f:
            sec.figures["soc_weeks"] = f
        self.sanity = {"battery": b.id, "efc": r.efc, "balance_error": balance}
        self.emit(sec)

    # ------------------------------------------------------------------ step 7
    def strategies_for(self, c: Contract) -> list[str]:
        enabled = self.cfg.get("strategies", {}).get("enabled", [])
        out = [s for s in enabled if s in ("self_consumption", "timed")]
        if c.is_dynamic:
            presets = self.cfg.get("strategies", {}).get("hbc_presets", {}) or {}
            out += [s for s in enabled if s in ("dynamic", "dynamic_sell", "perfect_foresight") or s in presets]
        return out

    def sim_strategy(self, strategy, b, cap, per: Period, idx, c, scn, wear, windows=None, scale=1.0,
                     rte=None, standby=None):
        net = per.imp - per.exp
        bb = b
        if rte is not None or standby is not None:
            from dataclasses import replace
            bb = replace(b, rte=rte if rte is not None else b.rte,
                         standby_w=standby if standby is not None else b.standby_w)
        if strategy == "self_consumption":
            return simulate(net, SELF, bb, cap)
        u, s = marginal_values(per, c, self.regimes[scn], self.taxes, self.net_importer)
        if strategy == "timed":
            modes = plan_timed(idx, windows or [])
            return simulate(net, modes, bb, cap)
        if strategy in ("dynamic", "dynamic_sell"):
            modes = plan_dynamic(idx, u, s, bb, cap, wear, scale, strategy == "dynamic_sell",
                                 self.cfg.get("strategies", {}).get("dynamic", {}), export=per.exp,
                                 solar_forecast=self.pv_for(idx), imports=per.imp)
            return simulate(net, modes, bb, cap)
        presets = self.cfg.get("strategies", {}).get("hbc_presets", {}) or {}
        if strategy in presets:
            return simulate(net, plan_hbc(idx, u, presets[strategy]), bb, cap)
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
                        years = self.head_years if strat == "perfect_foresight" else None
                        if strat == "perfect_foresight" and self.opts.quick:
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
              "`perfect_foresight` is the upper bound (headline years only).",
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
                variants = b.price_variants(d.get("de_travel_cost_eur", 0.0))
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
