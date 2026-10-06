"""Contract schema shared by fixed and dynamic contracts (section 3.1).

All money values are stored excl. VAT; the cost engine adds VAT."""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field, asdict

import pandas as pd


@dataclass
class Contract:
    id: str
    supplier: str
    product: str
    type: str                                # "fixed" or "dynamic"
    duration_months: int = 12
    start_date: str = ""
    end_date: str = ""
    price_single: float | None = None        # €/kWh excl. VAT, kale (fixed)
    price_t1: float | None = None            # dal / low
    price_t2: float | None = None            # normaal / normal
    markup: float = 0.0                      # inkoopvergoeding (dynamic)
    fixed_eur_year: float = 0.0              # vaste leveringskosten
    feed_in: float | None = None             # fixed feed-in €/kWh
    feed_in_markup: float = 0.0              # dynamic: spot + markup
    feed_in_cost_kwh: float = 0.0
    feed_in_cost_tiers: list = field(default_factory=list)   # [(lo, hi, €/yr)]
    welcome_bonus: float = 0.0
    cancellation_fee: float = 0.0
    netting_method: str = "tariff_blocks"    # fixed: tariff_blocks; dynamic: spot_kwh_net
    post2027_feed_in: float | None = None
    post2027_feed_in_cost_tiers: list | None = None
    post2027_feed_in_cost_kwh: float | None = None   # per-kWh feed-in cost from 2027, if it differs
    source: str = ""
    retrieved_at: str = ""
    verified: bool = False
    notes: str = ""

    @property
    def is_dynamic(self) -> bool:
        return self.type == "dynamic"

    @property
    def label(self) -> str:
        return f"{self.supplier} – {self.product}"

    def kale_price(self, is_low: bool | None = None) -> float:
        if is_low is True and self.price_t1 is not None:
            return self.price_t1
        if is_low is False and self.price_t2 is not None:
            return self.price_t2
        if self.price_single is not None:
            return self.price_single
        vals = [v for v in (self.price_t1, self.price_t2) if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def feed_in_cost_per_kwh(self, post2027: bool = False) -> float:
        if post2027 and self.post2027_feed_in_cost_kwh is not None:
            return self.post2027_feed_in_cost_kwh
        return self.feed_in_cost_kwh

    def feed_in_cost_year(self, export_kwh: float, post2027: bool = False) -> float:
        tiers = self.post2027_feed_in_cost_tiers if (post2027 and self.post2027_feed_in_cost_tiers) \
            else self.feed_in_cost_tiers
        for lo, hi, eur in tiers or []:
            if lo <= export_kwh < hi:
                return float(eur)
        return 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["feed_in_cost_tiers"] = tiers_to_text(self.feed_in_cost_tiers)
        d["post2027_feed_in_cost_tiers"] = tiers_to_text(self.post2027_feed_in_cost_tiers or [])
        return d


def tiers_to_text(tiers) -> str:
    return ";".join(f"{lo:g}-{'inf' if math.isinf(hi) else f'{hi:g}'}:{eur:g}" for lo, hi, eur in tiers or [])


def tiers_from_text(text) -> list:
    out = []
    if not isinstance(text, str) or not text.strip():
        return out
    for part in text.split(";"):
        rng, eur = part.split(":")
        lo, hi = rng.split("-")
        out.append((float(lo), math.inf if hi.strip() in ("inf", "") else float(hi), float(eur)))
    return out


# ---------------------------------------------------------------------------
# vast.txt

_NUM = r"€\s*([\d\.]+,\d+)"


def _num(text: str) -> float:
    return float(text.replace(".", "").replace(",", "."))


def parse_vast(text: str, source: str = "data/vast.txt") -> tuple[Contract, dict]:
    """Parse the supplier's tariff sheet. Returns the contract and extra facts
    (grid charges, the energy tax implied by the incl.-levies section)."""
    sections = re.split(r"\n(?=Inclusief btw|Exclusief btw)", "\n" + text)
    parsed = {}
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        title = sec.splitlines()[0].strip()
        parsed[title] = _parse_section(sec)
    excl = parsed.get("Exclusief btw") or next(iter(parsed.values()))
    facts = {}
    incl = parsed.get("Inclusief btw")
    levies = parsed.get("Inclusief btw en overheidsheffingen")
    if incl and levies and "Enkeltarief" in incl and "Enkeltarief" in levies:
        facts["energy_tax_incl_vat_implied"] = levies["Enkeltarief"] - incl["Enkeltarief"]
    for k, v in excl.items():
        if k.startswith("Netbeheerkosten"):
            facts["grid_eur_year_excl_vat"] = v
            facts["grid_operator"] = k
    m1 = re.search(r"Startdatum\s*\n\s*(\d{2}-\d{2}-\d{4})", text)
    m2 = re.search(r"Einddatum\s*\n\s*(\d{2}-\d{2}-\d{4})", text)
    start = pd.to_datetime(m1.group(1), dayfirst=True) if m1 else None
    end = pd.to_datetime(m2.group(1), dayfirst=True) if m2 else None
    months = int(round((end - start).days / 30.44)) if start is not None and end is not None else 12

    tiers = []
    for k, v in excl.items():
        m = re.match(r"Terugleverkosten \w+ \((.+) kWh\)", k)
        if m:
            rng = m.group(1).strip()
            if rng.startswith(">="):
                tiers.append((float(rng[2:].strip()), math.inf, v))
            else:
                lo, hi = rng.split("-")
                tiers.append((float(lo), float(hi), v))
    tiers.sort()
    c = Contract(
        id="current_fixed",
        supplier="Current contract",
        product="fixed",
        type="fixed",
        duration_months=months,
        start_date=start.strftime("%Y-%m-%d") if start is not None else "",
        end_date=end.strftime("%Y-%m-%d") if end is not None else "",
        price_single=excl.get("Enkeltarief"),
        price_t1=excl.get("Daltarief"),
        price_t2=excl.get("Normaaltarief"),
        fixed_eur_year=excl.get("Vaste leveringskosten", 0.0),
        feed_in=excl.get("Teruglevering normaal", excl.get("Teruglevering enkeltarief", 0.0)),
        feed_in_cost_kwh=excl.get("Terugleverkosten normaal", excl.get("Terugleverkosten enkel", 0.0)) or 0.0,
        feed_in_cost_tiers=tiers,
        netting_method="tariff_blocks",
        source=source,
        retrieved_at="",
        verified=True,
        notes="Parsed from the 'Exclusief btw' section.",
    )
    return c, facts


def _parse_section(sec: str) -> dict:
    out = {}
    lines = [l.strip() for l in sec.splitlines()]
    label = None
    for line in lines:
        if not line:
            continue
        m = re.match(_NUM + r"\s*per\s*(kWh|dag|jaar)", line)
        if m:
            if label is None:
                continue
            val, unit = _num(m.group(1)), m.group(2)
            if unit == "kWh":
                out[label] = val
            elif unit == "jaar":
                out[label] = val          # per-year value wins over per-day
            elif unit == "dag" and label not in out:
                out[label] = val * 365
        else:
            label = line
    return out


# ---------------------------------------------------------------------------
# contracts.csv


def _f(v, default=None):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
            return default
        return float(str(v).replace(",", "."))
    except ValueError:
        return default


def load_contracts_csv(source) -> list[Contract]:
    df = pd.read_csv(source, dtype=str).fillna("")
    out = []
    for _, r in df.iterrows():
        typ = r.get("type", "fixed").strip().lower()
        fixed_month = _f(r.get("fixed_delivery_eur_month_excl_vat"), 0.0)
        out.append(Contract(
            id=r["id"], supplier=r.get("supplier", ""), product=r.get("product", ""), type=typ,
            duration_months=int(_f(r.get("duration_months"), 12)),
            price_single=_f(r.get("price_single_excl_vat")),
            price_t1=_f(r.get("price_t1_excl_vat")),
            price_t2=_f(r.get("price_t2_excl_vat")),
            markup=_f(r.get("markup_excl_vat"), 0.0),
            fixed_eur_year=fixed_month * 12,
            feed_in=_f(r.get("feed_in_eur_kwh_excl_vat")),
            feed_in_markup=_f(r.get("feed_in_markup_excl_vat"), 0.0),
            feed_in_cost_kwh=_f(r.get("feed_in_cost_eur_kwh_excl_vat"), 0.0),
            feed_in_cost_tiers=tiers_from_text(r.get("feed_in_cost_tiers")),
            welcome_bonus=_f(r.get("welcome_bonus_eur"), 0.0),
            cancellation_fee=_f(r.get("cancellation_fee_eur"), 0.0),
            netting_method=r.get("netting_method") or ("spot_kwh_net" if typ == "dynamic" else "tariff_blocks"),
            post2027_feed_in=_f(r.get("post2027_feed_in_eur_kwh_excl_vat")),
            post2027_feed_in_cost_tiers=tiers_from_text(r.get("post2027_feed_in_cost_tiers")) or None,
            post2027_feed_in_cost_kwh=_f(r.get("post2027_feed_in_cost_eur_kwh_excl_vat")),
            source=r.get("source_url", ""), retrieved_at=r.get("retrieved_at", ""),
            verified=str(r.get("verified", "")).lower() in ("yes", "true", "1"),
            notes=r.get("notes", ""),
        ))
    return out


def contract_from_form(d: dict) -> Contract:
    """Web form input (section 9): same fields as the schema, excl. VAT."""
    c = Contract(id="current_fixed", supplier=d.get("supplier") or "Current contract",
                 product=d.get("product") or "uploaded", type=d.get("type", "fixed"))
    for k in ("price_single", "price_t1", "price_t2", "feed_in", "markup", "feed_in_markup",
              "feed_in_cost_kwh", "fixed_eur_year", "welcome_bonus", "cancellation_fee"):
        v = _f(d.get(k))
        if v is not None:
            setattr(c, k, v)
    c.duration_months = int(_f(d.get("duration_months"), 12))
    c.feed_in_cost_tiers = tiers_from_text(d.get("feed_in_cost_tiers", ""))
    c.netting_method = "spot_kwh_net" if c.is_dynamic else "tariff_blocks"
    c.source = "web form"
    return c


def read_text(source) -> str:
    if hasattr(source, "read"):
        t = source.read()
        return t.decode("utf-8") if isinstance(t, bytes) else t
    with open(source, encoding="utf-8") as f:
        return f.read()
