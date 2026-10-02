"""Energy tax, tax reduction, VAT and grid charges per year (section 4)."""
from __future__ import annotations


class Taxes:
    def __init__(self, cfg: dict):
        t = cfg.get("taxes", {}) if cfg else {}
        self.vat = float(t.get("vat", 0.21))
        self.escalation = float(t.get("escalation_pct", 0.0)) / 100.0
        self.years = {int(k): dict(v) for k, v in (t.get("years") or {}).items()}
        self._load_scraped(cfg)
        self.grid_cfg = (cfg or {}).get("grid", {})
        self.warnings: list[str] = []

    def _load_scraped(self, cfg: dict):
        """data/online/taxes.csv (written by the scraper) overrides config years."""
        import os

        import pandas as pd

        from .config import resolve
        path = resolve(cfg, (cfg or {}).get("paths", {}).get("taxes_file")) if cfg and "_base_dir" in cfg else None
        if not path or not os.path.exists(path):
            return
        try:
            df = pd.read_csv(path)
        except Exception:
            return
        for r in df.to_dict("records"):
            y = int(r["year"])
            d = self.years.setdefault(y, {})
            for k in ("energy_tax_eur_kwh", "tax_reduction_eur_year", "source_url", "retrieved_at"):
                if k in r and pd.notna(r[k]):
                    d[k] = r[k]
            d["verify"] = False

    def _year(self, year: int) -> tuple[dict, int]:
        if not self.years:
            return {"energy_tax_eur_kwh": 0.0, "tax_reduction_eur_year": 0.0}, 0
        if year in self.years:
            return self.years[year], 0
        known = sorted(self.years)
        if year > known[-1]:
            return self.years[known[-1]], year - known[-1]
        # Before the first known year: hold the earliest rates (price years are
        # replayed under current rules anyway).
        return self.years[known[0]], 0

    def energy_tax(self, year: int) -> float:
        d, n = self._year(year)
        return float(d.get("energy_tax_eur_kwh", 0.0)) * (1 + self.escalation) ** n

    def tax_reduction(self, year: int) -> float:
        d, n = self._year(year)
        return float(d.get("tax_reduction_eur_year", 0.0)) * (1 + self.escalation) ** n

    def source(self, year: int) -> str:
        d, n = self._year(year)
        extra = f" (held flat from latest year, +{self.escalation:.1%}/yr)" if n else ""
        return f"{d.get('source_url', 'config.yaml')} retrieved {d.get('retrieved_at', '?')}{extra}"

    def grid_eur_year(self, connection_label: str) -> float:
        table = self.grid_cfg.get("eur_year_excl_vat", {}) or {}
        v = table.get(connection_label)
        if v is None:
            fallback = table.get("3x25") or 0.0
            msg = (f"Grid charges for {connection_label} are not configured; using the 3x25 value "
                   f"(€{fallback:.2f}/yr). They are equal for every supplier, so rankings are unaffected.")
            if msg not in self.warnings:
                self.warnings.append(msg)
            return float(fallback)
        return float(v)


def regime_for_year(cfg: dict, year: int) -> str:
    starts = cfg.get("regime_by_year", {"saldering": 0, "nosal_min50": 2027, "nosal_2030": 2030})
    best, best_y = None, -1
    for name, y in starts.items():
        if year >= int(y) and int(y) >= best_y:
            best, best_y = name, int(y)
    return best
