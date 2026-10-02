"""Synthetic day-ahead price files in a few plausible jeroen.nl-like layouts,
used only by tests (no real price data is shipped with the repo)."""
import os

import numpy as np
import pandas as pd


def synth_prices(year: int, seed: int = 0, quarter: bool = False) -> pd.Series:
    rng = np.random.default_rng(seed + year)
    idx = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="15min" if quarter else "h",
                        inclusive="left", tz="Europe/Amsterdam")
    h = idx.hour + idx.minute / 60
    base = 0.08 + 0.02 * (year - 2013) / 10
    daily = 0.05 * np.cos((h - 19) / 24 * 2 * np.pi) - 0.04 * np.exp(-((h - 13) ** 2) / 6) * (1 + np.sin((idx.dayofyear - 80) / 365 * 2 * np.pi))
    p = base + daily + rng.normal(0, 0.01, len(idx))
    return pd.Series(p, index=idx)


def write_year(path_dir: str, year: int, layout: str = "comma_kwh") -> str:
    s = synth_prices(year, quarter=year >= 2026)
    if year == 2026:
        s = s[s.index < "2026-10-01"]   # partial year, like the real 2026 file
    f = os.path.join(path_dir, f"jeroen_punt_nl_dynamische_stroomprijzen_jaar_{year}.csv")
    t = s.index.tz_localize(None).strftime("%Y-%m-%d %H:%M")
    if layout == "comma_kwh":
        pd.DataFrame({"datum": t, "prijs_excl_belastingen": s.values.round(5)}).to_csv(f, index=False)
    elif layout == "semicolon_mwh":
        df = pd.DataFrame({"Datum": t, "Prijs EPEX (EUR/MWh)": [f"{v * 1000:.2f}".replace(".", ",") for v in s.values]})
        df.to_csv(f, index=False, sep=";")
    elif layout == "incl_only":
        vat, eb = 0.21, 0.09161
        pd.DataFrame({"datum": t, "prijs_incl_belastingen": ((s.values + eb) * (1 + vat)).round(6)}).to_csv(f, index=False)
    return f
