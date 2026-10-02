"""Day-ahead price history: load every jeroen.nl year file by glob, normalise
to €/kWh kale (EPEX, excl. taxes), resample to 15 minutes, and replay price
years against the meter profile."""
from __future__ import annotations

import glob
import io
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .p1 import localize
from .taxes import Taxes


@dataclass
class PriceReport:
    files: list = field(default_factory=list)
    columns_used: dict = field(default_factory=dict)
    duplicates_removed: int = 0
    missing_hours: dict = field(default_factory=dict)
    years: list = field(default_factory=list)
    partial_years: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    crosscheck: str = ""


def _read_any(source) -> pd.DataFrame:
    if hasattr(source, "read"):
        text = source.read()
        text = text.decode("utf-8-sig") if isinstance(text, bytes) else text
    else:
        with open(source, encoding="utf-8-sig") as f:
            text = f.read()
    head = text.splitlines()[0] if text else ""
    sep = ";" if head.count(";") > head.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    return df


def _to_float(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.replace("€", "", regex=False).str.replace(" ", "", regex=False)
    # Decimal comma ("0,123") -> dot; thousands dots ("1.234,5") removed first.
    has_comma = s.str.contains(",", regex=False)
    s = s.where(~has_comma, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


def parse_price_frame(df: pd.DataFrame, cfg: dict, taxes: Taxes, name: str = "") -> tuple[pd.Series, str]:
    """Return a tz-aware series of kale prices in €/kWh and a description."""
    pc = cfg.get("prices", {})
    tz = pc.get("timezone", "Europe/Amsterdam")
    # Time: first column(s) that parse as datetimes. Some files split date and hour.
    time_col, hour_col = None, None
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ("datum", "date", "tijd", "time", "start", "van", "from")):
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=_dayfirst(df[c]))
            if parsed.notna().mean() > 0.9:
                time_col = c
                break
    if time_col is None:
        for c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=_dayfirst(df[c]))
            if parsed.notna().mean() > 0.9:
                time_col = c
                break
    if time_col is None:
        raise ValueError(f"{name}: no date/time column found in {list(df.columns)}")
    t = pd.to_datetime(df[time_col], errors="coerce", dayfirst=_dayfirst(df[time_col]))
    if (t.dt.hour == 0).all():
        for c in df.columns:
            if c != time_col and re.search(r"uur|hour|tijd|time", c.lower()):
                h = pd.to_numeric(df[c].astype(str).str.extract(r"(\d{1,2})")[0], errors="coerce")
                if h.notna().mean() > 0.9:
                    hour_col = c
                    t = t + pd.to_timedelta(h, unit="h")
                    break
    numeric = {c: _to_float(df[c]) for c in df.columns if c not in (time_col, hour_col)}
    numeric = {c: v for c, v in numeric.items() if v.notna().mean() > 0.9}
    if not numeric:
        raise ValueError(f"{name}: no numeric price column found")

    def pick(hints):
        for h in hints:
            for c in numeric:
                if h in c.lower():
                    return c
        return None

    kale_col = pick(pc.get("kale_hints", ["excl", "kale"]))
    incl_col = pick(pc.get("incl_hints", ["incl"]))
    if kale_col is not None and incl_col is not None and kale_col == incl_col:
        incl_col = None
    col = kale_col or (incl_col if incl_col else next(iter(numeric)))
    values = numeric[col]
    unit = "€/kWh"
    if values.abs().median() > float(pc.get("mwh_threshold", 5.0)):
        values = values / 1000.0
        unit = "€/MWh"
    ts = localize(t, tz)
    s = pd.Series(values.values, index=pd.DatetimeIndex(ts)).dropna()
    s = s[~s.index.isna()]
    kind = "kale"
    if kale_col is None and incl_col is not None:
        # All-in consumer price: strip VAT and energy tax to get back to kale.
        vat = taxes.vat
        eb = np.array([taxes.energy_tax(y) for y in s.index.year])
        s = s / (1 + vat) - eb
        kind = "incl. taxes (converted to kale)"
    desc = f"{col} [{unit}, {kind}]"
    return s.sort_index(), desc


def _dayfirst(col: pd.Series) -> bool:
    sample = str(col.dropna().iloc[0]) if col.notna().any() else ""
    return bool(re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}", sample))


def load_price_history(cfg: dict, taxes: Taxes, base_dir: str = ".", sources: dict | None = None):
    """Load all price files. `sources` maps names to file-like objects (web)."""
    rep = PriceReport()
    parts = []
    if sources is None:
        sources = {}
        # Extra (scraped) sources first, so the main jeroen.nl files win on overlap.
        for pattern in list(cfg["paths"].get("extra_price_globs") or []) + [cfg["paths"]["price_glob"]]:
            if not os.path.isabs(pattern):
                pattern = os.path.join(base_dir, pattern)
            for f in sorted(glob.glob(pattern)):
                sources[os.path.basename(f)] = f
    for name, src in sources.items():
        try:
            s, desc = parse_price_frame(_read_any(src), cfg, taxes, name)
        except Exception as e:  # report and continue: one bad file must not stop the run
            rep.notes.append(f"{name}: could not be read ({e})")
            continue
        rep.files.append(name)
        rep.columns_used[name] = desc
        parts.append(s)
    if not parts:
        rep.notes.append("No day-ahead price files found: dynamic contracts and the Dynamic/Sell "
                         "strategies are skipped. Add jeroen_punt_nl_dynamische_stroomprijzen_jaar_YYYY.csv "
                         "files to data/.")
        return None, rep
    s = pd.concat(parts).sort_index()
    rep.duplicates_removed = int(s.index.duplicated().sum())
    s = s[~s.index.duplicated(keep="last")]
    s15 = to_quarter_hours(s)
    for y, g in s15.groupby(s15.index.year):
        expected = pd.date_range(pd.Timestamp(year=y, month=1, day=1, tz=s15.index.tz),
                                 pd.Timestamp(year=y + 1, month=1, day=1, tz=s15.index.tz),
                                 freq="15min", inclusive="left")
        missing = len(expected.difference(g.index)) / 4
        rep.missing_hours[int(y)] = missing
        months = sorted(set(g.index.month))
        if len(months) < 12:
            rep.partial_years[int(y)] = months
    rep.years = sorted(rep.missing_hours)
    return s15, rep


def to_quarter_hours(s: pd.Series) -> pd.Series:
    """Hourly prices repeated per quarter; 15-min data kept as is."""
    s = s.copy()
    s.index = s.index.tz_convert("UTC")
    out = []
    step = s.index.to_series().diff().dt.total_seconds().fillna(3600)
    hourly = s[step.values >= 3600]
    quarter = s[step.values < 3600]
    if len(hourly):
        h = hourly.copy()
        reps = [h.copy() for _ in range(4)]
        for k, r in enumerate(reps):
            r.index = r.index + pd.Timedelta(minutes=15 * k)
        out.append(pd.concat(reps))
    if len(quarter):
        out.append(quarter)
    q = pd.concat(out).sort_index()
    # 15-min values win over the expanded hourly ones where both exist.
    q = q[~q.index.duplicated(keep="last")]
    q.index = q.index.tz_convert("Europe/Amsterdam")
    return q


def price_overview(s15: pd.Series) -> pd.DataFrame:
    """Per year: average price, average daily max-min spread, negative hours."""
    utc = s15.index.tz_convert("UTC")
    hourly = s15.groupby(utc.floor("h")).mean()
    hourly.index = hourly.index.tz_convert(s15.index.tz)
    df = pd.DataFrame({"p": hourly})
    df["year"] = df.index.year
    df["day"] = df.index.date
    daily = df.groupby(["year", "day"])["p"].agg(lambda x: x.max() - x.min())
    out = pd.DataFrame({
        "avg_price_eur_kwh": df.groupby("year")["p"].mean(),
        "avg_daily_spread_eur_kwh": daily.groupby(level=0).mean(),
        "negative_price_hours": df.groupby("year")["p"].apply(lambda x: int((x < 0).sum())),
        "hours": df.groupby("year")["p"].size(),
    })
    return out


def replay_prices(profile_index: pd.DatetimeIndex, prices: pd.Series, year: int):
    """Map a price year onto the meter profile (section 5).

    Returns (target_index, profile_positions, price_values): for every
    15-min interval of calendar `year` that has a price, the position of the
    meter interval with the same ISO week, weekday and quarter-hour."""
    tz = profile_index.tz
    py = prices[prices.index.year == year]
    if py.empty:
        return None
    iso = profile_index.isocalendar()
    qh = profile_index.hour * 4 + profile_index.minute // 15
    key = (iso.week.values.astype(int) * 7 + profile_index.weekday.values) * 100 + qh.values
    lookup = pd.Series(np.arange(len(profile_index)), index=key)
    lookup = lookup[~lookup.index.duplicated(keep="last")]
    tiso = py.index.isocalendar()
    week = tiso.week.values.astype(int)
    tq = py.index.hour * 4 + py.index.minute // 15
    tkey = (week * 7 + py.index.weekday.values) * 100 + tq.values
    pos = lookup.reindex(tkey).values
    # Week 53 or a hole: fall back to week 52, then to the same weekday/quarter anywhere.
    if np.isnan(pos).any():
        alt = ((np.minimum(week, 52) * 7 + py.index.weekday.values) * 100 + tq.values)
        pos = np.where(np.isnan(pos), lookup.reindex(alt).values, pos)
    if np.isnan(pos).any():
        wkq = pd.Series(np.arange(len(profile_index)), index=profile_index.weekday.values * 100 + qh.values)
        wkq = wkq[~wkq.index.duplicated()]
        alt2 = py.index.weekday.values * 100 + tq.values
        pos = np.where(np.isnan(pos), wkq.reindex(alt2).values, pos)
    ok = ~np.isnan(pos)
    return py.index[ok], pos[ok].astype(int), py.values[ok]


def load_energieknl(source) -> pd.DataFrame | None:
    """The energieKNL file. In the current repo it holds yearly delivery costs
    per supplier and month, not day-ahead prices; detected and reported."""
    try:
        df = _read_any(source)
    except Exception:
        return None
    return df


def crosscheck(s15: pd.Series | None, knl: pd.DataFrame | None, threshold: float) -> str:
    if knl is None:
        return "energieKNL file not present; cross-check skipped."
    cols = [c.lower() for c in knl.columns]
    if "leveringskost" in cols and not any("prijs" in c or "price" in c for c in cols):
        sup = knl.iloc[:, 0].nunique()
        return (f"energieKNL file contains yearly delivery costs ('leveringskost') for {sup} dynamic suppliers, "
                "not day-ahead prices, so the price cross-check cannot run. Shown for reference in the report.")
    if s15 is None:
        return "No jeroen.nl prices loaded; cross-check skipped."
    try:
        other, _ = parse_price_frame(knl, {"prices": {}}, Taxes({}), "energieKNL")
    except Exception as e:
        return f"energieKNL file could not be parsed as prices ({e})."
    o15 = to_quarter_hours(other)
    common = s15.index.intersection(o15.index)
    if not len(common):
        return "jeroen.nl and energieKNL prices do not overlap."
    diff = (s15.loc[common] - o15.loc[common]).abs()
    n = int((diff > threshold).sum())
    return (f"Overlap {common[0]:%Y-%m-%d}..{common[-1]:%Y-%m-%d}: {n} of {len(common)} quarter-hours "
            f"differ by more than {threshold * 100:.1f} ct/kWh (max {diff.max() * 100:.2f} ct).")
