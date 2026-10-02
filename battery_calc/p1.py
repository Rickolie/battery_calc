"""HomeWizard P1 meter data: load, validate, convert to per-interval kWh and
build a 12-month profile with synthetic months where data is missing."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

Q = pd.Timedelta(minutes=15)

FLAG_REAL = "real"
FLAG_INTERP = "interpolated"
FLAG_GAP = "gap_filled"
FLAG_SYNTH = "synthetic"


@dataclass
class P1Report:
    source: str = ""
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    rows: int = 0
    duplicates: int = 0
    resets: list = field(default_factory=list)
    interpolated_gaps: list = field(default_factory=list)
    long_gaps: list = field(default_factory=list)
    window: tuple | None = None
    missing_months: list = field(default_factory=list)
    partially_real_months: list = field(default_factory=list)
    low_confidence: bool = False
    tariff_rule_agreement: float | None = None
    notes: list = field(default_factory=list)


def localize(ts: pd.Series, tz: str) -> pd.Series:
    """Naive local timestamps -> tz-aware, handling the 23 h and 25 h days."""
    ts = pd.to_datetime(ts)
    if ts.dt.tz is not None:
        return ts.dt.tz_convert(tz)
    try:
        return ts.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        # Ambiguous hour without enough context: assume the first is DST.
        amb = ts.duplicated(keep="first")
        return ts.dt.tz_localize(tz, ambiguous=~amb.values, nonexistent="shift_forward")


def tariff_is_low(index: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    th = cfg.get("tariff_hours", {})
    start, end = th.get("normal_start_hour", 7), th.get("normal_end_hour", 21)
    hours = index.hour
    low = (hours < start) | (hours >= end)
    if th.get("weekend_is_low", True):
        low |= index.weekday >= 5
    return np.asarray(low)


def read_p1(source, cfg: dict) -> pd.DataFrame:
    """Read the raw export. `source` is a path or a file-like object."""
    p = cfg["p1"]
    raw = pd.read_csv(source)
    needed = [p["time_column"], p["import_t1"], p["import_t2"], p["export_t1"], p["export_t2"]]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"P1 file is missing columns {missing}; found {list(raw.columns)}")
    return raw


def load_p1(source, cfg: dict, label: str = "") -> tuple[pd.DataFrame, P1Report]:
    """Return per-interval data indexed by interval start (tz-aware)."""
    p = cfg["p1"]
    tz = p.get("timezone", "Europe/Amsterdam")
    raw = read_p1(source, cfg)
    rep = P1Report(source=label or str(source), rows=len(raw))

    raw = raw.copy()
    raw["_t"] = localize(raw[p["time_column"]], tz)
    raw = raw.dropna(subset=["_t"]).sort_values("_t")
    rep.duplicates = int(raw["_t"].duplicated().sum())
    raw = raw.drop_duplicates("_t", keep="last")

    cols = {"imp_t1": p["import_t1"], "imp_t2": p["import_t2"],
            "exp_t1": p["export_t1"], "exp_t2": p["export_t2"]}
    counters = pd.DataFrame({k: pd.to_numeric(raw[v], errors="coerce").values for k, v in cols.items()},
                            index=pd.DatetimeIndex(raw["_t"]).tz_convert("UTC"))
    counters = counters[~counters.index.duplicated()]

    # Regular 15-min grid in UTC (DST-safe), then interpolate short gaps.
    grid = pd.date_range(counters.index[0].ceil("15min"), counters.index[-1].floor("15min"), freq="15min")
    counters = counters.reindex(counters.index.union(grid)).sort_index()
    present = counters.notna().all(axis=1)
    max_gap = int(p.get("max_interpolate_gap_minutes", 60)) // 15

    # Counter resets: a counter going down. The interval of the reset is unknown.
    reset_mask = pd.Series(False, index=counters.index)
    for c in counters.columns:
        s = counters[c].dropna()
        d = s.diff()
        bad = d[d < -1e-6].index
        for t in bad:
            rep.resets.append((t.tz_convert(tz), c))
            reset_mask.loc[t] = True
            # Re-base everything after the reset so the series is monotonic again.
            jump = s.loc[:t].iloc[-2] - s.loc[t]
            counters.loc[counters.index >= t, c] += jump

    counters = counters.interpolate(method="time", limit_area="inside")
    counters = counters.reindex(grid)
    present = present.reindex(grid).fillna(False)

    # Identify runs of missing readings.
    miss = ~present.values
    runs = []
    i = 0
    while i < len(miss):
        if miss[i]:
            j = i
            while j < len(miss) and miss[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1

    deltas = counters.diff()
    if p.get("timestamp_marks", "interval_end") == "interval_end":
        deltas.index = deltas.index - Q
        shift = 1
    else:
        deltas = deltas.shift(-1)
        shift = 0
    deltas = deltas.iloc[1:] if shift else deltas.iloc[:-1]
    deltas = deltas.clip(lower=0)
    flag = pd.Series(FLAG_REAL, index=deltas.index, dtype=object)

    for a, b in runs:
        # Readings a..b-1 missing -> intervals covering them are affected.
        n = b - a
        idx_lo, idx_hi = max(a - shift, 0), min(b + 1 - shift, len(deltas))
        span = deltas.index[idx_lo:idx_hi]
        start, end = grid[a].tz_convert(tz), grid[b - 1].tz_convert(tz)
        if n <= max_gap:
            flag.loc[span] = FLAG_INTERP
            rep.interpolated_gaps.append((start, end))
        else:
            deltas.loc[span] = np.nan
            flag.loc[span] = FLAG_GAP
            rep.long_gaps.append((start, end))
    for t, _ in rep.resets:
        ts = t.tz_convert("UTC") - Q * shift
        if ts in deltas.index:
            deltas.loc[ts] = np.nan
            flag.loc[ts] = FLAG_GAP

    df = deltas.copy()
    df.index = df.index.tz_convert(tz)
    df["imp"] = df["imp_t1"] + df["imp_t2"]
    df["exp"] = df["exp_t1"] + df["exp_t2"]
    rule_low = tariff_is_low(df.index, cfg)
    moved_t1 = (df["imp_t1"] + df["exp_t1"]) > 0
    moved_t2 = (df["imp_t2"] + df["exp_t2"]) > 0
    low = np.where(moved_t1 & ~moved_t2, True, np.where(moved_t2 & ~moved_t1, False, rule_low))
    df["is_low"] = low.astype(bool)
    known = (moved_t1 ^ moved_t2).values
    if known.any():
        rep.tariff_rule_agreement = float((rule_low[known] == low[known]).mean())
    df["flag"] = flag.values
    rep.first, rep.last = df.index[0], df.index[-1] + Q
    return df, rep


# ---------------------------------------------------------------------------
# 12-month profile with synthetic months


def _month_starts(first: pd.Timestamp, n: int, tz: str) -> list[pd.Timestamp]:
    out = []
    y, m = first.year, first.month
    for _ in range(n + 1):
        out.append(pd.Timestamp(year=y, month=m, day=1, tz=tz))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean import/export per weekday and quarter-hour of the day."""
    real = df[df["flag"].isin([FLAG_REAL, FLAG_INTERP])].dropna(subset=["imp", "exp"])
    if real.empty:
        return pd.DataFrame()
    key = [real.index.weekday, real.index.hour * 4 + real.index.minute // 15]
    return real.groupby(key)[["imp", "exp", "imp_t1", "imp_t2", "exp_t1", "exp_t2"]].mean()


def _real_month_data(df: pd.DataFrame, year: int, month: int, prefer_year: int):
    """Real intervals of a calendar month, from the year closest to prefer_year."""
    sub = df[(df.index.month == month) & df["flag"].isin([FLAG_REAL, FLAG_INTERP])]
    if sub.empty:
        return None
    years = sorted(set(sub.index.year), key=lambda y: abs(y - prefer_year))
    return sub[sub.index.year == years[0]]


def build_profile_year(df: pd.DataFrame, rep: P1Report, cfg: dict) -> pd.DataFrame:
    """Return exactly 12 calendar months of 15-min data, filling missing months
    synthetically (section 5). Columns as load_p1 plus `flag`."""
    tz = cfg["p1"].get("timezone", "Europe/Amsterdam")
    real_mask = df["flag"].isin([FLAG_REAL, FLAG_INTERP]) & df["imp"].notna()
    if real_mask.sum() < 28 * 96:
        raise ValueError("Less than one month of real P1 data: cannot build a yearly profile.")

    last_real = df.index[real_mask][-1] + Q
    # Window ends at the last complete calendar month (a month whose data stops
    # less than an hour before its end counts as complete).
    probe = last_real + pd.Timedelta(hours=1)
    end = pd.Timestamp(year=probe.year, month=probe.month, day=1, tz=tz)
    start_y, start_m = end.year - 1, end.month
    months = _month_starts(pd.Timestamp(year=start_y, month=start_m, day=1, tz=tz), 12, tz)
    window_start, window_end = months[0], months[12]
    idx = pd.date_range(window_start, window_end, freq="15min", inclusive="left")
    rep.window = (window_start, window_end)

    cols = ["imp_t1", "imp_t2", "exp_t1", "exp_t2", "imp", "exp"]
    out = df.reindex(idx)[cols + ["is_low", "flag"]].copy()
    out["flag"] = out["flag"].fillna(FLAG_SYNTH)
    out.loc[out["imp"].isna() & (out["flag"] != FLAG_GAP), "flag"] = FLAG_SYNTH

    for k in range(12):
        ms, me = months[k], months[k + 1]
        sel = (out.index >= ms) & (out.index < me)
        n_total = sel.sum()
        n_real = (sel & out["flag"].isin([FLAG_REAL, FLAG_INTERP]).values).sum()
        if n_real >= 0.5 * n_total:
            # Mostly real: fill any holes from this month's own profile.
            holes = sel & out["imp"].isna().values
            if holes.any():
                prof = _profile(df[(df.index >= ms) & (df.index < me)])
                _fill_from_profile(out, holes, prof, FLAG_GAP)
            continue
        rep.missing_months.append(ms.strftime("%Y-%m"))
        _synthesize_month(out, df, ms, me, sel, tz)
        # Real data of the same calendar month in another year replaces synthetic days.
        other = df[(df.index.month == ms.month) & (df.index.year != ms.year)
                   & df["flag"].isin([FLAG_REAL, FLAG_INTERP])]
        if not other.empty:
            shifted = other.copy()
            shifted.index = [_same_wallclock(t, ms.year, tz) for t in other.index]
            shifted = shifted[~pd.Index(shifted.index).duplicated()]
            common = out.index.intersection(pd.DatetimeIndex(shifted.index))
            if len(common):
                out.loc[common, cols + ["is_low"]] = shifted.loc[common, cols + ["is_low"]].values
                out.loc[common, "flag"] = FLAG_REAL
                rep.partially_real_months.append(f"{ms:%Y-%m}: {len(common) / 96:.1f} day(s) of real data taken from {other.index[0].year}")

    out["is_low"] = np.where(out["is_low"].isna(), tariff_is_low(out.index, cfg), out["is_low"]).astype(bool)
    rep.low_confidence = len(rep.missing_months) > 3
    for c in cols:
        out[c] = out[c].astype(float).fillna(0.0)
    return out


def _same_wallclock(t: pd.Timestamp, year: int, tz: str):
    try:
        return pd.Timestamp(year=year, month=t.month, day=t.day, hour=t.hour, minute=t.minute).tz_localize(
            tz, ambiguous=True, nonexistent="shift_forward")
    except ValueError:
        return pd.NaT


def _fill_from_profile(out: pd.DataFrame, mask, prof: pd.DataFrame, flag: str):
    if prof.empty:
        return
    ix = out.index[mask]
    keys = list(zip(ix.weekday, ix.hour * 4 + ix.minute // 15))
    vals = prof.reindex(keys)
    for c in prof.columns:
        out.loc[ix, c] = vals[c].values
    out.loc[ix, "flag"] = flag


def _synthesize_month(out, df, ms, me, sel, tz):
    """Blend the last two weeks before and the first two weeks after the month."""
    prev_m = (ms - pd.Timedelta(days=1))
    before = _real_month_data(df, prev_m.year, prev_m.month, prev_m.year)
    after = _real_month_data(df, me.year, me.month, me.year)
    if before is not None:
        before = before[before.index.day > before.index.days_in_month.max() - 14]
    if after is not None:
        after = after[after.index.day <= 14]
    pb = _profile(before.assign(flag=FLAG_REAL)) if before is not None and len(before) else pd.DataFrame()
    pa = _profile(after.assign(flag=FLAG_REAL)) if after is not None and len(after) else pd.DataFrame()
    if pb.empty and pa.empty:
        pb = pa = _profile(df)
    elif pb.empty:
        pb = pa
    elif pa.empty:
        pa = pb
    ix = out.index[sel]
    keys = list(zip(ix.weekday, ix.hour * 4 + ix.minute // 15))
    vb, va = pb.reindex(keys).fillna(0.0), pa.reindex(keys).fillna(0.0)
    ndays = ix.days_in_month[0]
    w = np.asarray(((ix.day - 1) + (ix.hour * 4 + ix.minute // 15) / 96) / max(ndays, 1))
    for c in ["imp", "exp", "imp_t1", "imp_t2", "exp_t1", "exp_t2"]:
        out.loc[ix, c] = (1 - w) * vb[c].values + w * va[c].values
    out.loc[ix, "flag"] = FLAG_SYNTH
    out.loc[ix, "is_low"] = np.nan


def load_pv(source, cfg: dict) -> pd.Series:
    """Optional PV production, resampled to 15-min kWh."""
    pv = cfg.get("pv", {})
    raw = pd.read_csv(source)
    t = localize(raw[pv.get("time_column", "time")], cfg["p1"].get("timezone", "Europe/Amsterdam"))
    if pv.get("power_column"):
        s = pd.Series(pd.to_numeric(raw[pv["power_column"]], errors="coerce").values, index=t)
        return s.resample("15min").mean() / 1000 * 0.25
    s = pd.Series(pd.to_numeric(raw[pv.get("energy_column", "kWh")], errors="coerce").values, index=t)
    step = s.index.to_series().diff().median()
    if step > Q:
        n = int(step / Q)
        s = s.resample("15min").ffill() / n
    return s.resample("15min").sum()


def monthly_summary(profile: pd.DataFrame) -> pd.DataFrame:
    g = profile.groupby(profile.index.strftime("%Y-%m"))
    out = pd.DataFrame({
        "import_kwh": g["imp"].sum(),
        "export_kwh": g["exp"].sum(),
        "synthetic_share": g["flag"].apply(lambda s: float((s == FLAG_SYNTH).mean())),
    })
    return out
