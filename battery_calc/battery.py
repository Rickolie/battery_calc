"""Home battery dataset, per-interval physics and control strategies
(sections 3.2, 5 and 8)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Modes per interval
SELF, GRID_CHARGE, FULL_DISCHARGE, CHARGE_PV, ZERO_IMPORT, IDLE = range(6)
MODE_NAMES = {SELF: "self_consumption", GRID_CHARGE: "charge", FULL_DISCHARGE: "sell",
              CHARGE_PV: "charge_pv", ZERO_IMPORT: "zero_import", IDLE: "idle"}
SUB_MODES = {"self_consumption": SELF, "charge_pv": CHARGE_PV, "zero_import": ZERO_IMPORT, "idle": IDLE}

DT_H = 0.25


@dataclass
class Battery:
    id: str
    brand: str
    model: str
    type: str = ""
    phases: int = 1
    connection: str = ""
    nominal_kwh: float | None = None
    usable_kwh: float = 0.0
    dod: float = 1.0
    max_charge_w: float = 800.0
    max_discharge_w: float = 800.0
    rte: float = 0.85
    standby_w: float = 10.0
    cycle_life: float = 6000.0
    eol_capacity: float = 0.7
    warranty_years: float = 10.0
    warranty_mwh: float | None = None
    warranty_self_install: str = ""
    expandable: str = ""
    control_interface: str = ""
    needs_meter: str = ""
    price_nl: float | None = None
    price_nl_lowest: float | None = None
    price_nl_lowest_date: str = ""
    price_de: float | None = None
    extra_hardware: float = 0.0
    source: str = ""
    retrieved_at: str = ""
    verified: bool = False
    notes: str = ""
    estimated_fields: list = field(default_factory=list)
    missing_fields: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.brand} {self.model}"

    @property
    def estimated(self) -> bool:
        return bool(self.estimated_fields)

    def price_variants(self, de_travel: float = 0.0) -> dict:
        out = {}
        if self.price_nl is not None:
            out["NL current"] = self.price_nl + self.extra_hardware
        if self.price_nl_lowest is not None:
            out["NL lowest-ever"] = self.price_nl_lowest + self.extra_hardware
        if self.price_de is not None:
            out["DE 0% VAT (scenario)"] = self.price_de + self.extra_hardware + de_travel
        return out


def _num(v):
    try:
        if v is None or str(v).strip() == "" or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


def load_batteries(source, defaults: dict) -> list[Battery]:
    df = pd.read_csv(source, dtype=str).fillna("")
    out = []
    for _, r in df.iterrows():
        est, missing = [], []

        def get(col, dflt_key=None, scale=1.0):
            v = _num(r.get(col))
            if v is None:
                missing.append(col)
                if dflt_key is not None and dflt_key in defaults:
                    est.append(col)
                    return float(defaults[dflt_key])
                return None
            return v * scale

        nominal = _num(r.get("nominal_kwh"))
        dod = _num(r.get("dod"))
        if dod is not None and dod > 1:
            dod /= 100.0
        usable = _num(r.get("usable_kwh"))
        if usable is None:
            missing.append("usable_kwh")
            if dod is None:
                missing.append("dod")
                dod = float(defaults.get("dod", 1.0))
                est.append("dod")
            usable = (nominal or 0.0) * dod
        rte = _num(r.get("rte"))
        if rte is not None and rte > 1:
            rte /= 100.0
        if rte is None:
            missing.append("rte")
            est.append("rte")
            rte = float(defaults.get("rte", 0.85))
        eol = _num(r.get("eol_capacity"))
        if eol is not None and eol > 1:
            eol /= 100.0
        if eol is None:
            missing.append("eol_capacity")
            est.append("eol_capacity")
            eol = float(defaults.get("eol_capacity", 0.7))
        b = Battery(
            id=r["id"], brand=r.get("brand", ""), model=r.get("model", ""), type=r.get("type", ""),
            phases=int(_num(r.get("phases")) or 1), connection=r.get("connection", ""),
            nominal_kwh=nominal, usable_kwh=usable, dod=dod or 1.0,
            max_charge_w=get("max_charge_w") or 0.0, max_discharge_w=get("max_discharge_w") or 0.0,
            rte=rte, standby_w=get("standby_w", "standby_w"), cycle_life=get("cycle_life", "cycle_life"),
            eol_capacity=eol, warranty_years=get("warranty_years", "warranty_years"),
            warranty_mwh=_num(r.get("warranty_mwh")),
            warranty_self_install=r.get("warranty_self_install", ""),
            expandable=r.get("expandable_max_modules", ""), control_interface=r.get("control_interface", ""),
            needs_meter=r.get("needs_meter", ""),
            price_nl=_num(r.get("price_nl_incl_vat")), price_nl_lowest=_num(r.get("price_nl_lowest_incl_vat")),
            price_nl_lowest_date=r.get("price_nl_lowest_date", ""), price_de=_num(r.get("price_de_excl_vat")),
            extra_hardware=_num(r.get("extra_hardware_eur")) or 0.0,
            source=r.get("source_url", ""), retrieved_at=r.get("retrieved_at", ""),
            verified=str(r.get("verified", "")).lower() in ("yes", "true", "1"), notes=r.get("notes", ""),
        )
        for col in ("price_nl_incl_vat", "warranty_mwh"):
            if _num(r.get(col)) is None:
                missing.append(col)
        b.estimated_fields = sorted(set(est))
        b.missing_fields = sorted(set(missing))
        out.append(b)
    return out


def in_scope(b: Battery) -> tuple[bool, str]:
    v = (b.warranty_self_install or "").strip().lower()
    if v in ("no", "false", "0"):
        return False, "warranty void without certified installation"
    return True, "" if v else "self-install warranty unknown (kept, flagged)"


# ---------------------------------------------------------------------------
# Physics


@dataclass
class SimResult:
    imp: np.ndarray
    exp: np.ndarray
    charged_ac: float
    discharged_ac: float
    efc: float
    end_capacity: float
    soc: np.ndarray | None = None
    batt_ac: np.ndarray | None = None   # + charge, - discharge (AC side)


def simulate(net: np.ndarray, modes, b: Battery, power_cap_w: float, soc0: float = 0.0,
             min_soc_frac: float = 0.0, keep_trace: bool = False, rte: float | None = None,
             standby_w: float | None = None, degrade: bool = True) -> SimResult:
    """Step through every interval. `net` = house import − export (kWh) without
    battery; `modes` is a mode per interval or one mode for all."""
    rte = b.rte if rte is None else rte
    standby = (b.standby_w if standby_w is None else standby_w) or 0.0
    eta = math.sqrt(rte)
    usable = b.usable_kwh
    p_ch = min(b.max_charge_w, power_cap_w) / 1000.0 * DT_H
    p_dis = min(b.max_discharge_w, power_cap_w) / 1000.0 * DT_H
    sb = standby / 1000.0 * DT_H
    eol, life = b.eol_capacity, b.cycle_life or 1e12
    fade = (1.0 - eol) / life if degrade else 0.0
    lo = usable * min_soc_frac
    n = len(net)
    netl = (np.asarray(net, dtype=float) + sb).tolist()
    single = isinstance(modes, int)
    ml = None if single else list(modes)
    soc = min(max(soc0, lo), usable)
    cap = usable
    dis_dc_total = 0.0
    ch_ac_total = 0.0
    dis_ac_total = 0.0
    gl = [0.0] * n
    socs = [0.0] * n if keep_trace else None
    bat = [0.0] * n
    for t in range(n):
        L = netl[t]
        m = modes if single else ml[t]
        ac = 0.0
        if m == SELF:
            if L < 0:
                ac = min(-L, p_ch, (cap - soc) / eta)
            elif L > 0:
                ac = -min(L, p_dis, (soc - lo) * eta)
        elif m == GRID_CHARGE:
            ac = min(p_ch, (cap - soc) / eta)
        elif m == FULL_DISCHARGE:
            ac = -min(p_dis, (soc - lo) * eta)
        elif m == CHARGE_PV:
            if L < 0:
                ac = min(-L, p_ch, (cap - soc) / eta)
        elif m == ZERO_IMPORT:
            if L > 0:
                ac = -min(L, p_dis, (soc - lo) * eta)
        if ac > 0:
            soc += ac * eta
            ch_ac_total += ac
        elif ac < 0:
            dc = -ac / eta
            soc -= dc
            dis_dc_total += dc
            dis_ac_total -= ac
            if fade:
                cap = usable * max(eol, 1.0 - fade * dis_dc_total / usable)
                if soc > cap:
                    soc = cap
        if soc < lo:
            soc = lo
        gl[t] = L + ac
        bat[t] = ac
        if keep_trace:
            socs[t] = soc
    g = np.asarray(gl)
    return SimResult(np.maximum(g, 0.0), np.maximum(-g, 0.0), ch_ac_total, dis_ac_total,
                     dis_dc_total / usable if usable else 0.0, cap,
                     np.asarray(socs) if keep_trace else None, np.asarray(bat))


# ---------------------------------------------------------------------------
# Strategy plans


def plan_timed(index: pd.DatetimeIndex, windows: list[tuple[int, int, int]], base: int = SELF) -> np.ndarray:
    modes = np.full(len(index), base, dtype=np.int64)
    h = index.hour
    for start, end, mode in windows:
        if start <= end:
            sel = (h >= start) & (h < end)
        else:
            sel = (h >= start) | (h < end)
        modes[np.asarray(sel)] = mode
    return modes


def plan_dynamic(index: pd.DatetimeIndex, u: np.ndarray, s: np.ndarray, b: Battery, power_cap_w: float,
                 wear: float, scale: float = 1.0, sell: bool = False, cfg: dict | None = None,
                 rte: float | None = None, export: np.ndarray | None = None,
                 solar_forecast: np.ndarray | None = None, imports: np.ndarray | None = None) -> np.ndarray:
    """Per-day plan from the day's own day-ahead prices (known the day before):
    pair the cheapest intervals (charge) with the most expensive (discharge or
    sell) only while the spread clears the battery's break-even.

    Grid charging leaves room for expected solar surplus: by default a
    persistence forecast (yesterday's export), or `solar_forecast` (kWh per
    interval, e.g. from PV data) when given.

    With `imports` given, stored energy is reserved for the day's most
    expensive load: self-consumption discharge is only allowed in the
    priciest intervals whose forecast load (yesterday's import, a persistence
    forecast) adds up to the battery's energy; elsewhere the battery holds
    (still absorbing solar surplus)."""
    cfg = cfg or {}
    rte = b.rte if rte is None else rte
    eta = math.sqrt(rte)
    lo_q, hi_q = cfg.get("low_quantile", 0.25), cfg.get("high_quantile", 0.75)
    sub = cfg.get("sub_strategy", {}) or {}
    m_low = SUB_MODES.get(sub.get("low", "self_consumption"), SELF)
    m_neu = SUB_MODES.get(sub.get("neutral", "self_consumption"), SELF)
    m_high = SUB_MODES.get(sub.get("high", "self_consumption"), SELF)
    p_ch = min(b.max_charge_w, power_cap_w) / 1000.0 * DT_H
    p_dis = min(b.max_discharge_w, power_cap_w) / 1000.0 * DT_H
    n_ch = max(1, math.ceil(b.usable_kwh / max(p_ch * eta, 1e-9)))
    n_dis = max(1, math.ceil(b.usable_kwh * eta / max(p_dis, 1e-9)))
    w = wear * scale
    modes = np.full(len(index), m_neu, dtype=np.int64)
    day = np.asarray(index.normalize().asi8)
    bounds = np.flatnonzero(np.diff(day)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(index)]])
    prev_surplus = 0.0
    prev_load = None
    reserve = float(cfg.get("reserve_factor", 1.5))
    min_spread = float(cfg.get("reserve_min_spread", 0.01))
    for a, z in zip(starts, ends):
        uu, ss = u[a:z], s[a:z]
        if len(uu) == 0:
            continue
        if solar_forecast is not None:
            expected = float(np.nansum(solar_forecast[a:z]))
        else:
            expected = prev_surplus
        if export is not None:
            prev_surplus = float(np.sum(export[a:z]))
        room = max(0.0, b.usable_kwh - expected * eta)
        n_ch_day = math.ceil(room / max(p_ch * eta, 1e-9)) if room > 1e-9 else 0
        ql, qh = np.quantile(uu, [lo_q, hi_q])
        seg = np.full(z - a, m_neu)
        seg[uu <= ql] = m_low
        seg[uu >= qh] = m_high
        value = np.maximum(uu, ss) if sell else uu
        order_c = np.argsort(uu, kind="stable")
        order_d = np.argsort(-value, kind="stable")
        chosen_c, chosen_d = [], []
        for k in range(max(n_ch, n_dis)):
            c = order_c[min(k, len(order_c) - 1)]
            d = order_d[min(k, len(order_d) - 1)]
            if value[d] < uu[c] / rte + w:
                break
            if k < n_ch_day and c not in chosen_d:
                chosen_c.append(c)
            if k < n_dis and d not in chosen_c:
                chosen_d.append(d)
        if chosen_d:
            last_d = max(chosen_d)
            basis = (np.mean(uu[chosen_c]) if chosen_c else float(np.min(uu))) / rte + w
            for c in chosen_c:
                if c < last_d:
                    seg[c] = GRID_CHARGE
            for d in chosen_d:
                # Without selling, a planned discharge covers the load and still
                # absorbs any solar surplus (self-consumption behaviour).
                seg[d] = FULL_DISCHARGE if (sell and ss[d] >= basis) else SELF
        if imports is not None:
            today = np.asarray(imports[a:z], dtype=float)
            sunny = expected * eta >= float(cfg.get("reserve_skip_solar_frac", 1.0)) * b.usable_kwh
            if prev_load is not None and reserve > 0 and not sunny and uu.max() - uu.min() >= min_spread:
                fc = np.resize(prev_load, len(uu)) if len(prev_load) != len(uu) else prev_load
                order = np.argsort(-uu, kind="stable")
                cum = np.cumsum(fc[order])
                k = int(np.searchsorted(cum, reserve * b.usable_kwh * eta)) + 1
                allowed = np.zeros(len(uu), bool)
                allowed[order[:k]] = True
                hold = (seg == SELF) & ~allowed
                seg[hold] = CHARGE_PV
            prev_load = today
        modes[a:z] = seg
    return modes


def perfect_foresight(net: np.ndarray, u: np.ndarray, s: np.ndarray, b: Battery, power_cap_w: float,
                      wear: float = 0.0, levels: int = 21) -> SimResult:
    """Upper bound on bill savings: all prices and flows known in advance.
    Solved exactly as a linear programme (HiGHS); falls back to a discretised
    dynamic programme when scipy is unavailable. No degradation inside the year.
    `wear` defaults to 0 so the result bounds the bill saving of every strategy."""
    try:
        return _perfect_lp(net, u, s, b, power_cap_w, wear)
    except ImportError:
        return _perfect_dp(net, u, s, b, power_cap_w, wear, levels)


def _perfect_lp(net, u, s, b: Battery, power_cap_w: float, wear: float,
                window: int = 96 * 7, lookahead: int = 96) -> SimResult:
    """Weekly windows with one day of look-ahead, carrying the state of charge
    (≈3× faster than one year-long LP, within a fraction of a percent of it)."""
    import scipy.optimize  # noqa: F401  (ImportError -> DP fallback)
    net, u, s = (np.asarray(a, dtype=float) for a in (net, u, s))
    n = len(net)
    c_all, d_all, soc_all = np.zeros(n), np.zeros(n), np.zeros(n)
    soc0 = 0.0
    for a in range(0, n, window):
        z = min(a + window + lookahead, n)
        keep = min(window, n - a)
        c, d, soc = _lp_window(net[a:z], u[a:z], s[a:z], b, power_cap_w, wear, soc0)
        c_all[a:a + keep], d_all[a:a + keep], soc_all[a:a + keep] = c[:keep], d[:keep], soc[:keep]
        soc0 = soc[keep - 1]
    eta = math.sqrt(b.rte)
    sb = (b.standby_w or 0.0) / 1000.0 * DT_H
    g = net + sb + c_all - d_all
    cap = b.usable_kwh
    return SimResult(np.maximum(g, 0), np.maximum(-g, 0), float(c_all.sum()), float(d_all.sum()),
                     d_all.sum() / eta / cap if cap else 0.0, cap, soc_all, c_all - d_all)


def _lp_window(net, u, s, b: Battery, power_cap_w: float, wear: float, soc0: float):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    eta = math.sqrt(b.rte)
    p_ch = min(b.max_charge_w, power_cap_w) / 1000.0 * DT_H
    p_dis = min(b.max_discharge_w, power_cap_w) / 1000.0 * DT_H
    cap = b.usable_kwh
    sb = (b.standby_w or 0.0) / 1000.0 * DT_H
    L = np.asarray(net, dtype=float) + sb
    n = len(L)
    u = np.asarray(u, dtype=float)
    s = np.minimum(np.asarray(s, dtype=float), u)   # no simultaneous import/export arbitrage
    # variables per interval: c, d, imp, exp, soc
    C, D, I, X, S = 0, 1, 2, 3, 4
    idx = lambda k: np.arange(n) * 5 + k
    cost = np.zeros(5 * n)
    cost[idx(I)] = u
    cost[idx(X)] = -s
    cost[idx(D)] = wear + 1e-6       # tiny tie-breaker against pointless cycling
    t = np.arange(n)
    rows = np.concatenate([t, t, t, t,                       # balance rows 0..n-1
                           n + t, n + t, n + t, n + t[1:]])  # soc rows n..2n-1
    cols = np.concatenate([idx(I), idx(X), idx(C), idx(D),
                           idx(S), idx(C), idx(D), idx(S)[:-1]])
    vals = np.concatenate([np.ones(n), -np.ones(n), -np.ones(n), np.ones(n),
                           np.ones(n), -eta * np.ones(n), np.ones(n) / eta, -np.ones(n - 1)])
    A = coo_matrix((vals, (rows, cols)), shape=(2 * n, 5 * n)).tocsr()
    rhs = np.concatenate([L, np.zeros(n)])
    rhs[n] = soc0
    lb = np.zeros(5 * n)
    ub = np.full(5 * n, np.inf)
    ub[idx(C)] = p_ch
    ub[idx(D)] = p_dis
    ub[idx(S)] = cap
    res = linprog(cost, A_eq=A, b_eq=rhs, bounds=np.column_stack([lb, ub]), method="highs")
    if not res.success:
        raise RuntimeError(f"perfect-foresight LP failed: {res.message}")
    x = res.x
    return x[idx(C)], x[idx(D)], x[idx(S)]


def _perfect_dp(net: np.ndarray, u: np.ndarray, s: np.ndarray, b: Battery, power_cap_w: float,
                      wear: float, levels: int = 21) -> SimResult:
    """Fallback: dynamic programming over a discretised state of charge."""
    eta = math.sqrt(b.rte)
    p_ch = min(b.max_charge_w, power_cap_w) / 1000.0 * DT_H
    p_dis = min(b.max_discharge_w, power_cap_w) / 1000.0 * DT_H
    cap = b.usable_kwh
    step_needed = min(p_ch * eta, p_dis / eta) / 2.0
    K = int(max(levels, min(121, math.ceil(cap / max(step_needed, 1e-9)) + 1)))
    grid = np.linspace(0.0, cap, K)
    d = grid[None, :] - grid[:, None]               # DC change from level a to b
    ac = np.where(d > 0, d / eta, d * eta)          # AC flow, + charge
    feasible = (ac <= p_ch + 1e-9) & (-ac <= p_dis + 1e-9)
    sb = (b.standby_w or 0.0) / 1000.0 * DT_H
    L = np.asarray(net, dtype=float) + sb
    n = len(L)
    V = np.zeros(K)
    policy = np.zeros((n, K), dtype=np.int16)
    wear_m = wear * np.maximum(-ac, 0.0)
    big = 1e9
    for t in range(n - 1, -1, -1):
        g = L[t] + ac
        cost = np.where(g > 0, u[t] * g, s[t] * g) + wear_m
        cost = np.where(feasible, cost, big) + V[None, :]
        bi = np.argmin(cost, axis=1)
        policy[t] = bi
        V = cost[np.arange(K), bi]
    k = 0
    flows = np.empty(n)
    bat = np.empty(n)
    ch = dis = 0.0
    for t in range(n):
        nb = policy[t, k]
        a = ac[k, nb]
        flows[t] = L[t] + a
        bat[t] = a
        if a > 0:
            ch += a
        else:
            dis -= a
        k = nb
    efc = dis / eta / cap if cap else 0.0
    return SimResult(np.maximum(flows, 0), np.maximum(-flows, 0), ch, dis, efc, cap, None, bat)
