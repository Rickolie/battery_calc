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
BEST_PRICE = "Cheapest NL/DE"      # price variant: the lower of today's NL and DE price (German prices on)


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
    backup_w: float | None = None            # backup socket power during an outage (W)
    outage_solar: str = ""                   # can solar keep charging it when the grid is off? "" = unknown
    dc_solar_w: float | None = None          # DC solar (MPPT) input on the battery itself (W)
    price_bf_nl: float | None = None         # Black Friday deal NL, incl. VAT
    price_bf_de: float | None = None         # Black Friday deal DE, 0% VAT
    price_nl_ref: float | None = None        # last normal NL price before the Black Friday window
    price_de_ref: float | None = None
    bf_nl_info: str = ""                     # date and shop of the deal
    bf_de_info: str = ""
    estimated_fields: list = field(default_factory=list)
    missing_fields: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.brand} {self.model}"

    @property
    def estimated(self) -> bool:
        return bool(self.estimated_fields)

    def best_price(self, de_travel: float = 0.0) -> tuple[float | None, str]:
        """Today's lower price of the Dutch shops (incl. VAT) and the German manufacturer
        shop (0% VAT for home solar storage, plus travel/shipping): (price, "NL"/"DE")."""
        nl = None if self.price_nl is None else self.price_nl + self.extra_hardware
        de = None if self.price_de is None else self.price_de + self.extra_hardware + de_travel
        if de is not None and (nl is None or de < nl):
            return de, "DE"
        return nl, "NL" if nl is not None else ""

    def price_variants(self, de_travel: float = 0.0, bf: dict | None = None, best: bool = False) -> dict:
        """Purchase price per scenario. Black Friday prices use the real deal when
        known, plus a configured discount on the current price ("est.") unless
        `bf["estimates"]` is false. With `best`, the first variant is the cheaper of
        today's NL and DE price (BEST_PRICE)."""
        out = {}
        if best:
            p, _ = self.best_price(de_travel)
            if p is not None:
                out[BEST_PRICE] = p
        if self.price_nl is not None:
            out["NL current"] = self.price_nl + self.extra_hardware
        if self.price_nl_lowest is not None:
            out["NL lowest-ever"] = self.price_nl_lowest + self.extra_hardware
        if self.price_de is not None:
            out["DE 0% VAT (scenario)"] = self.price_de + self.extra_hardware + de_travel
        if bf is not None and bf.get("enabled", True):
            vat = float(bf.get("vat", 0.21))
            # Real deals found by the scraper during the Black Friday window ...
            if self.price_bf_nl is not None:
                out["NL Black Friday"] = self.price_bf_nl + self.extra_hardware
            if self.price_bf_de is not None:
                out["DE Black Friday"] = self.price_bf_de + self.extra_hardware + de_travel
            # ... and, unless switched off, the estimate (configured discount on today's price).
            if not bf.get("estimates", True):
                return out
            if self.price_nl is not None:
                out["NL Black Friday (est.)"] = self.price_nl * (1 - float(bf.get("discount_nl", 0.15))) + self.extra_hardware
            base = self.price_de if self.price_de is not None else (
                self.price_nl / (1 + vat) if self.price_nl is not None else None)
            if base is not None:
                out["DE Black Friday (est.)"] = (base * (1 - float(bf.get("discount_de", 0.15)))
                                                 + self.extra_hardware + de_travel)
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
            backup_w=_num(r.get("backup_socket_w")), outage_solar=r.get("outage_solar", ""),
            dc_solar_w=_num(r.get("dc_solar_input_w")),
            price_bf_nl=_num(r.get("price_blackfriday_nl_incl_vat")),
            price_bf_de=_num(r.get("price_blackfriday_de_excl_vat")),
            price_nl_ref=_num(r.get("price_nl_ref_incl_vat")), price_de_ref=_num(r.get("price_de_ref_excl_vat")),
            bf_nl_info=" ".join(x for x in (r.get("price_blackfriday_nl_date", ""), r.get("price_blackfriday_nl_source", "")) if x),
            bf_de_info=" ".join(x for x in (r.get("price_blackfriday_de_date", ""), r.get("price_blackfriday_de_source", "")) if x),
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
    curtailed: np.ndarray | None = None  # solar export switched off (kWh per interval)


def curtail(res: SimResult, s: np.ndarray, below: float = 0.0) -> SimResult:
    """Zero export: whatever the house still exports after the battery has taken what
    it can (battery full or at its power limit) is curtailed – the solar inverter scales
    back – in every interval where exporting is worth less than `below` €/kWh
    (default: negative-value export). The curtailed energy is simply not produced."""
    from dataclasses import replace
    cut = np.where(np.asarray(s, dtype=float) < below, res.exp, 0.0)
    return replace(res, exp=res.exp - cut, curtailed=cut)


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


HBC_MODES = {"charge": GRID_CHARGE, "charge_pv": CHARGE_PV, "self_consumption": SELF, "sell": FULL_DISCHARGE,
             "full_stop": IDLE, "zero_import": ZERO_IMPORT}


def plan_hbc(index: pd.DatetimeIndex, price: np.ndarray, cfg: dict) -> np.ndarray:
    """Home Battery Control's Dynamic strategy ("Extreme-Pair Matching",
    flow 02 strategy-dynamic-2.json), reproduced so its results are achievable
    as-is: per local day, pair the cheapest with the most expensive interval
    while the spread >= min_delta (respecting the per-day hour caps), mark them
    low/high and apply the configured sub-strategy per mark. Pairs by price only,
    not by time, exactly like the original."""
    min_delta = float(cfg.get("min_delta", 0.06))
    per_h = 4
    cap_lo = int(round(float(cfg.get("cheapest_hrs", 0)) * per_h))
    cap_hi = int(round(float(cfg.get("expensive_hrs", 0)) * per_h))
    m_low = HBC_MODES[cfg.get("low", "charge")]
    m_neu = HBC_MODES[cfg.get("neutral", "charge_pv")]
    m_high = HBC_MODES[cfg.get("high", "self_consumption")]
    modes = np.full(len(index), m_neu, dtype=np.int64)
    day = np.asarray(index.normalize().asi8)
    bounds = np.flatnonzero(np.diff(day)) + 1
    for a, z in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(index)]])):
        p = price[a:z]
        order = np.argsort(p, kind="stable")
        lo, hi = 0, len(order) - 1
        n_lo = n_hi = 0
        seg = modes[a:z]
        while lo < hi:
            if p[order[hi]] - p[order[lo]] < min_delta:
                break
            full_lo = cap_lo and n_lo >= cap_lo
            full_hi = cap_hi and n_hi >= cap_hi
            if full_lo and full_hi:
                break
            if not full_lo:
                seg[order[lo]] = m_low
                n_lo += 1
            if not full_hi:
                seg[order[hi]] = m_high
                n_hi += 1
            lo += 1
            hi -= 1
    return modes


def perfect_foresight(net: np.ndarray, u: np.ndarray, s: np.ndarray, b: Battery, power_cap_w: float,
                      wear: float = 0.0, levels: int = 21, grid_charge: bool = True, sell: bool = True) -> SimResult:
    """Upper bound on bill savings: all prices and flows known in advance.
    Solved exactly as a linear programme (HiGHS); falls back to a discretised
    dynamic programme when scipy is unavailable. No degradation inside the year.
    `wear` defaults to 0 so the result bounds the bill saving of every strategy.
    `grid_charge=False` / `sell=False` restrict the battery to solar surplus /
    the house's own load (used to show where the upper bound's value comes from)."""
    if not (grid_charge and sell):
        return _perfect_lp(net, u, s, b, power_cap_w, wear, grid_charge=grid_charge, sell=sell)
    try:
        return _perfect_lp(net, u, s, b, power_cap_w, wear)
    except ImportError:
        return _perfect_dp(net, u, s, b, power_cap_w, wear, levels)


def _perfect_lp(net, u, s, b: Battery, power_cap_w: float, wear: float,
                window: int = 96 * 7, lookahead: int = 96, grid_charge: bool = True, sell: bool = True) -> SimResult:
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
        c, d, soc = _lp_window(net[a:z], u[a:z], s[a:z], b, power_cap_w, wear, soc0, grid_charge, sell)
        c_all[a:a + keep], d_all[a:a + keep], soc_all[a:a + keep] = c[:keep], d[:keep], soc[:keep]
        soc0 = soc[keep - 1]
    eta = math.sqrt(b.rte)
    sb = (b.standby_w or 0.0) / 1000.0 * DT_H
    g = net + sb + c_all - d_all
    cap = b.usable_kwh
    return SimResult(np.maximum(g, 0), np.maximum(-g, 0), float(c_all.sum()), float(d_all.sum()),
                     d_all.sum() / eta / cap if cap else 0.0, cap, soc_all, c_all - d_all)


def _lp_window(net, u, s, b: Battery, power_cap_w: float, wear: float, soc0: float,
               grid_charge: bool = True, sell: bool = True, steps: int = 1):
    """`steps` = quarter-hours per LP interval (4 plans on an hourly grid)."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    eta = math.sqrt(b.rte)
    p_ch = min(b.max_charge_w, power_cap_w) / 1000.0 * DT_H * steps
    p_dis = min(b.max_discharge_w, power_cap_w) / 1000.0 * DT_H * steps
    cap = b.usable_kwh
    sb = (b.standby_w or 0.0) / 1000.0 * DT_H * steps
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
    if not grid_charge:
        ub[idx(C)] = np.minimum(p_ch, np.maximum(-L, 0.0))
    if not sell:
        ub[idx(D)] = np.minimum(p_dis, np.maximum(L, 0.0))
    res = linprog(cost, A_eq=A, b_eq=rhs, bounds=np.column_stack([lb, ub]), method="highs")
    if not res.success:
        raise RuntimeError(f"perfect-foresight LP failed: {res.message}")
    x = res.x
    return x[idx(C)], x[idx(D)], x[idx(S)]


def plan_forecast(index: pd.DatetimeIndex, net: np.ndarray, u: np.ndarray, s: np.ndarray, b: Battery,
                  power_cap_w: float, wear: float, cfg: dict | None = None,
                  forecast: np.ndarray | None = None) -> np.ndarray:
    """Forecast optimiser (model-predictive control, like EMHASS in Home Assistant).

    Every day at `replan_hour` (13:00, when tomorrow's day-ahead prices are
    published) solve the same linear programme as perfect foresight, but over
    the prices actually known then (until the end of tomorrow) and a forecast
    of the house's net load: the mean of the last `forecast_days` days at the
    same quarter-hour (persistence). The plan is executed until the next
    replan as battery modes, so the real load and solar decide the flows:
    - planned grid charge → charge; planned discharge beyond the load → sell;
    - load the plan leaves uncovered → hold (still absorbs solar surplus);
    - solar surplus the plan does not store → discharge-only (wait for a
      cheaper export hour to absorb solar, e.g. negative midday prices);
    - otherwise self-consumption.
    Wear enters the plan as `wear_scale` × wear per kWh discharged. The plan
    uses `plan_minutes` steps (hourly by default: as good on the forecast's
    accuracy, and 3–4× faster than quarter-hours).
    `forecast` replaces the persistence forecast (e.g. the real net load, to
    measure what a perfect usage/solar forecast would be worth)."""
    cfg = cfg or {}
    days = int(cfg.get("forecast_days", 3))
    tol = float(cfg.get("tolerance_kwh", 0.02))
    hour = int(cfg.get("replan_hour", 13))
    w = wear * float(cfg.get("wear_scale", 0.5))
    steps = max(1, int(cfg.get("plan_minutes", 60)) // 15)
    net = np.asarray(net, dtype=float)
    u, s = np.asarray(u, dtype=float), np.asarray(s, dtype=float)
    n = len(net)
    modes = np.full(n, SELF, dtype=np.int64)
    if n == 0:
        return modes
    day = np.asarray(index.normalize().asi8)
    day_end = np.concatenate([np.flatnonzero(np.diff(day)) + 1, [n]])    # first index of the next day
    nxt_end = lambda t: day_end[np.searchsorted(day_end, t, side="right")]  # noqa: E731
    h, mi = np.asarray(index.hour), np.asarray(index.minute)
    starts = [0] + [int(q) for q in np.flatnonzero((h == hour) & (mi == 0)) if q > 0]
    soc = 0.0
    for i, a in enumerate(starts):
        z_exec = starts[i + 1] if i + 1 < len(starts) else n
        z = int(nxt_end(a))                                  # end of today
        if h[a] >= hour and z < n:
            z = int(day_end[np.searchsorted(day_end, z, side="right")])  # tomorrow's prices are known
        z = max(z, z_exec)
        f = np.zeros(z - a)
        k_used = 0
        for j in range(1, days + 1):
            if a - 96 * j < 0:
                break
            src = np.arange(a, z) - 96 * j
            f += net[np.minimum(src, n - 1)]
            k_used += 1
        if k_used:
            f /= k_used
        if forecast is not None:
            f = np.asarray(forecast[a:z], dtype=float)
        k = steps
        pad = (-len(f)) % k
        blk = lambda x, red: red(np.concatenate([x, np.repeat(x[-1:], pad)]).reshape(-1, k), axis=1)  # noqa: E731
        fb, ub, sbk = blk(f, np.sum), blk(u[a:z], np.mean), blk(s[a:z], np.mean)
        cc, dd, _ = _lp_window(fb, ub, sbk, b, power_cap_w, w, soc, steps=k)
        m_len = z_exec - a
        nb = -(-m_len // k)
        L, ch, dis = fb[:nb], cc[:nb], dd[:nb]
        sur, load = np.maximum(-L, 0.0), np.maximum(L, 0.0)
        tl = tol * k
        hold = (load > tl) & (dis < tl / 2)
        no_store = (sur > tl) & (ch < tl / 2)
        seg = np.full(nb, SELF, dtype=np.int64)
        seg[hold & ~no_store] = CHARGE_PV
        seg[no_store & ~hold] = ZERO_IMPORT
        seg[hold & no_store] = IDLE
        seg[ch > sur + tl] = GRID_CHARGE
        seg[dis > load + tl] = FULL_DISCHARGE
        seg = np.repeat(seg, k)[:m_len]
        modes[a:z_exec] = seg
        r = simulate(net[a:z_exec], seg, b, power_cap_w, soc0=soc, keep_trace=True, degrade=False)
        soc = float(r.soc[-1])
    return modes


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
