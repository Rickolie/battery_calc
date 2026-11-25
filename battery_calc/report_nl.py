"""Simplified Dutch report page for a household (e.g. Peter, or Rick's own data).

Built from a finished Analysis run: the same numbers as the full report, but only the
sections a household needs, in Dutch, and per section one "Meer details" dropdown that
holds every table and explanation. The battery runs on its standard control
(self-consumption from its own app) on every contract; on the dynamic contract the page
also shows smart control on hourly prices (the `forecast` strategy) as an extra option.

Household-specific sentences (where the contract came from, why it gets dearer, notes on the
batteries) come from the profile's `report.notes` (see peter/profile.yaml); everything else is
computed from the analysis.
"""
from __future__ import annotations

import html
import json
import math

import numpy as np
import pandas as pd

from .report import CSS, _static, chart_html

CONTRACT_COLORS = {"current": "#4a3aa7", "fixed": "#eb6834", "dynamic": "#2a78d6", "smart": "#1baf7a"}
STANDARD, SMART = "self_consumption", "forecast"
MAANDEN = ["jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]
APPARATEN = {
    "Airco (split unit, cooling)": "Airco (split-unit, koelen)",
    "Heat pump (air-water)": "Warmtepomp (lucht-water)",
    "Heat pump + 3 kW backup heater": "Warmtepomp + 3 kW bijverwarming",
    "Car charger 1-phase 16 A": "Laadpaal 1-fase 16 A",
    "Car charger 1-phase 32 A": "Laadpaal 1-fase 32 A",
    "Car charger 3-phase 16 A": "Laadpaal 3-fase 16 A",
}
OORDEEL = {
    "fits": "past",
    "fits, close to the limit at peaks – a load-managed (smart) device is safer":
        "past, maar krap op piekmomenten – een slim apparaat met belastingsturing is veiliger",
    "does not fit without load management (or a heavier connection)":
        "past niet zonder belastingsturing (of een zwaardere aansluiting)",
    "needs a 3-phase connection": "vraagt een 3-fase aansluiting",
}


# ------------------------------------------------------------------ formatting
def num(v, d=0) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–" if not (isinstance(v, float) and math.isinf(v)) else "nooit"
    s = f"{v:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def eur(v, d=0) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else ("€ " + num(v, d))


def esc(s) -> str:
    return html.escape(str(s))


def contract_nl(label: str) -> str:
    return (label.replace("Huidig contract – ", "").replace("(fixed)", "(vast)")
            .replace("Current contract – fixed", "Huidig vast contract"))


def table(rows: list[list], head: list[str], align_left: int = 1) -> str:
    out = ["<div class='tbl'><table><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td{' style=text-align:left' if i < align_left else ''}>{esc(c)}</td>"
                                    for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def bullets(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def chart(spec: dict) -> str:
    return chart_html(spec, lang="nl")


def section(num_: int | None, title: str, finding: str, body: str, details: str) -> str:
    h = f"{num_}. {title}" if num_ else title
    return (f"<section><h2>{esc(h)}</h2><p class='keyfind'>{esc(finding)}</p>{body}"
            f"<details class='more'><summary>Meer details</summary>{details}</details></section>")


# ------------------------------------------------------------------ the page
def render(an, meta: dict) -> str:
    T = {s.id: s for s in an.sections}
    N = meta.get("notes") or {}
    cur, fx, dy = an.current, an.cheapest_fixed(), an.cheapest_dynamic()
    roles = [(k, c) for k, c in (("current", cur), ("fixed", fx), ("dynamic", dy)) if c is not None]
    role_name = {"current": N.get("current_role", "Huidig contract houden"), "fixed": "Goedkoopste vaste aanbieding",
                 "dynamic": "Goedkoopste dynamische contract", "smart": "Dynamisch met slimme sturing"}
    comp, broles = battery_rows(an, roles)
    focus = an.focus
    parts = [advies(an, broles, role_name, comp, focus, N),
             verbruik(an, T), vermogen(an, T), huidig(an, T, cur, N),
             contracten(an, T, roles, role_name), batterijen(an, T, focus, N),
             terugverdientijd(an, broles, role_name, comp, focus, N), door_het_jaar(an, T)]
    head = (f"<h1 style='font-size:1.5rem'>Energiecontract & thuisbatterij – {esc(meta.get('label', ''))}</h1>"
            f"<p class='keyfind'>Berekend {esc(meta.get('run_at', ''))} · aansluiting {esc(an.conn.label)} "
            f"({num(an.conn.phase_kw * an.conn.phases, 2)} kW) · meterdata {esc(meta.get('period', ''))}</p>")
    return (f"<!doctype html><html lang='nl'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>Energie & thuisbatterij – {esc(meta.get('label', ''))}"
            f"</title><style>{CSS}{_static('uPlot.min.css')}section{{margin:18px 0 26px}}"
            f"h2{{margin-bottom:2px}}</style><script>{_static('uPlot.iife.min.js')}</script>"
            f"<script>{_static('charts.js')}</script></head><body>{head}{''.join(parts)}</body></html>")


def battery_rows(an, roles):
    """Payback rows per role: every contract with the standard control, plus the dynamic
    contract with smart control ("smart") when that strategy ran. Returns (rows, roles with
    batteries); rows carry `_role` so the same contract can appear twice."""
    rk = an.payback_summary["ranked"]
    parts, broles = [], list(roles)
    std = an.payback_compact(rk[rk.strategy == STANDARD])
    for k, c in roles:
        parts.append(std[std._cid == c.id].assign(_role=k))
    dy = next((c for k, c in roles if k == "dynamic"), None)
    sm = rk[(rk.strategy == SMART) & (rk._cid == dy.id)] if dy is not None else rk.iloc[:0]
    if len(sm):
        parts.append(an.payback_compact(sm).assign(_role="smart"))
        broles.append(("smart", dy))
    comp = pd.concat(parts, ignore_index=True)
    keep = {f["b"].id for f in (an.focus or [])}        # one battery per size class
    if keep:
        comp = comp[comp._bid.isin(keep)]
    return comp.sort_values(["yearly_cost_with_battery_2027", "payback_years"]).reset_index(drop=True), broles


SMART_UITLEG = ("Slimme sturing (alleen op een dynamisch contract) = elke middag, als de uurprijzen voor morgen "
                "bekend zijn, een plan maken met een verwachting van zonnestroom en verbruik (gemiddelde van de "
                "afgelopen 3 dagen): laden met zonnestroom en in goedkope uren van het net, ontladen of terugleveren "
                "in dure uren (vooral de avondpiek). Dit vraagt extra software, bijvoorbeeld Home Assistant met "
                "Home Battery Control; de app van de batterij doet dit meestal niet standaard. Gerekend zonder de zonnepanelen uit "
                "te schakelen bij negatieve prijzen.")


def advies(an, roles, role_name, comp, focus, N) -> str:
    sizes = [(f["b"], f["cls"]) for f in focus]
    rows, cands = [], []
    for k, c in roles:
        base = an.cost27(c)
        cands.append((base, k, c, None, None))
        r = [role_name[k] + f" ({contract_nl(c.label)})", eur(base)]
        for b, _ in sizes:
            x = comp[(comp._bid == b.id) & (comp._role == k)]
            v = float(x.yearly_cost_with_battery_2027.iloc[0]) if len(x) else np.nan
            r.append(eur(v))
            if not np.isnan(v):
                cands.append((v, k, c, b, x.iloc[0]))
        rows.append(r)
    best = min(cands, key=lambda z: z[0])
    head = ["Contract", "Zonder batterij"] + [f"Met {num(b.nominal_kwh, 2)} kWh" for b, _ in sizes]
    keep = an.cost27(an.current) if an.current else None
    now = (an.expected_by_contract.get(an.current.id, {}) or {}).get("saldering") if an.current else None
    cur_best = None
    if an.current is not None:
        cc = comp[comp._role == "current"].sort_values("yearly_cost_with_battery_2027")
        cur_best = cc.iloc[0] if len(cc) else None
    v, k, c, b, x = best
    items = []
    if keep is not None and now is not None:
        items.append(f"<b>Als u niets verandert</b> kost stroom u vanaf 2027 ongeveer <b>{eur(keep)} per jaar</b> "
                     f"(nu, met salderen, {eur(now)}): " + N.get("why_dearer_short", "de saldering stopt.") +
                     "")
    if b is None:
        items.append(f"<b>Goedkoopst:</b> {esc(role_name[k].lower())} ({esc(contract_nl(c.label))}) zonder batterij, "
                     f"ongeveer {eur(v)} per jaar.")
    else:
        items.append(f"<b>Goedkoopst:</b> {esc(contract_nl(c.label))} met de {esc(b.name)} (€ {num(x.price_eur)})"
                     f"{' en slimme sturing' if k == 'smart' else ''}: "
                     f"ongeveer <b>{eur(v)} per jaar</b>, de batterij is terugverdiend in {num(x.payback_years, 1)} jaar.")
    if cur_best is not None:
        items.append(f"<b>Wilt u uw huidige contract houden?</b> Dan bespaart de {esc(cur_best.battery)} het meest: "
                     f"{eur(cur_best.saving_eur_year_2027)} per jaar, terugverdiend in "
                     f"{num(cur_best.payback_years, 1)} jaar; u betaalt dan ongeveer "
                     f"{eur(cur_best.yearly_cost_with_battery_2027)} per jaar.")
    items.append("Alle bedragen: jaarkosten vanaf 2027 inclusief btw, netbeheer en vermindering energiebelasting, "
                 "zonder eenmalige welkomstbonus. De batterij werkt met de <b>standaardinstelling van de "
                 "batterij-app (zelfverbruik)</b>; op het dynamische contract staat ook <b>slimme sturing</b> als "
                 "extra optie (zie hoofdstuk 6).")
    spec = {"kind": "bars", "unit": "€", "yLabel": "€ per jaar",
            "title": "Jaarkosten vanaf 2027 per contract, met en zonder batterij",
            "subtitle": "Lager is beter. Batterij met de standaardinstelling (zelfverbruik), op dynamisch ook met "
                        "slimme sturing.",
            "categories": ["Geen batterij"] + [f"{num(b.nominal_kwh, 2)} kWh" for b, _ in sizes],
            "series": []}
    for k2, c2 in roles:
        vals = [round(an.cost27(c2) or 0)]
        for b2, _ in sizes:
            x2 = comp[(comp._bid == b2.id) & (comp._role == k2)]
            vals.append(round(float(x2.yearly_cost_with_battery_2027.iloc[0])) if len(x2) else 0)
        spec["series"].append({"label": f"{role_name[k2]} ({contract_nl(c2.label)})", "color": CONTRACT_COLORS[k2],
                               "values": vals})
    details = (table(rows, head) +
               "<h3>Aannames</h3>" + bullets([
                   *([N["contract_source"]] if N.get("contract_source") else []),
                   "Dynamische contracten zijn doorgerekend met de werkelijke uurprijzen van 2023–2025 op uw "
                   "verbruik; vaste contracten met hun vaste tarieven.",
                   "Standaardinstelling = de batterij laadt met zonnestroom die u anders teruglevert en ontlaadt "
                   "als u stroom afneemt.",
                   SMART_UITLEG,
                   f"Batterijprijzen van vandaag (Black Friday): {', '.join(f'{esc(b.name)} € {num(an.battery_price(b))}' for b, _ in sizes)}."]))
    finding = (f"Goedkoopst vanaf 2027: {contract_nl(c.label)}" + (f" met {b.name}" if b is not None else "")
               + f", ongeveer {eur(v)} per jaar.")
    return section(None, "Advies", finding, bullets(items) + chart(spec), details)


def verbruik(an, T) -> str:
    m = T["data"].tables["monthly_flows"]
    labels = [f"{MAANDEN[int(i[5:7]) - 1]} {i[2:4]}" for i in m.index.astype(str)]
    yr = 12 / max(len(m), 1)              # meter data can cover more or less than a year
    imp, exp = float(m.import_kwh.sum()) * yr, float(m.export_kwh.sum()) * yr
    spec = {"kind": "bars", "unit": "kWh", "yLabel": "kWh per maand", "title": "Afname en teruglevering per maand",
            "categories": labels,
            "series": [{"label": "Afname van het net", "color": "#2a78d6", "values": [round(v) for v in m.import_kwh]},
                       {"label": "Teruglevering (zonnestroom)", "color": "#eda100", "values": [round(v) for v in m.export_kwh]}]}
    body = bullets([f"Afname van het net: <b>{num(imp)} kWh</b> per jaar.",
                    f"Teruglevering van zonnestroom: <b>{num(exp)} kWh</b> per jaar, vooral van maart tot oktober.",
                    "In de zomer levert u meer terug dan u afneemt; in de winter andersom. Een batterij kan alleen "
                    "zonnestroom van dezelfde dag of nacht verschuiven, niet van zomer naar winter."]) + chart(spec)
    rows = [[lab, num(a), num(b)] for lab, a, b in zip(labels, m.import_kwh, m.export_kwh)]
    details = (table(rows, ["Maand", "Afname (kWh)", "Teruglevering (kWh)"]) +
               bullets(["Bron: de 15-minutenstanden van uw P1-meter (HomeWizard).",
                        "Een maand zonder metingen wordt aangevuld met het verbruik van de weken ervoor en erna."]))
    return section(1, "Uw verbruik", f"{num(imp)} kWh afname en {num(exp)} kWh teruglevering per jaar.", body, details)


def vermogen(an, T) -> str:
    s = T["power"]
    room = s.tables.get("room_on_connection")
    st = s.tables["import_export_power"].set_index("flow")
    body = bullets([f"Aansluiting <b>{esc(an.conn.label)} A</b>: maximaal {num(an.conn.phase_kw, 2)} kW per fase.",
                    f"Hoogste afname (kwartiergemiddelde): {num(st.loc['import', 'max_kw'], 1)} kW; 99% van de tijd "
                    f"onder {num(st.loc['import', 'p99_kw'], 1)} kW. Hoogste teruglevering: "
                    f"{num(st.loc['export', 'max_kw'], 1)} kW."])
    if room is not None and len(room):
        rows = [[APPARATEN.get(r.appliance, r.appliance), num(r.kw, 1),
                 "–" if pd.isna(r.worst_total_kw) else num(r.worst_total_kw, 1),
                 "–" if pd.isna(r.over_limit_pct) else num(r.over_limit_pct, 1) + "%", OORDEEL.get(r.verdict, r.verdict)]
                for r in room.itertuples()]
        body += "<h3>Ruimte voor een extra apparaat</h3>" + table(
            rows, ["Apparaat", "Vermogen (kW)", "Hoogste totaal (kW)", "Tijd boven de grens", "Oordeel"])
    ph = s.tables.get("peak_per_phase")
    details = ""
    if ph is not None and len(ph):
        details += table([[r.phase.replace(" max W", ""), num(r.p50_kw, 2), num(r.p99_kw, 2), num(r.max_kw, 2)]
                          for r in ph.itertuples()], ["Fase", "Mediaan (kW)", "99% onder (kW)", "Hoogste piek (kW)"])
    details += bullets([
        "Per kwartier wordt <i>de piek van uw huis + het apparaat</i> vergeleken met de zekering (ampère × 230 V). "
        "Voor de piek wordt de meting per fase van de meter gebruikt; waar die ontbreekt het "
        "kwartiergemiddelde, dat korte pieken (waterkoker, oven) niet laat zien.",
        "Airco: mei–september 12–23 uur; warmtepomp: oktober–april; bijverwarming: december–februari; laadpaal: "
        "17–24 uur.",
        "Een slimme laadpaal of warmtepomp met belastingsturing leest de P1-meter en regelt terug: dan zijn de krappe "
        "gevallen veilig."])
    fits = [APPARATEN.get(r.appliance, r.appliance) for r in room.itertuples() if str(r.verdict).startswith("fits")] \
        if room is not None else []
    return section(2, "Vermogen en aansluiting", "Past op uw aansluiting: " + (", ".join(fits) or "–") + ".",
                   body, details)


def huidig(an, T, cur, N) -> str:
    t = T["current"].tables["current_contract_cost"]
    names = ["2026 (met salderen)", "2027–2029 (zonder salderen)", "vanaf 2030"]
    rows = [[n, eur(r.total), num(r.import_kwh), num(r.export_kwh)] for n, r in zip(names, t.itertuples())]
    kind = f" ({esc(N['contract_kind'])})" if N.get("contract_kind") else ""
    body = (bullets([f"Uw contract: <b>{esc(contract_nl(cur.label))}</b>{kind}.",
                     f"Nu kost stroom u ongeveer <b>{eur(t.total.iloc[0])}</b> per jaar; vanaf 2027 ongeveer "
                     f"<b>{eur(t.total.iloc[1])}</b> als u dit contract houdt.",
                     N.get("why_dearer", "Waarom duurder: de saldering stopt (teruggeleverde stroom wordt niet meer "
                                         "weggestreept tegen afname).")])
            + table(rows, ["Jaar", "Kosten per jaar", "Afname (kWh)", "Teruglevering (kWh)"]))
    cols = [("supply", "Levering"), ("energy_tax", "Energiebelasting"), ("feed_in_income", "Terugleververgoeding"),
            ("feed_in_costs", "Terugleverkosten"), ("fixed_supplier", "Vaste leveringskosten"), ("grid", "Netbeheer"),
            ("tax_reduction", "Vermindering energiebelasting"), ("total", "Totaal")]
    rows2 = [[label] + [eur(getattr(r, k)) for r in t.itertuples()] for k, label in cols]
    details = (table(rows2, ["Post"] + names) +
               bullets(["Bedragen per jaar inclusief btw, op uw gemeten verbruik.",
                        "Terugleververgoeding is wat u krijgt; terugleverkosten is wat uw leverancier rekent voor teruglevering."]))
    return section(3, "Uw huidige contract", f"Nu {eur(t.total.iloc[0])} per jaar, vanaf 2027 {eur(t.total.iloc[1])} als u "
                   "niets verandert.", body, details)


def contracten(an, T, roles, role_name) -> str:
    rows = []
    for k, c in roles:
        x = an.expected_by_contract.get(c.id, {})
        rows.append([f"{role_name[k]}: {contract_nl(c.label)}", eur(x.get("saldering")), eur(x.get("nosal_min50")),
                     eur(x.get("nosal_2030")), eur(c.welcome_bonus)])
    spec = dict(T["contracts"].interactive.get("contract_years") or {})
    if spec:
        spec["title"] = "Jaarkosten zonder batterij per prijsjaar (regels 2027–2029)"
        spec["subtitle"] = ("Dynamische contracten doorgerekend met de uurprijzen van elk jaar; vaste en variabele "
                            "contracten kosten elk jaar hetzelfde. Zonder welkomstbonus.")
        ser = []
        for s in spec["series"]:
            s = dict(s)
            if s.get("context"):           # the current contract: shown as a real option here
                s = {"label": "Huidig contract houden", "short": "huidig", "color": CONTRACT_COLORS["current"],
                     "values": s["values"]}
            elif "(cheapest fixed offer)" in s["label"]:
                s = dict(s, label=contract_nl(s["label"]).replace("(cheapest fixed offer)", "(goedkoopste vaste aanbieding)"),
                         short="vaste aanbieding", color=CONTRACT_COLORS["fixed"])
            else:
                s["label"] = contract_nl(s["label"]).replace("(dynamic)", "(dynamisch)").replace(
                    "(cheapest fixed offer)", "(goedkoopste vaste aanbieding)")
                s["short"] = contract_nl(s.get("short", "")).replace("fixed offer", "vaste aanbieding")
            ser.append(s)
        spec["series"] = ser
        spec["yLabel"] = "€ per jaar"
        spec.pop("contextLabel", None)
    d = (an.expected_by_contract.get(roles[0][1].id, {}) or {}).get("nosal_min50")
    dyn = next((c for k, c in roles if k == "dynamic"), None)
    body = table(rows, ["Contract", "2026 (salderen)", "2027–2029", "vanaf 2030", "Welkomstbonus (eenmalig)"])
    body += bullets([f"Overstappen naar <b>{esc(contract_nl(dyn.label))}</b> bespaart vanaf 2027 ongeveer "
                     f"<b>{eur(d - an.cost27(dyn))} per jaar</b> ten opzichte van uw huidige contract (zonder batterij)."
                     if dyn is not None and d is not None else ""]) + (chart(spec) if spec else "")
    top = T["contracts"].tables.get("Last 3 years (headline) – 2027-2029 rules (no saldering, 50% minimum)")
    details = ""
    if top is not None:
        details += "<h3>Alle contracten, regels 2027–2029</h3>" + table(
            [[contract_nl(r.contract), "dynamisch" if r.type == "dynamic" else "vast/variabel", eur(r.expected_eur_year),
              eur(r.switch_bonus_eur_once)] for r in top.head(12).itertuples()],
            ["Contract", "Soort", "Kosten per jaar", "Welkomstbonus"])
    details += bullets(["Dynamisch: u betaalt per uur (of kwartier) de beursprijs plus een opslag; zonnestroom levert "
                        "midden op de dag weinig op omdat de beursprijs dan laag is.",
                        *([f"Vaste aanbieding: {esc(contract_nl(fx.label))}."] if (fx := an.cheapest_fixed()) else []),
                        "Kosten inclusief btw, netbeheer en vermindering energiebelasting, zonder welkomstbonus."])
    return section(4, "Contracten zonder batterij", (f"Dynamisch ({contract_nl(dyn.label)}) is vanaf 2027 ongeveer "
                   f"{eur(d - an.cost27(dyn))} per jaar goedkoper dan uw huidige contract.")
                   if dyn is not None and d is not None else "Contractvergelijking.", body, details)


def batterijen(an, T, focus, N) -> str:
    rows = []
    for f in focus:
        b = f["b"]
        p = an.battery_price(b)
        src = b.best_price(float(an.cfg.get("battery_defaults", {}).get("de_travel_cost_eur", 0)))[1] if an.use_de else "NL"
        rows.append([b.name, num(b.nominal_kwh, 2), num(b.usable_kwh, 2), num(min(b.max_charge_w, b.max_discharge_w) / 1000, 1),
                     eur(p), eur(p / b.usable_kwh if p else None), "Nederland" if src != "DE" else "Duitsland"])
    body = (table(rows, ["Batterij", "Capaciteit (kWh)", "Bruikbaar (kWh)", "Vermogen (kW)", "Prijs", "Per bruikbare kWh",
                         "Gekocht in"]) +
            bullets(N.get("battery_bullets") or
                    [f"Per grootteklasse ({', '.join(lab for lab, *_ in an.size_classes())}) de "
                     "batterij met de snelste terugverdientijd uit de vergelijking.",
                     "Prijzen van vandaag, inclusief Black Friday-aanbiedingen waar bekend."]))
    life = [f"{esc(f['b'].name)}: rendement {num(f['b'].rte * 100)}%, sluimerverbruik {num(f['b'].standby_w)} W, "
            f"levensduur {num(f['b'].warranty_years)} jaar of {num(f['b'].cycle_life)} cycli, daarna "
            f"{num(f['b'].eol_capacity * 100)}% capaciteit" for f in focus]
    details = bullets(life + (N.get("battery_details") or
                              ["De capaciteit loopt over de levensduur terug; rendement en sluimerverbruik zijn "
                               "aannames waar de fabrikant ze niet noemt."]))
    prices = [p for p in (an.battery_price(f["b"]) for f in focus) if p]
    return section(5, N.get("battery_title", "Batterijen"),
                   f"{len(rows)} uitvoeringen, vanaf {eur(min(prices))}." if prices else "", body, details)


def terugverdientijd(an, roles, role_name, comp, focus, N) -> str:
    sizes = [f["b"] for f in focus]
    rows, pay_series, cost_rows = [], [], []
    for k, c in roles:
        vals = []
        for b in sizes:
            x = comp[(comp._bid == b.id) & (comp._role == k)]
            if not len(x):
                vals.append(None)
                continue
            x = x.iloc[0]
            vals.append(round(float(x.payback_years), 1) if np.isfinite(x.payback_years) else None)
            rows.append([role_name[k], b.name, eur(x.saving_eur_year_2027), num(x.payback_years, 1) + " jaar",
                         eur(x.yearly_cost_with_battery_2027), eur(x.npv_eur)])
        pay_series.append({"label": f"{role_name[k]} ({contract_nl(c.label)})", "color": CONTRACT_COLORS[k],
                           "values": vals})
    spec = {"kind": "bars", "unit": "jaar", "yLabel": "terugverdientijd (jaar)",
            "title": "Terugverdientijd van de batterij per contract",
            "subtitle": "Korter is beter. Standaardinstelling van de Marstek (zelfverbruik); op dynamisch ook "
                        "met slimme sturing.",
            "categories": [f"{num(b.nominal_kwh, 2)} kWh" for b in sizes], "series": pay_series}
    cur = next((c for k, c in roles if k == "current"), None)
    dyn = next((c for k, c in roles if k == "dynamic"), None)
    txt = []
    if cur is not None and dyn is not None:
        a = comp[comp._role == "current"].sort_values("payback_years").iloc[0]
        bdy = comp[comp._role == "dynamic"].sort_values("yearly_cost_with_battery_2027").iloc[0]
        txt = [f"<b>{esc(role_name['current'])}:</b> de {esc(a.battery)} bespaart {eur(a.saving_eur_year_2027)} per jaar en "
               f"is terugverdiend in <b>{num(a.payback_years, 1)} jaar</b>.",
               f"<b>Dynamisch contract:</b> de batterij bespaart daar minder (teruggeleverde stroom is midden op de dag "
               f"weinig waard, maar uw afname ook goedkoper), maar uw totale kosten zijn lager: met de "
               f"{esc(bdy.battery)} ongeveer <b>{eur(bdy.yearly_cost_with_battery_2027)} per jaar</b>.",
               *smart_bullet(comp, role_name, N),
               *([N["stored_kwh"]] if N.get("stored_kwh") else [])]
    body = bullets(txt) + chart(spec)
    details = (table(rows, ["Contract", "Batterij", "Besparing per jaar", "Terugverdientijd", "Jaarkosten met batterij",
                            "Netto contante waarde"]) +
               bullets(["Terugverdientijd vanaf aankoop op 2 oktober 2026: 2026 met salderen, 2027–2029 zonder "
                        "salderen, daarna de regels vanaf 2030; de capaciteit loopt terug met het gebruik.",
                        "Netto contante waarde: alle besparingen over de levensduur, verdisconteerd met 3% per jaar, "
                        "min de prijs. Positief = de batterij levert meer op dan ze kost.",
                        "Standaardinstelling: zelfverbruik (laden met zonnestroom die u anders teruglevert, ontladen als "
                        "u afneemt).", SMART_UITLEG]))
    best = comp.sort_values("yearly_cost_with_battery_2027").iloc[0]
    fast = comp.sort_values("payback_years").iloc[0]
    fast_sm = " met slimme sturing" if fast["_role"] == "smart" else ""
    best_sm = " en slimme sturing" if best["_role"] == "smart" else ""
    return section(6, "Terugverdientijd van de batterij",
                   f"Snelst terugverdiend: {fast.battery} op {contract_nl(fast.contract)}{fast_sm} "
                   f"({num(fast.payback_years, 1)} jaar); "
                   f"laagste jaarkosten: {contract_nl(best.contract)} met de {best.battery}{best_sm} "
                   f"({eur(best.yearly_cost_with_battery_2027)} per jaar).", body, details)


def smart_bullet(comp, role_name, N=None) -> list[str]:
    N = N or {}
    """One line: what smart control adds on the dynamic contract, per battery size."""
    std, sm = comp[comp._role == "dynamic"], comp[comp._role == "smart"]
    parts = []
    for _, r in sm.sort_values("price_eur").iterrows():
        x = std[std._bid == r["_bid"]]
        if len(x):
            x = x.iloc[0]
            parts.append(f"{esc(r.battery.replace(N.get('battery_prefix', '@@'), ''))}: {eur(r.saving_eur_year_2027)} in plaats "
                         f"van {eur(x.saving_eur_year_2027)} per jaar, terugverdiend in {num(r.payback_years, 1)} "
                         f"in plaats van {num(x.payback_years, 1)} jaar")
    if not parts:
        return []
    return [f"<b>{esc(role_name['smart'])}:</b> de batterij bespaart meer omdat ze ook in de winter wordt gebruikt "
            f"(laden in goedkope uren, ontladen in de avondpiek). " + "; ".join(parts) + "."]


def door_het_jaar(an, T) -> str:
    s = T["sanity"]
    t = s.tables["self_consumption_check"]
    spec = dict(s.interactive.get("soc_year_15min") or {})
    if spec:
        spec["title"] = "Laadtoestand van de batterij door het jaar (elk kwartier)"
        spec["y"] = "kWh"
        spec["xName"] = "Tijd"
        spec["series"] = [dict(x, label=x["label"].replace("– state of charge (kWh)", "– laadtoestand (kWh)"))
                          for x in spec["series"]]
    rows = [[r.battery, str(r.days_full), str(r.days_below_half), num(r.efc_per_year), num(r.import_before_kwh),
             num(r.import_after_kwh), num(r.export_before_kwh), num(r.export_after_kwh)] for r in t.itertuples()]
    body = bullets([f"De {esc(r.battery)} is op <b>{r.days_full} dagen</b> helemaal vol en blijft op "
                    f"{r.days_below_half} (winter)dagen onder de helft." for r in t.itertuples()])
    body += chart(spec) if spec else ""
    details = (table(rows, ["Batterij", "Dagen vol", "Dagen onder de helft", "Laadcycli per jaar", "Afname zonder (kWh)",
                            "Afname met (kWh)", "Teruglevering zonder (kWh)", "Teruglevering met (kWh)"]) +
               bullets(["Gesimuleerd met de standaardinstelling (zelfverbruik) op uw gemeten jaar.",
                        "Sleep over de grafiek om in te zoomen, dubbelklik om terug te gaan."]))
    return section(7, "De batterij door het jaar", "Hoe vaak de batterij vol raakt en hoeveel afname ze bespaart.",
                   body, details)
