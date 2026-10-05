"""Plain-language descriptions of every report column, shown under each table
(website and report). Money is € incl. VAT unless stated; prices per kWh are
all-in (energy, energy tax, VAT) unless stated."""
from __future__ import annotations

import re

GLOSSARY = {
    # ------------------------------------------------------------ general
    "battery": "Brand and model of the battery configuration (capacity in the name).",
    "battery_id": "Short code of the battery in data/online/batteries.csv.",
    "id": "Short code of the battery in data/online/batteries.csv.",
    "contract": "Electricity contract: supplier – product. 'Current contract – vast.txt' is your own fixed contract.",
    "strategy": "How the battery is controlled: self_consumption = store solar surplus, use it when the house "
                "imports; timed = fixed daily charge/discharge windows; dynamic = day-ahead prices with peak "
                "reservation (needs custom control); dynamic_sell = same, may also sell to the grid; hbc_default / "
                "hbc_pv_first = Home Battery Control's Dynamic strategy as-is; perfect_foresight = theoretical "
                "upper bound with all prices and usage known in advance (not achievable).",
    "scenario": "Which rules are applied: 2026 (saldering), 2027–2029 (no saldering, at least 50% feed-in "
                "compensation) or 2030+ (no legal minimum).",
    "rules": "Rules that apply in that calendar year (2026 saldering, 2027–2029, 2030+).",
    "analysis": "headline = average of the last 3 full price years (2023–2025); full = all price years 2013–2025.",
    "price_variant": "Which purchase price is used: NL current (today's lowest Dutch price incl. VAT), NL lowest-ever "
                     "(lowest price the scraper has seen), DE 0% VAT (German shop price without VAT + travel), "
                     "NL/DE Black Friday (real deal found in the Black Friday window) and (est.) = estimated "
                     "Black Friday discount on today's price.",
    "estimated": "Specs that were not published and use defaults (RTE 85%, standby 10 W, cycle life 6,000, "
                 "end-of-life capacity 70%, DoD 90%); 'yes' or the list of fields concerned.",
    "verified": "yes = all values checked against the source; partial = prices/specs from a shop page, some specs "
                "defaulted; no = hand-entered or a template.",
    "usable_kwh": "Usable storage capacity in kWh (nominal capacity × depth of discharge).",
    "rte": "Round-trip efficiency: share of the energy put in that comes back out (0.85 = 85%; 15% is lost).",
    "year": "Calendar year.",
    "type": "Contract type (fixed or dynamic) or battery type (plug_in, hybrid).",
    # ------------------------------------------------------------ data quality
    "index": "Month (YYYY-MM) of the profile year.",
    "import_kwh": "Electricity taken from the grid (kWh).",
    "export_kwh": "Electricity sent to the grid, i.e. solar surplus (kWh).",
    "synthetic_share": "Share of the month that is filled in synthetically because meter data is missing "
                       "(1.0 = completely synthetic).",
    "avg_price_eur_kwh": "Average day-ahead (EPEX) price that year, € per kWh excl. taxes and VAT.",
    "avg_daily_spread_eur_kwh": "Average difference between the most expensive and the cheapest hour of a day "
                                "(€/kWh, excl. taxes) – what a battery could trade on.",
    "negative_price_hours": "Hours with a negative day-ahead price that year.",
    "hours": "Hours with a price in the file (8,760, or 8,784 in a leap year; less for a partial year).",
    "leverancier": "Supplier (energieKNL reference file).",
    "jaar": "Year of the energieKNL snapshot.",
    "maand": "Month of the energieKNL snapshot.",
    "leveringskost_eur_year": "Yearly delivery costs per supplier as listed by energieKNL (reference only).",
    # ------------------------------------------------------------ power profile
    "flow": "import = power taken from the grid; export = solar surplus sent to the grid.",
    "hours_per_year": "Hours per year with any import (or export).",
    "median_kw": "Typical power when importing/exporting (half of the quarter-hours are lower).",
    "p50_kw": "Typical (median) peak power on this phase per quarter-hour.",
    "p90_kw": "Power exceeded in only 10% of the quarter-hours with import/export.",
    "p99_kw": "Power exceeded in only 1% of the quarter-hours – the high peaks.",
    "max_kw": "Highest value measured.",
    "battery_power_kw": "Maximum charge/discharge power of a battery (kW).",
    "share_of_surplus_it_can_store": "Share of the yearly solar surplus that a battery with this power could "
                                     "absorb (if it is not full). 800 W socket batteries miss the midday peaks.",
    "share_of_import_it_can_cover": "Share of the yearly grid import a battery with this power could supply "
                                    "(if it is not empty).",
    "phase": "Phase of the grid connection (meter column with the highest power per quarter-hour).",
    # ------------------------------------------------------------ current contract / contract tables
    "supply": "Cost of the energy itself (supplier tariff × kWh), incl. VAT.",
    "energy_tax": "Energy tax (energiebelasting) incl. VAT; under saldering only on import minus export.",
    "feed_in_income": "Money received for exported power (feed-in compensation).",
    "feed_in_costs": "Feed-in costs (terugleverkosten) charged by the supplier.",
    "fixed_supplier": "Fixed delivery costs of the supplier per year (vaste leveringskosten).",
    "grid": "Grid operator costs (netbeheerkosten) – the same for every supplier.",
    "tax_reduction": "Yearly energy tax reduction (vermindering energiebelasting), negative = you get it back.",
    "bonus_amortised": "Switch bonus spread over the contract term (negative = discount).",
    "total": "Total yearly cost: all items above added up.",
    "total_real_months_only": "Same total but only over months with real meter data (synthetic month left out) – "
                              "a check that the filled-in month does not distort the result.",
    "expected_eur_year": "Expected yearly cost: average over the price years in this analysis (the switch bonus "
                         "is not included here).",
    "expected": "Average yearly cost over the years in this group.",
    "min": "Lowest yearly cost of all price years (cheapest year).",
    "max": "Highest yearly cost of all price years (most expensive year).",
    "eur_per_kwh_imported": "Total yearly cost divided by kWh imported – an all-in price per kWh.",
    "real_months_only": "Expected cost over the real meter months only (synthetic month left out).",
    "price_years": "Day-ahead price years used; 'stated tariffs' for fixed contracts.",
    "diff_vs_current": "Difference with your current contract per year (negative = cheaper).",
    "avg_eur_year_over_term": "Average cost per year over the contract term starting today, switching rules on "
                              "1 January 2027, with the one-off switch bonus subtracted once.",
    "switch_bonus_eur_once": "One-off welcome/switch bonus of the supplier (€).",
    "group": "Group of price years: pre-crisis (2013–2020), crisis (2021–2022), recent (2023–2026).",
    "years": "Years in the group.",
    # ------------------------------------------------------------ battery dataset
    "phases": "1 = single-phase battery, 3 = three-phase.",
    "max_charge_w": "Maximum charging power (W) according to the shop.",
    "max_discharge_w": "Maximum discharging power (W); plug-in batteries on a normal socket are limited to 800 W.",
    "power_cap_w": "Power actually used in the simulation: the lower of the battery's power and the connection "
                   "limit (amps × 230 V minus the margin).",
    "standby_w": "Own power use when idle (W); 10 W costs about 88 kWh per year.",
    "cycle_life": "Number of full charge/discharge cycles until the end-of-life capacity is reached.",
    "eol_capacity": "Remaining capacity at end of life (0.70 = 70%).",
    "warranty_years": "Warranty in years; also used as the calendar life of the battery.",
    "warranty_mwh": "Warranty limit on total energy delivered (MWh), if the manufacturer sets one.",
    "price_nl": "Lowest current Dutch shop price incl. VAT (€).",
    "price_nl_lowest": "Lowest Dutch price ever seen by the scraper (€).",
    "price_de": "Lowest current German shop price, 0% VAT (€).",
    "extra_hw": "Extra hardware cost (meter, cabling, own group) added to the price (€).",
    "eur_per_kwh_nominal": "Price per kWh of nominal (stated) capacity.",
    "eur_per_kwh_usable": "Price per kWh of usable capacity – the fairer comparison.",
    "solar_during_outage": "Can solar power still charge the battery when the grid or main switch is off? Only "
                           "with panels on the battery's own solar input (or a micro-inverter on its backup port).",
    "dc_solar_input_w": "Power of the battery's own DC solar inputs (MPPT), W.",
    "missing_fields": "Specs not published by the shop.",
    "estimated_with_defaults": "Missing specs that were filled with default values for the simulation.",
    "in_scope": "Whether the battery is included for this connection (three-phase batteries are left out on a "
                "single-phase connection).",
    "note": "Reason a battery is excluded or flagged.",
    # ------------------------------------------------------------ objective 1
    "purchase_eur": "Purchase price used (incl. extra hardware; DE incl. travel cost).",
    "cycles_per_year": "Full cycles per year assumed for the wear cost (250 by default; section 8 uses the "
                       "simulated number).",
    "lifetime_kwh": "kWh the battery delivers over its whole life (limited by cycles, warranty or calendar life).",
    "lifetime_limit": "Which limit ends the battery's life first: cycles, warranty throughput or calendar life.",
    "wear_eur_kwh": "Wear cost: purchase price ÷ lifetime kWh, € per kWh delivered.",
    "be_grid_0": "Break-even price when charging for free (€0): equal to the wear cost.",
    "be_grid_avg": "Break-even price when charging from the grid at the average all-in price.",
    "be_own_solar_2026": "Break-even price for stored own solar under saldering (charge price = full import price, "
                         "because exported solar is netted).",
    "be_own_solar_2027": "Break-even price for stored own solar from 2027 (charge price = feed-in compensation "
                         "given up, about €0.07).",
    "setting_all_in_eur_kwh": "Minimum price difference to enter in a battery app / Home Assistant that compares "
                              "prices incl. taxes (all-in).",
    "setting_spot_eur_kwh": "Same, for an app that compares spot (EPEX) prices without taxes (all-in ÷ 1.21).",
    # ------------------------------------------------------------ objective 3
    "price_eur": "Purchase price used for this row (€).",
    "payback_years": "Years until the savings equal the purchase price ('never' = not within the battery's life).",
    "payback_min": "Fastest payback over the individual price years.",
    "payback_max": "Slowest payback over the individual price years.",
    "npv_eur": "Net present value: savings over the battery's life discounted at 3% per year, minus the price. "
               "Positive = a good investment.",
    "lifetime_net_saving_eur": "Total savings over the battery's life minus the purchase price (not discounted).",
    "efc_per_year": "Equivalent full cycles per year: total energy discharged ÷ usable capacity.",
    "end_of_life_year": "Year in which the battery reaches end of life.",
    "life_limit": "What ends its life first: calendar life (warranty years), cycles or warranty throughput.",
    "standby_eur_year_approx": "Approximate yearly cost of the battery's standby power use (€).",
    "variant": "Sensitivity case: base, RTE 5 points lower/higher, or standby power doubled.",
    "saving_eur_year_2027_rules": "Yearly saving under the 2027–2029 rules (€).",
    "scale": "Factor applied to the battery's break-even for Dynamic decisions (1.0 = exact break-even).",
    "saving_eur_year": "Yearly saving with that factor (€).",
    "best_for_battery": "True for the factor that saves most for this battery.",
    "avoided_import": "Value of battery energy used in the house instead of grid power (€).",
    "sold_to_grid": "Value of battery energy sold to the grid (€).",
    "solar_feed_in_given_up": "Feed-in compensation missed because solar surplus was stored instead of exported "
                              "(negative, €).",
    "grid_charging": "Cost of charging the battery from the grid (negative, €).",
    "standby_and_other": "Standby use, saldering netting and feed-in cost tiers (closes to the real saving, €).",
    "net_saving": "Saving that year: the sum of the parts (€).",
    # ------------------------------------------------------------ Black Friday
    "nl_now_eur": "Today's lowest Dutch price incl. VAT (€).",
    "payback_now": "Payback at today's Dutch price (years).",
    "nl_deal_eur": "Real Dutch Black Friday deal found by the scraper (€, incl. VAT); empty = no deal found yet.",
    "nl_deal_discount": "Discount of the Dutch deal against the last normal price before the Black Friday window.",
    "payback_nl_deal": "Payback at the Dutch deal price (years).",
    "nl_est_eur": "Estimated Dutch Black Friday price: today's price minus the estimated discount (€).",
    "payback_nl_est": "Payback at the estimated Dutch Black Friday price (years).",
    "de_now_eur": "Today's German price, 0% VAT (€); German shops often deliver only in Germany.",
    "de_deal_eur": "Real German Black Friday deal found by the scraper (€, 0% VAT).",
    "de_deal_discount": "Discount of the German deal against the last normal German price.",
    "payback_de_deal": "Payback at the German deal price plus travel cost (years).",
    "payback_de_est": "Payback at the estimated German Black Friday price plus travel cost (years).",
    "deal_found": "Date and shop where the deal price was found.",
    "best_payback": "Best payback of the real deals, or of the estimate/today's price if no deal is known yet.",
    # ------------------------------------------------------------ advice
    "saving_eur_year_2027": "Average yearly saving with this battery from 2027 (2027–2029 rules, last 3 price years), €.",
    "days_full": "Days per year on which the battery gets completely full (self-consumption on your profile year).",
    "days_below_half": "Days per year on which the battery never gets more than half full (mostly winter) – "
                       "capacity that sits idle.",
    "share_of_surplus_stored": "Share of your yearly solar surplus that ends up in the battery instead of the grid.",
    "option": "Contract, with or without the recommended battery.",
    "yearly_cost_eur_2027": "Total yearly electricity cost from 2027 (2027–2029 rules, last 3 price years), incl. "
                            "taxes and fixed costs, minus the battery saving (purchase price not included).",
    "battery_saving_eur": "Yearly saving of the battery in this combination (€).",
    "date": "Day.",
    "solar_surplus_kwh": "Solar surplus that day without a battery: what would otherwise go to the grid (kWh).",
    "charged_from_solar_kwh": "Energy stored in the battery from solar surplus that day (kWh).",
    "charged_from_grid_kwh": "Energy charged from the grid that day (kWh), e.g. at very low dynamic prices.",
    "discharged_kwh": "Energy the battery delivered that day (kWh).",
    "import_before_kwh": "Grid import that day without a battery (kWh).",
    "import_after_kwh": "Grid import that day with the battery (kWh).",
    "max_soc_kwh": "Highest state of charge that day (kWh).",
    "min_soc_kwh": "Lowest state of charge that day (kWh).",
    "full": "True if the battery got completely full that day.",
    "saving_eur": "Saving that day compared with no battery (€).",
    # ------------------------------------------------------------ kiln
    "kiln_kw": "Rated power of the kiln (kW).",
    "firing_kwh": "Energy for one firing to maximum temperature: kW × hours × average power share.",
    "free_days_no_battery": "Days per year on which a full firing runs on solar surplus alone (at most 5% from "
                            "the grid).",
    "avg_grid_kwh_per_firing_no_battery": "Average extra grid energy per firing without a battery, over all days.",
    "avg_cost_per_firing_no_battery": "Average extra grid cost per firing without a battery (€).",
    "free_days_with_battery": "Days per year on which a full firing runs on solar surplus plus the chosen battery.",
    "avg_grid_kwh_per_firing_with_battery": "Average extra grid energy per firing with the battery, incl. evening "
                                            "use the battery can then no longer cover.",
    "avg_cost_per_firing_with_battery": "Average extra grid cost per firing with the battery (€).",
    "months_with_free_days": "Months (1 = January) with at least one free firing day.",
}

# Columns whose name carries a value: (regex, description using groups)
PATTERNS = [
    (r"at_charge_(\d+\.\d+)", "Minimum price difference worth charging when charging at €{0}/kWh: the discharge "
                              "price must be at least this much higher (all-in, €/kWh)."),
    (r"saving_(.+)", "Average yearly saving with the battery under the {0} (€)."),
    (r"^(\d{1,2})$", "Free firing days in month {0} (1 = January)."),
]

# Table-specific wording: table-name regex -> {column: description}
TABLE_OVERRIDES = [
    (r"by_year_group", {"min": "Cheapest year in the group (€).", "max": "Most expensive year in the group (€)."}),
    (r"battery_specs", {"type": "plug_in = plug-in battery, hybrid = battery with its own solar inputs."}),
    (r"Last 3 years|Full history", {"type": "fixed = fixed tariffs, dynamic = hourly/quarter-hourly day-ahead prices."}),
    (r"import_export_power", {"max_kw": "Highest 15-minute average power measured (kW)."}),
    (r"peak_per_phase", {"max_kw": "Highest power peak on this phase (kW).",
                         "p99_kw": "Peak exceeded in only 1% of the quarter-hours on this phase (kW)."}),
    (r"monthly_flows", {"index": "Month (YYYY-MM) of the profile year."}),
]


def describe(column: str, table: str = "") -> str | None:
    for pat, over in TABLE_OVERRIDES:
        if re.search(pat, table, re.I) and column in over:
            return over[column]
    if column in GLOSSARY:
        return GLOSSARY[column]
    for pat, text in PATTERNS:
        m = re.fullmatch(pat, column)
        if m:
            return text.format(*m.groups())
    return None


def describe_table(columns, table: str = "") -> list[tuple[str, str]]:
    out = []
    for c in columns:
        c = str(c)
        if c.startswith("_"):
            continue
        d = describe(c, table)
        if d:
            out.append((c, d))
    return out
