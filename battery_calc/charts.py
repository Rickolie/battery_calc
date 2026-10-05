"""Matplotlib charts, returned as PNG bytes so the CLI and the website share them."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f", "#9c6644", "#6d597a"]


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def monthly_flows(monthly) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(monthly))
    ax.bar(x - 0.2, monthly["import_kwh"], 0.4, label="Import", color=PALETTE[0])
    ax.bar(x + 0.2, monthly["export_kwh"], 0.4, label="Export", color=PALETTE[1])
    for i, share in enumerate(monthly["synthetic_share"]):
        if share > 0.5:
            ax.annotate("synthetic", (x[i], max(monthly["import_kwh"].iloc[i], monthly["export_kwh"].iloc[i])),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(monthly.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("kWh")
    ax.set_title("Monthly import and export (profile year)")
    ax.legend()
    return _png(fig)


def price_overview(ov) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(ov.index.astype(str), ov["avg_price_eur_kwh"] * 100, color=PALETTE[0], label="Average price")
    ax.plot(ov.index.astype(str), ov["avg_daily_spread_eur_kwh"] * 100, "o-", color=PALETTE[1],
            label="Avg daily max–min spread")
    ax.set_ylabel("ct/kWh (kale)")
    ax2 = ax.twinx()
    ax2.plot(ov.index.astype(str), ov["negative_price_hours"], "s--", color=PALETTE[2], label="Negative hours")
    ax2.set_ylabel("negative-price hours")
    ax.set_title("Day-ahead price history")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    return _png(fig)


def contract_years(series: dict, fixed_label: str | None, fixed_value: float | None, title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (label, s) in enumerate(series.items()):
        ax.plot([str(k) for k in s.index], s.values, "o-", color=PALETTE[i % len(PALETTE)], label=label)
    if fixed_value is not None:
        ax.axhline(fixed_value, color="black", ls="--", label=fixed_label)
    ax.set_ylabel("€ / year")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _png(fig)


def breakeven_lines(rows: list[dict], max_charge: float = 0.5) -> bytes:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.linspace(0, max_charge, 50)
    ax.plot(x, x, color="grey", ls=":", label="no losses, no wear")
    for i, r in enumerate(rows):
        ax.plot(x, x / r["rte"] + r["wear_eur_kwh"], color=PALETTE[i % len(PALETTE)],
                label=f"{r['battery']} ({r['price_variant']})")
    ax.set_xlabel("charge price (€/kWh)")
    ax.set_ylabel("break-even sell/use price (€/kWh)")
    ax.set_title("Break-even price vs charge price")
    ax.legend(fontsize=7)
    return _png(fig)


def soc_weeks(traces: dict) -> bytes:
    fig, axes = plt.subplots(len(traces), 1, figsize=(8, 2.6 * len(traces)), squeeze=False)
    for ax, (title, (idx, soc, net)) in zip(axes[:, 0], traces.items()):
        ax.plot(idx, soc, color=PALETTE[0], label="SoC (kWh)")
        ax2 = ax.twinx()
        ax2.plot(idx, net * 4, color=PALETTE[1], alpha=0.5, lw=0.7, label="house net (kW)")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("kWh")
        ax2.set_ylabel("kW")
    return _png(fig)


def payback_scatter(df) -> bytes:
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (strategy, g) in enumerate(df.groupby("strategy")):
        ax.scatter(g["usable_kwh"], g["payback_years"], color=PALETTE[i % len(PALETTE)], label=strategy)
    ax.set_xlabel("usable capacity (kWh)")
    ax.set_ylabel("simple payback (years)")
    ax.set_title("Payback vs capacity")
    ax.legend(fontsize=8)
    return _png(fig)


def savings_years(series: dict, title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (label, s) in enumerate(series.items()):
        ax.plot([str(k) for k in s.index], s.values, "o-", color=PALETTE[i % len(PALETTE)], label=label)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("€ saved / year")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _png(fig)


def power_duration(imp_sorted, exp_sorted, marks) -> bytes:
    """Load-duration curves: how many hours per year import/export exceed a power."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for arr, label, col in ((imp_sorted, "Import", PALETTE[0]), (exp_sorted, "Export (solar surplus)", PALETTE[1])):
        hrs = np.arange(1, len(arr) + 1) * 0.25
        ax.plot(hrs, arr, color=col, label=label)
    for m in marks:
        ax.axhline(m, color="grey", ls=":", lw=1)
        ax.annotate(f"{m:g} kW", (ax.get_xlim()[1] * 0.98, m), ha="right", va="bottom", fontsize=8, color="grey")
    ax.set_xlabel("hours per year at or above this power")
    ax.set_ylabel("kW (15-min average)")
    ax.set_title("Power duration curve")
    ax.legend(fontsize=8)
    return _png(fig)


EARN_PARTS = [("avoided_import", "Avoided import", "#2a6f97"), ("sold_to_grid", "Sold to grid", "#81b29a"),
              ("solar_feed_in_given_up", "Feed-in given up (stored solar)", "#e07a5f"),
              ("grid_charging", "Grid charging", "#9c6644"), ("standby_and_other", "Standby / other", "#8d8d8d")]


def earnings_stacked(tbl, title) -> bytes:
    """One panel per strategy: stacked yearly earnings (positive up, costs down)."""
    strategies = list(dict.fromkeys(tbl["strategy"]))
    n = len(strategies)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), sharey=True, squeeze=False)
    for ax, strat in zip(axes.flat, strategies):
        g = tbl[tbl.strategy == strat]
        x = np.arange(len(g))
        pos = np.zeros(len(g))
        neg = np.zeros(len(g))
        for key, label, col in EARN_PARTS:
            v = g[key].values
            up = np.where(v > 0, v, 0)
            dn = np.where(v < 0, v, 0)
            ax.bar(x, up, bottom=pos, color=col, label=label, width=0.75)
            ax.bar(x, dn, bottom=neg, color=col, width=0.75)
            pos += up
            neg += dn
        ax.plot(x, g["net_saving"].values, "o", color="black", ms=4, label="Net saving")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(g["year"].astype(str), rotation=90, fontsize=7)
        ax.set_title(strat, fontsize=9)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes[0, 0].set_ylabel("€ per year")
    h, l = axes[0, 0].get_legend_handles_labels()
    seen = dict(zip(l, h))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"Yearly earnings per strategy – {title}", fontsize=10)
    fig.tight_layout(rect=(0, 0.08 + 0.02 * rows, 1, 0.95))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def kiln_days(summ, battery_name, label) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = summ["kiln_kw"].values
    ax.plot(x, summ["free_days_no_battery"], "o-", color=PALETTE[1], label="Solar surplus only")
    key = f"free_days_{label}"
    if key in summ:
        ax.plot(x, summ[key], "o-", color=PALETTE[0], label=f"Surplus + {battery_name}")
    ax.set_xlabel("kiln rated power (kW)")
    ax.set_ylabel("free firing days per year")
    ax.set_title("Pottery kiln: days per year a full firing runs on free power")
    ax2 = ax.twinx()
    ax2.bar(x, summ["firing_kwh"], width=0.25, color="grey", alpha=0.25, label="energy per firing")
    ax2.set_ylabel("kWh per firing", color="grey")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(fontsize=8, loc="upper right")
    return _png(fig)


def soc_year(daily, usable, title) -> bytes:
    """Whole year, one bar group per day: energy in/out and the SoC range, with a legend."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = daily.index
    ax.bar(x, daily["charged_from_solar_kwh"], width=1.0, color=PALETTE[3], label="Charged from solar (kWh/day)")
    ax.bar(x, daily["charged_from_grid_kwh"], width=1.0, bottom=daily["charged_from_solar_kwh"], color=PALETTE[4],
           label="Charged from grid (kWh/day)")
    ax.bar(x, -daily["discharged_kwh"], width=1.0, color=PALETTE[1], label="Discharged (kWh/day, negative)")
    ax.plot(x, daily["max_soc_kwh"], color=PALETTE[0], lw=1.2, label="Highest state of charge that day (kWh)")
    ax.plot(x, daily["min_soc_kwh"], color=PALETTE[2], lw=0.8, label="Lowest state of charge that day (kWh)")
    ax.axhline(usable, color=PALETTE[0], ls="--", lw=1, label=f"Usable capacity ({usable:.1f} kWh)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("kWh")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.autofmt_xdate()
    return _png(fig)


def daily_charge(daily, usable, title) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = daily.index
    ax.bar(x, daily["solar_surplus_kwh"], width=1.0, color="#f2cc8f", label="Solar surplus that day (kWh)")
    ax.bar(x, daily["charged_from_solar_kwh"], width=1.0, color=PALETTE[3], label="Stored in the battery (kWh)")
    ax.plot(x, daily["import_before_kwh"], color=PALETTE[2], lw=0.8, label="Grid import without battery (kWh)")
    ax.plot(x, daily["import_after_kwh"], color=PALETTE[1], lw=0.8, label="Grid import with battery (kWh)")
    ax.axhline(usable, color=PALETTE[0], ls="--", lw=1, label=f"Usable capacity ({usable:.1f} kWh)")
    ax.set_ylabel("kWh per day")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.autofmt_xdate()
    return _png(fig)


def daily_savings(saving, title) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = saving.index
    ax.bar(x, saving.values, width=1.0, color=np.where(saving.values >= 0, PALETTE[3], PALETTE[1]),
           label="Saving that day (€)")
    ax.set_ylabel("€ per day")
    ax2 = ax.twinx()
    ax2.plot(x, saving.cumsum().values, color=PALETTE[0], lw=1.5, label="Cumulative saving (€)")
    ax2.set_ylabel("€ cumulative")
    ax.set_title(title)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.autofmt_xdate()
    return _png(fig)
