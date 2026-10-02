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
