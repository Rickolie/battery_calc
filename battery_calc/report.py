"""Markdown / HTML / CSV output for a run (shared by CLI and website)."""
from __future__ import annotations

import base64
import html
import os
import re

import numpy as np
import pandas as pd


YEAR_COLS = {"year", "jaar", "price_year", "maand"}


def _fmt(v, col=""):
    if str(col).lower() in YEAR_COLS and isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        if np.isinf(v):
            return "never" if v > 0 else "-inf"
        return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:,.2f}"
    return "" if v is None else str(v)


def table_md(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df is None or df.empty:
        return "_(empty)_"
    d = df.head(max_rows)
    if not isinstance(d.index, pd.RangeIndex):
        d = d.reset_index()
    cols = [c for c in d.columns if not str(c).startswith("_")]
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in d.to_dict("records"):
        lines.append("| " + " | ".join(_fmt(r[c], c).replace("|", "/") for c in cols) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_{len(df) - max_rows} more rows in the CSV._")
    return "\n".join(lines)


def table_html(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df is None or df.empty:
        return "<p><em>(empty)</em></p>"
    d = df.head(max_rows)
    if not isinstance(d.index, pd.RangeIndex):
        d = d.reset_index()
    cols = [c for c in d.columns if not str(c).startswith("_")]
    out = ["<div class='tbl'><table><thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols) + "</tr></thead><tbody>"]
    for r in d.to_dict("records"):
        out.append("<tr>" + "".join(f"<td>{html.escape(_fmt(r[c], c))}</td>" for c in cols) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def md_inline_to_html(text: str) -> str:
    """Tiny Markdown subset: paragraphs, bullets, bold, code."""
    out, in_list = [], False
    for line in text.splitlines():
        esc = html.escape(line)
        esc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc)
        esc = re.sub(r"`(.+?)`", r"<code>\1</code>", esc)
        m = re.match(r"^(\s*)- (.*)", esc)
        if m:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{m.group(2)}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if esc.strip():
            out.append(f"<p>{esc}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:80]


def section_md(sec, fig_dir_rel: str | None = None) -> str:
    parts = [f"## {sec.title}", "", sec.md, ""]
    for name, df in sec.tables.items():
        parts += [f"### {name}", "", table_md(df), ""]
    for name in sec.figures:
        if fig_dir_rel is not None:
            parts += [f"![{name}]({fig_dir_rel}/{sec.id}_{name}.png)", ""]
    return "\n".join(parts)


def section_html(sec) -> str:
    parts = [f"<section id='{sec.id}'><h2>{html.escape(sec.title)}</h2>", md_inline_to_html(sec.md)]
    for name, df in sec.tables.items():
        parts += [f"<h3>{html.escape(name)}</h3>", table_html(df)]
    for name, png in sec.figures.items():
        b64 = base64.b64encode(png).decode()
        parts.append(f"<figure><img alt='{html.escape(name)}' src='data:image/png;base64,{b64}'/></figure>")
    parts.append("</section>")
    return "\n".join(parts)


CSS = """body{font-family:system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:16px;color:#1d1d1f;background:#fff}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:3px 6px;text-align:right}
th{background:#f3f4f6}td:first-child,th:first-child{text-align:left}.tbl{overflow-x:auto}
img{max-width:100%}code{background:#f3f4f6;padding:0 3px}
@media (prefers-color-scheme: dark){body{background:#111;color:#eee}th{background:#222}td,th{border-color:#333}code{background:#222}}"""


def header_md(meta: dict) -> str:
    return "\n".join([f"# Energy contract & home battery payback – {meta.get('label', '')}", "",
                      f"Run {meta.get('run_at', '')} · {meta.get('connection', '')} · "
                      f"price variant: {meta.get('price_variant', 'all')}", ""])


def render_markdown(sections, meta, fig_dir_rel="figures") -> str:
    return header_md(meta) + "\n".join(section_md(s, fig_dir_rel) for s in sections)


def render_html(sections, meta) -> str:
    body = "\n".join(section_html(s) for s in sections)
    title = "Battery payback report"
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{title}</title><style>{CSS}</style></head><body>"
            + md_inline_to_html(header_md(meta).replace("# ", "**", 1).replace(" –", "** –", 1)) + body + "</body></html>")


def write_outputs(sections, meta, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    for s in sections:
        for name, png in s.figures.items():
            with open(os.path.join(fig_dir, f"{s.id}_{name}.png"), "wb") as f:
                f.write(png)
        for name, df in s.tables.items():
            d = df[[c for c in df.columns if not str(c).startswith("_")]]
            d.to_csv(os.path.join(csv_dir, f"{s.id}_{slug(name)}.csv"), index=not isinstance(df.index, pd.RangeIndex))
    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(sections, meta))
    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_html(sections, meta))
    return {"markdown": md_path, "html": html_path, "csv_dir": csv_dir}
