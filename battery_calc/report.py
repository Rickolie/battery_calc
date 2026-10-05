"""Markdown / HTML / CSV output for a run (shared by CLI and website)."""
from __future__ import annotations

import base64
import html
import os
import re

import numpy as np
import pandas as pd

from .columns import describe_table
from .summaries import summarize


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
    """Tiny Markdown subset: paragraphs, bullets, bold, italic, code, ##/### headings.
    A `#### Title` line starts a collapsible block (closed) that runs to the next
    heading or the end, so long explanations stay out of the way."""
    out, in_list, in_more = [], False, False
    for line in text.splitlines():
        mm = re.match(r"^#### (.*)", line)
        if mm or (in_more and re.match(r"^#{2,3} ", line)):
            if in_list:
                out.append("</ul>")
                in_list = False
            if in_more:
                out.append("</details>")
                in_more = False
            if mm:
                title = re.sub(r"`(.+?)`", r"<code>\1</code>", html.escape(mm.group(1)))
                out.append(f"<details class='more'><summary>{title}</summary>")
                in_more = True
                continue
        esc = html.escape(line)
        esc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc)
        esc = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"<em>\1</em>", esc)
        esc = re.sub(r"`(.+?)`", r"<code>\1</code>", esc)
        hm = re.match(r"^(#{2,4}) (.*)", esc)
        if hm:
            if in_list:
                out.append("</ul>")
                in_list = False
            lvl = min(len(hm.group(1)) + 1, 5)
            out.append(f"<h{lvl}>{hm.group(2)}</h{lvl}>")
            continue
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
    if in_more:
        out.append("</details>")
    return "\n".join(out)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:80]


def shown_columns(df: pd.DataFrame) -> list[str]:
    d = df.head(1)
    if not isinstance(d.index, pd.RangeIndex):
        d = d.reset_index()
    return [str(c) for c in d.columns if not str(c).startswith("_")]


def columns_md(df: pd.DataFrame, name: str) -> str:
    items = describe_table(shown_columns(df), name)
    if not items:
        return ""
    return "\n".join(["**Columns**", ""] + [f"- `{c}`: {d}" for c, d in items])


def columns_html(df: pd.DataFrame, name: str) -> str:
    items = describe_table(shown_columns(df), name)
    if not items:
        return ""
    rows = "".join(f"<dt>{html.escape(c)}</dt><dd>{html.escape(d)}</dd>" for c, d in items)
    return f"<details class='cols' open><summary>Column descriptions</summary><dl>{rows}</dl></details>"


def section_key_finding(sec) -> str:
    """One line shown next to the (collapsed) section title."""
    if getattr(sec, "summary", ""):
        return sec.summary
    for name, df in sec.tables.items():
        s = summarize(name, df)
        if s:
            return s
    return ""


def section_md(sec, fig_dir_rel: str | None = None) -> str:
    parts = [f"## {sec.title}", ""]
    key = section_key_finding(sec)
    if key:
        parts += [f"**Key finding:** {key}", ""]
    parts += [sec.md, ""]
    for name, df in sec.tables.items():
        if name in getattr(sec, "csv_only", set()):
            parts += [f"_{name}: {len(df)} rows, in the CSV download._", ""]
            continue
        s = summarize(name, df)
        parts += [f"### {name}", ""] + ([f"_Finding: {s}_", ""] if s else []) + [table_md(df), "",
                                                                                columns_md(df, name), ""]
    for name in sec.figures:
        if fig_dir_rel is not None:
            parts += [f"![{name}]({fig_dir_rel}/{sec.id}_{name}.png)", ""]
    only_html = [s.get("title", n) for n, s in (getattr(sec, "interactive", {}) or {}).items() if n not in sec.figures]
    if only_html:
        parts += ["_Interactive charts in report.html: " + "; ".join(only_html) + "._", ""]
    return "\n".join(parts)


def chart_html(spec: dict) -> str:
    """Placeholder + data for an interactive (zoomable) chart, rendered by static/charts.js."""
    import json
    data = json.dumps(spec, separators=(",", ":"), allow_nan=False, default=float).replace("</", "<\\/")
    note = ("<p class='zoomhint'>Hover a bar, point or line to see the values.</p>" if spec.get("kind") else
            "<p class='zoomhint'>Drag across the chart to zoom in, double-click to zoom out, click a legend item to hide it.</p>")
    return f"<div class='ichart'></div><script type='application/json'>{data}</script>{note}"


def section_html(sec, open_: bool = False) -> str:
    """A section as a collapsible block (closed by default) whose title line shows
    the key finding; every table is collapsible too, with its own finding."""
    key = section_key_finding(sec)
    head = (f"<summary><h2>{html.escape(sec.title)}</h2>"
            + (f"<p class='keyfind'>{html.escape(key)}</p>" if key else "") + "</summary>")
    parts = [f"<section id='{sec.id}'><details class='sec'{' open' if open_ else ''}>{head}",
             "<div class='secbody'>", md_inline_to_html(sec.md)]
    interactive = getattr(sec, "interactive", {}) or {}
    groups: dict = {}
    for name, spec in interactive.items():
        g = spec.get("group")
        if g:
            groups.setdefault(g, []).append(spec)
        else:
            parts.append(chart_html(spec))
    for g, specs in groups.items():
        parts.append(f"<details class='more'><summary>{html.escape(g)}</summary>"
                     + "".join(chart_html(s) for s in specs) + "</details>")
    for name, df in sec.tables.items():
        if name in getattr(sec, "csv_only", set()):
            parts.append(f"<p class='zoomhint'>{html.escape(name)}: {len(df)} rows – see the CSV download below.</p>")
            continue
        s = summarize(name, df)
        parts.append(f"<details class='tblw'><summary><span class='tname'>{html.escape(name)}</span>"
                     + (f" – {html.escape(s)}" if s else f" – {len(df)} rows") + "</summary>"
                     + table_html(df) + columns_html(df, name) + "</details>")
    for name, png in sec.figures.items():
        if name in interactive:
            continue          # shown as a zoomable chart above
        b64 = base64.b64encode(png).decode()
        parts.append(f"<figure><img alt='{html.escape(name)}' src='data:image/png;base64,{b64}'/></figure>")
    parts.append("</div></details></section>")
    return "\n".join(parts)


TOGGLE_BAR = ("<p class='togglebar'><button type='button' onclick='setAllDetails(true)'>Expand all</button> "
              "<button type='button' onclick='setAllDetails(false)'>Collapse all</button> "
              "<span class='zoomhint'>Click a section or table title to open it.</span></p>")


def _static(name: str) -> str:
    with open(os.path.join(os.path.dirname(__file__), "static", name), encoding="utf-8") as f:
        return f.read()


def order_sections(sections):
    """The recommendation comes first in the final report."""
    return sorted(sections, key=lambda s: 0 if s.id == "advice" else 1)


CSS = """body{font-family:system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:16px;color:#1d1d1f;background:#fff}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:3px 6px;text-align:right}
th{background:#f3f4f6}td:first-child,th:first-child{text-align:left}.tbl{overflow-x:auto}
img{max-width:100%}
.ichart{margin:8px 0}.zoomhint{font-size:11px;color:#777;margin:0 0 12px}
.cols{font-size:12px;margin:4px 0 14px}.cols summary{cursor:pointer;color:#555}
.cols dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 12px;margin:6px 0}
.cols dt{font-family:monospace;font-weight:600}.cols dd{margin:0}code{background:#f3f4f6;padding:0 3px}
details.sec>summary{cursor:pointer;padding:6px 0}details.sec>summary h2{display:inline;font-size:1.2rem}
.keyfind{margin:2px 0 0 18px;color:#555;font-size:.9rem}
details.tblw{margin:6px 0}details.tblw>summary{cursor:pointer;font-size:.9rem}.tname{font-family:monospace;font-weight:600}
.togglebar button{margin-right:6px}
details.more{margin:6px 0 10px;border-left:3px solid #e4e3df;padding-left:10px}details.more>summary{cursor:pointer;font-weight:600}
@media (prefers-color-scheme: dark){body{background:#111;color:#eee}th{background:#222}td,th{border-color:#333}code{background:#222}}"""


def header_md(meta: dict) -> str:
    return "\n".join([f"# Energy contract & home battery payback – {meta.get('label', '')}", "",
                      f"Run {meta.get('run_at', '')} · {meta.get('connection', '')} · "
                      f"price variant: {meta.get('price_variant', 'all')}", ""])


def render_markdown(sections, meta, fig_dir_rel="figures") -> str:
    return header_md(meta) + "\n".join(section_md(s, fig_dir_rel) for s in order_sections(sections))


def render_html(sections, meta) -> str:
    body = "\n".join(section_html(s) for s in order_sections(sections))
    title = "Battery payback report"
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{title}</title><style>{CSS}{_static('uPlot.min.css')}</style>"
            f"<script>{_static('uPlot.iife.min.js')}</script><script>{_static('charts.js')}</script></head><body>"
            + md_inline_to_html(header_md(meta).replace("# ", "**", 1).replace(" –", "** –", 1)) + TOGGLE_BAR + body
            + "</body></html>")


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
