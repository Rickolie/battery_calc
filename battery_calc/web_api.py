"""Entry points for the static website (Pyodide). The browser writes the
site's files and any uploads into the in-memory file system and calls these
functions; the analysis code is exactly the one the CLI runs."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from .analysis import Analysis, Inputs, Options
from .config import load_config
from .contracts import contract_from_form
from .p1 import FLAG_SYNTH, build_profile_year, load_p1
from .report import render_html, render_markdown, section_html, slug


def _json(o):
    return json.dumps(o, default=lambda x: None if isinstance(x, float) and np.isnan(x) else str(x))


def validate_p1(path: str, config_path: str = "config.yaml") -> str:
    """Section 2 checks on an upload, shown before the visitor continues."""
    cfg = load_config(config_path)
    try:
        df, rep = load_p1(path, cfg, label=os.path.basename(path))
        pr = build_profile_year(df, rep, cfg)
    except Exception as e:
        return _json({"ok": False, "error": str(e)})
    real = pr["flag"] != FLAG_SYNTH
    return _json({
        "ok": True,
        "first": f"{rep.first:%Y-%m-%d %H:%M}", "last": f"{rep.last:%Y-%m-%d %H:%M}",
        "rows": rep.rows, "duplicates": rep.duplicates, "resets": len(rep.resets),
        "interpolated_gaps": len(rep.interpolated_gaps),
        "long_gaps": [f"{a:%Y-%m-%d %H:%M} → {b:%Y-%m-%d %H:%M}" for a, b in rep.long_gaps[:20]],
        "window": f"{rep.window[0]:%Y-%m-%d} – {rep.window[1]:%Y-%m-%d}",
        "missing_months": rep.missing_months, "low_confidence": rep.low_confidence,
        "import_kwh": round(float(pr["imp"].sum()), 1), "export_kwh": round(float(pr["exp"].sum()), 1),
        "import_real_kwh": round(float(pr.loc[real, "imp"].sum()), 1),
        "export_real_kwh": round(float(pr.loc[real, "exp"].sum()), 1),
    })


def run(settings_json: str, on_section=None, config_path: str = "config.yaml") -> str:
    """Run the full analysis. `on_section` gets one JSON string per section."""
    s = json.loads(settings_json)
    cfg = load_config(config_path)
    cfg["paths"]["results_dir"] = s.get("results_dir", "results")
    inputs = Inputs(label=s.get("label", "Rick's data"))
    if not s.get("use_ricks", True):
        inputs.p1 = s["p1_path"]
        inputs.pv = s.get("pv_path")
        form = s.get("contract")
        if form:
            inputs.fixed_contract = contract_from_form(form)
        else:
            inputs.no_current_contract = True
    if s.get("use_german_prices") is not None:
        cfg.setdefault("battery_defaults", {})["use_german_prices"] = bool(s["use_german_prices"])
    bf = s.get("blackfriday") or {}
    for k in ("discount_nl", "discount_de"):
        if bf.get(k) is not None:
            cfg.setdefault("blackfriday", {})[k] = float(bf[k])
    if bf.get("estimates") is not None:
        cfg.setdefault("blackfriday", {})["estimates"] = bool(bf["estimates"])
    kiln = {k: float(v) for k, v in (s.get("kiln") or {}).items() if v not in (None, "")}
    for k, v in (s.get("solar") or {}).items():
        if v not in (None, ""):
            cfg.setdefault("solar", {})[k] = v if k == "install_date" else float(v)
    opts = Options(connection=s.get("connection", cfg["connection"]["default"]),
                   margin=float(s.get("margin", cfg["connection"]["margin"])),
                   price_variant=s.get("price_variant", "all"), quick=bool(s.get("quick", False)),
                   plots=True, feed_in_2030=s.get("feed_in_2030"),
                   chosen_battery=s.get("chosen_battery") or None, battery_set=s.get("battery_set") or None,
                   kiln=kiln or None)

    def emit(sec):
        if on_section is None:
            return
        on_section(_json({"id": sec.id, "title": sec.title, "html": section_html(sec),
                          "csv": {f"{sec.id}_{slug(k)}.csv": v[[c for c in v.columns if not str(c).startswith('_')]]
                                  .to_csv(index=not isinstance(v.index, pd.RangeIndex))
                                  for k, v in sec.tables.items()}}))

    an = Analysis(cfg, inputs, opts, on_section=emit, log=None)
    an.no_write = True
    sections = an.run()
    meta = {"label": inputs.label, "run_at": time.strftime("%Y-%m-%d %H:%M"), "connection": an.conn.describe(),
            "price_variant": opts.price_variant}
    return _json({"markdown": render_markdown(sections, meta, fig_dir_rel=None),
                  "html": render_html(sections, meta)})
