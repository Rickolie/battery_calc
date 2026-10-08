"""Precompute the website's default report (Rick's data, default settings), so the site
shows it immediately instead of running the analysis in the browser first.

    python tools/precompute.py --out _site/precomputed

Writes sections.json (what the browser would have received, section by section),
report.html and report.md. Runs the same code path as the website (web_api.run).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

from battery_calc import web_api  # noqa: E402
from battery_calc.config import load_config  # noqa: E402


def default_settings(cfg: dict) -> dict:
    """The same settings the website's form starts with (web/index.html defaults)."""
    d = cfg.get("battery_defaults", {})
    bf = cfg.get("blackfriday", {}) or {}
    return {
        "use_ricks": True, "label": "Rick's data", "contract": None,
        "connection": cfg["connection"]["default"], "margin": float(cfg["connection"]["margin"]),
        "price_variant": "all", "feed_in_2030": 0.0, "quick": False,
        "battery_set": cfg.get("battery_set", "all"), "chosen_battery": "",
        "use_german_prices": bool(d.get("use_german_prices", False)),
        "blackfriday": {"discount_nl": bf.get("discount_nl"), "discount_de": bf.get("discount_de"),
                        "estimates": bool(bf.get("estimates", True))},
        "kiln": {}, "solar": {}, "extension": None,
    }


def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_site/precomputed")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    os.chdir(ROOT)
    cfg = load_config("config.yaml")
    settings = default_settings(cfg)
    sections = []
    t0 = time.time()

    def on_section(js: str):
        sec = json.loads(js)
        sections.append(sec)
        print(f"  [{time.time() - t0:5.0f}s] {sec['title']}", flush=True)

    result = json.loads(web_api.run(json.dumps(settings), on_section=on_section, config_path="config.yaml"))
    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "commit": commit(),
            "settings": settings, "seconds": round(time.time() - t0)}
    with open(os.path.join(out, "sections.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "sections": sections}, f, separators=(",", ":"))
    with open(os.path.join(out, "report.html"), "w", encoding="utf-8") as f:
        f.write(result["html"])
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(result["markdown"])
    size = os.path.getsize(os.path.join(out, "sections.json")) / 1e6
    print(f"precomputed {len(sections)} sections in {meta['seconds']} s ({size:.1f} MB) -> {out}")


if __name__ == "__main__":
    main()
