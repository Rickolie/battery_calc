"""Command line: python -m battery_calc --connection 3x25"""
from __future__ import annotations

import argparse
import sys
import time

from .analysis import Analysis, Inputs, Options
from .config import load_config, resolve
from .report import write_outputs


def build_parser(cfg_default_conn="3x25"):
    p = argparse.ArgumentParser(prog="battery_calc", description="Energy contract & home battery payback analysis")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--connection", default=None, help="grid connection, e.g. 1x25, 1x35, 3x25 (default from config)")
    p.add_argument("--connection-margin", type=float, default=None, help="safety margin on the per-phase cap (0.2 = 20%%)")
    p.add_argument("--p1", default=None, help="P1 CSV (default from config)")
    p.add_argument("--pv", default=None, help="optional PV production CSV")
    p.add_argument("--no-current-contract", action="store_true")
    p.add_argument("--price-variant", default="all",
                   choices=["all", "NL current", "NL lowest-ever", "DE 0% VAT (scenario)"])
    p.add_argument("--feed-in-2030", type=float, default=None, help="minimum feed-in fraction from 2030 (default 0)")
    p.add_argument("--quick", action="store_true", help="headline years only")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--out", default=None, help="results directory (default from config)")
    p.add_argument("--batteries", default=None, help="all, shortlist, or comma-separated battery ids")
    p.add_argument("--battery", default=None, help="chosen battery id for the earnings split and the kiln objective")
    p.add_argument("--kiln-hours", type=float, default=None, help="hours of a firing to maximum temperature")
    p.add_argument("--kiln-duty", type=float, default=None, help="average share of rated power drawn during a firing")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    conn = args.connection or cfg.get("connection", {}).get("default", "3x25")
    margin = args.connection_margin if args.connection_margin is not None else cfg.get("connection", {}).get("margin", 0.2)
    if args.out:
        cfg["paths"]["results_dir"] = args.out
    opts = Options(connection=conn, margin=margin, price_variant=args.price_variant, quick=args.quick,
                   plots=not args.no_plots, feed_in_2030=args.feed_in_2030, chosen_battery=args.battery,
                   battery_set=args.batteries,
                   kiln={k: v for k, v in (("firing_hours", args.kiln_hours), ("avg_duty", args.kiln_duty))
                         if v is not None} or None)
    inputs = Inputs(p1=args.p1, pv=args.pv, no_current_contract=args.no_current_contract,
                    label="Rick's data" if not args.p1 else f"P1 file {args.p1}")

    def on_section(sec):
        print(f"[done] {sec.title}", flush=True)

    t0 = time.time()
    an = Analysis(cfg, inputs, opts, on_section=on_section, log=lambda m: print(m, flush=True))
    print(an.conn.describe(), flush=True)
    sections = an.run()
    meta = {"label": inputs.label, "run_at": time.strftime("%Y-%m-%d %H:%M"), "connection": an.conn.describe(),
            "price_variant": args.price_variant}
    out = write_outputs(sections, meta, resolve(cfg, cfg["paths"]["results_dir"]))
    print(f"Report: {out['markdown']} and {out['html']} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
