"""Assemble the static site for GitHub Pages:

    python tools/build_site.py --out _site [--pyodide-dist path] [--no-ricks-data]

Copies web/, the battery_calc package, config.yaml and the data files, and
writes manifest.json so the site finds new price years automatically."""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from battery_calc.config import load_config  # noqa: E402

PYODIDE_PACKAGES = ["numpy", "pandas", "matplotlib", "pyyaml", "scipy"]
CORE = ["pyodide.js", "pyodide.mjs", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip",
        "pyodide-lock.json", "package.json"]


def rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def copy(src, out):
    dst = os.path.join(out, rel(src))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return rel(src)


def vendor_pyodide(dist, out):
    """Copy the Pyodide core and only the wheels the analysis needs."""
    with open(os.path.join(dist, "pyodide-lock.json")) as f:
        lock = json.load(f)["packages"]
    need, todo = set(), list(PYODIDE_PACKAGES)
    while todo:
        n = todo.pop().lower()
        if n in need or n not in lock:
            continue
        need.add(n)
        todo += lock[n].get("depends", [])
    dst = os.path.join(out, "pyodide")
    os.makedirs(dst, exist_ok=True)
    for f in CORE:
        if os.path.exists(os.path.join(dist, f)):
            shutil.copy2(os.path.join(dist, f), dst)
    for n in sorted(need):
        shutil.copy2(os.path.join(dist, lock[n]["file_name"]), dst)
    return sorted(need)


def batteries_for_manifest(cfg):
    import csv
    path = os.path.join(ROOT, cfg["paths"]["batteries_file"])
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [{"id": r["id"], "name": f"{r['brand']} {r['model']}", "price": r.get("price_nl_incl_vat", "")}
                for r in csv.DictReader(f)]


def last_update(paths):
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", *paths], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        return out or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_site")
    ap.add_argument("--pyodide-dist", default=None)
    ap.add_argument("--no-ricks-data", action="store_true",
                    help="leave Rick's P1 file and contract out of the public site")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    cfg = load_config(os.path.join(ROOT, "config.yaml"))
    for f in glob.glob(os.path.join(ROOT, "web", "*")):
        shutil.copy2(f, out)
    package = [copy(f, out) for f in sorted(glob.glob(os.path.join(ROOT, "battery_calc", "*.py")))]
    config = copy(os.path.join(ROOT, "config.yaml"), out)
    data = []
    globs = [cfg["paths"]["price_glob"], *cfg["paths"].get("extra_price_globs", []), "data/online/*.csv"]
    for g in globs:
        data += [copy(f, out) for f in sorted(glob.glob(os.path.join(ROOT, g)))]
    ricks = []
    if not a.no_ricks_data:
        for key in ("p1_file", "fixed_contract_file", "pv_file", "energieknl_file"):
            p = os.path.join(ROOT, cfg["paths"].get(key) or "")
            if os.path.isfile(p):
                ricks.append(copy(p, out))
    vendored = vendor_pyodide(a.pyodide_dist, out) if a.pyodide_dist else []
    manifest = {
        "package": package, "config": config, "data": sorted(set(data + ricks)),
        "price_files": [d for d in data if "stroomprijzen" in d or "dayahead" in d],
        "ricks_data": any(r.endswith(os.path.basename(cfg["paths"]["p1_file"])) for r in ricks),
        "connections": cfg["connection"]["options"], "default_connection": cfg["connection"]["default"],
        "online_updated_at": last_update(["data/online"]), "pyodide_packages": vendored,
        "batteries": batteries_for_manifest(cfg),
        "battery_shortlist": cfg.get("battery_shortlist", []),
        "blackfriday": cfg.get("blackfriday", {}), "kiln": cfg.get("kiln", {}),
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(out, ".nojekyll"), "w"):
        pass
    print(f"site in {out}: {len(package)} modules, {len(manifest['data'])} data files, "
          f"Rick's data {'included' if manifest['ricks_data'] else 'excluded'}, "
          f"{len(vendored)} Pyodide packages vendored")


if __name__ == "__main__":
    main()
