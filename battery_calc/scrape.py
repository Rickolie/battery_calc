"""Scrapers for online data (section 3). Run weekly by GitHub Actions:

    python -m battery_calc.scrape [--only prices,taxes,batteries]

Every scraper writes to data/online/*.csv with `source_url` and
`retrieved_at`. A scraper that fails leaves its file untouched, so a broken
site never silently changes results. All files can also be edited by hand;
hand-entered values are only replaced by a successful scrape of that field.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.request

import pandas as pd

from .config import load_config, resolve

UA = "Mozilla/5.0 (battery_calc scraper; +https://github.com/)"
TODAY = dt.date.today().isoformat()


def fetch(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "nl,en;q=0.8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def eur(text: str) -> float:
    """'€ 1.234,56' or '1234.56' -> 1234.56"""
    t = text.replace("€", "").replace("\xa0", "").replace(" ", "").strip().rstrip("-").rstrip(",")
    if "," in t:
        t = t.replace(".", "").replace(",", ".")
    return float(t)


# ---------------------------------------------------------------- day-ahead prices

def scrape_dayahead(cfg: dict, year: int | None = None) -> str:
    """Current-year day-ahead prices (kale, €/kWh) from the EnergyZero public
    API. The jeroen.nl files stay the main source; these fill recent months."""
    year = year or dt.date.today().year
    out_dir = resolve(cfg, cfg["paths"]["online_dir"])
    path = os.path.join(out_dir, f"dayahead_energyzero_{year}.csv")
    rows = []
    start = dt.date(year, 1, 1)
    end = min(dt.date(year, 12, 31), dt.date.today() + dt.timedelta(days=1))
    d = start
    while d <= end:
        chunk_end = min(d + dt.timedelta(days=30), end)
        url = ("https://api.energyzero.nl/v1/energyprices?"
               f"fromDate={d.isoformat()}T00:00:00.000Z&tillDate={chunk_end.isoformat()}T23:59:59.999Z"
               "&interval=4&usageType=1&inclBtw=false")
        data = json.loads(fetch(url))
        for p in data.get("Prices", []):
            rows.append({"datum_utc": p["readingDate"], "prijs_excl_belastingen": p["price"],
                         "source_url": "https://api.energyzero.nl/v1/energyprices", "retrieved_at": TODAY})
        d = chunk_end + dt.timedelta(days=1)
    if not rows:
        raise RuntimeError("no prices returned")
    pd.DataFrame(rows).drop_duplicates("datum_utc").to_csv(path, index=False)
    return f"{len(rows)} prices → {path}"


# ---------------------------------------------------------------- energy tax

def scrape_taxes(cfg: dict) -> str:
    """Energy tax (first bracket) and tax reduction from Belastingdienst."""
    url = cfg.get("taxes", {}).get("years", {}).get(max(cfg["taxes"]["years"]), {}).get("source_url")
    html = fetch(url)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    out = []
    # Pattern: "Elektriciteit ... 0 tot en met 10.000 kWh € 0,09161" per year column
    m = re.search(r"0 tot en met 10\.000 kWh\s*((?:€\s*[\d,]+\s*){1,4})", text)
    heads = re.findall(r"Tarief\s*(20\d\d)", text)
    if not m:
        raise RuntimeError("energy tax table not found")
    vals = [eur(v) for v in re.findall(r"€\s*([\d,]+)", m.group(1))]
    red = re.search(r"[Vv]ermindering energiebelasting\s*((?:€\s*[\d\.,]+\s*){1,4})", text)
    reds = [eur(v) for v in re.findall(r"€\s*([\d\.,]+)", red.group(1))] if red else []
    for i, v in enumerate(vals):
        year = int(heads[i]) if i < len(heads) else None
        if year is None:
            continue
        out.append({"year": year, "energy_tax_eur_kwh": v,
                    "tax_reduction_eur_year": reds[i] if i < len(reds) else None,
                    "source_url": url, "retrieved_at": TODAY})
    if not out:
        raise RuntimeError("could not match years to rates")
    path = os.path.join(resolve(cfg, cfg["paths"]["online_dir"]), "taxes.csv")
    pd.DataFrame(out).to_csv(path, index=False)
    return f"{len(out)} tax years → {path}"


# ---------------------------------------------------------------- battery prices

PRICE_PATTERNS = [
    r"(?:vanaf|laagste prijs|Lowest price|ab)\s*€\s*([\d\.]+,\d{2}|[\d\.]+,-)",
    r"\"lowPrice\"\s*:\s*\"?([\d\.]+)",
    r"\"price\"\s*:\s*\"?([\d\.]+)",
    r"€\s*([\d\.]+,\d{2})",
]


def find_price(html: str) -> float | None:
    for pat in PRICE_PATTERNS:
        m = re.search(pat, html, flags=re.I)
        if m:
            try:
                return eur(m.group(1))
            except ValueError:
                continue
    return None


def _match(label: str, match: str) -> bool:
    label = label.lower().replace(",", ".")
    return all(tok.strip().lower() in label for tok in match.split("&") if tok.strip())


def shop_price(page: str, url: str, match: str) -> float | None:
    """Price incl. VAT of the configuration whose label contains every
    `&`-separated token of `match` (lowest if several), for the shops whose
    page structure we know: thuisbatterij.nl (WooCommerce variations) and
    123accu.nl (product blocks)."""
    import html as htmllib
    prices = []
    if "thuisbatterij.nl" in url and match.startswith("ajax:"):
        # Too many variations to embed: ask WooCommerce for one exact combination.
        pid = re.search(r'data-product_id="(\d+)"', page)
        if not pid:
            return None
        data = {"product_id": pid.group(1)}
        for part in match[5:].split(";"):
            k, _, v = part.partition("=")
            data[k.strip()] = v.strip()
        import urllib.parse
        req = urllib.request.Request("https://thuisbatterij.nl/?wc-ajax=get_variation",
                                     data=urllib.parse.urlencode(data).encode(), headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            v = json.loads(r.read() or b"null")
        return float(v["display_price"]) if isinstance(v, dict) and v.get("display_price") else None
    if "thuisbatterij.nl" in url:
        m = re.search(r'data-product_variations="([^"]*)"', page)
        if m:
            for v in json.loads(htmllib.unescape(m.group(1))) or []:
                label = " ".join(str(x).replace("-", " ") for x in v.get("attributes", {}).values())
                if _match(label, match) and v.get("is_in_stock", True):
                    prices.append(float(v["display_price"]))
    elif "123accu.nl" in url:
        text = htmllib.unescape(re.sub(r"<[^>]+>", "\n", re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)))
        lines = [l.replace("\xa0", " ").strip() for l in text.splitlines() if l.strip()]
        name = ""
        for i, l in enumerate(lines):
            if re.search(r"kWh (plug-in thuisbatterij|thuisbatterij set|uitbreidingsbatterij)|Basismodule\)", l) and len(l) < 110:
                name = l
            if l.startswith("€") and i + 2 < len(lines) and lines[i + 1].startswith("€") and "Exclusief" in lines[i + 2]:
                if _match(name or lines[0], match):
                    prices.append(eur(l.replace("€", "")))
                name = ""
    else:
        p = find_price(page)
        return p
    return min(prices) if prices else None


def scrape_battery_prices(cfg: dict) -> str:
    """Updates `price_nl_incl_vat` from the shop pages in `shop_urls`
    (`url|match` entries separated by spaces; the lowest price wins) and the
    optional `tweakers_url`, `price_de_excl_vat` from `idealo_url`, and tracks
    the lowest NL price seen."""
    path = resolve(cfg, cfg["paths"]["batteries_file"])
    df = pd.read_csv(path, dtype=str).fillna("")
    for col in ("tweakers_url", "idealo_url", "shop_urls"):
        if col not in df.columns:
            df[col] = ""
    n, errors, cache = 0, [], {}

    def get(url):
        if url not in cache:
            cache[url] = fetch(url)
        return cache[url]

    for i, r in df.iterrows():
        found = []
        entries = [e for e in re.split(r"\s+(?=https?://)", r["shop_urls"].strip()) if e]
        for entry in entries:
            url, _, match = entry.partition("|")
            try:
                p = shop_price(get(url), url, match if match.startswith("ajax:") else match.replace("_", " "))
                if p:
                    found.append((p, url))
                else:
                    errors.append(f"{r['id']}: no price matching '{match}' on {url}")
            except Exception as e:
                errors.append(f"{r['id']} {url}: {e}")
        if r["tweakers_url"]:
            try:
                p = find_price(get(r["tweakers_url"]))
                if p:
                    found.append((p, r["tweakers_url"]))
            except Exception as e:
                errors.append(f"{r['id']} tweakers: {e}")
        if found:
            p, url = min(found)
            df.at[i, "price_nl_incl_vat"] = f"{p:.2f}"
            df.at[i, "source_url"] = url
            low = float(r["price_nl_lowest_incl_vat"]) if r["price_nl_lowest_incl_vat"] else None
            if low is None or p < low:
                df.at[i, "price_nl_lowest_incl_vat"] = f"{p:.2f}"
                df.at[i, "price_nl_lowest_date"] = TODAY
            df.at[i, "retrieved_at"] = TODAY
            n += 1
        if r["idealo_url"]:
            try:
                p = find_price(get(r["idealo_url"]))
                if p:
                    df.at[i, "price_de_excl_vat"] = f"{p:.2f}"
            except Exception as e:
                errors.append(f"{r['id']} idealo: {e}")
    if n:
        df.to_csv(path, index=False)
    return f"{n} batteries priced" + (f"; errors: {'; '.join(errors)}" if errors else "")


SCRAPERS = {"prices": scrape_dayahead, "taxes": scrape_taxes, "batteries": scrape_battery_prices}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="battery_calc.scrape")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only", default=",".join(SCRAPERS))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    log = []
    for name in args.only.split(","):
        try:
            msg = SCRAPERS[name](cfg)
            log.append({"scraper": name, "ok": True, "message": msg, "at": TODAY})
        except Exception as e:  # a failed scrape never touches the existing file
            log.append({"scraper": name, "ok": False, "message": str(e), "at": TODAY})
    for entry in log:
        print(("OK   " if entry["ok"] else "FAIL ") + f"{entry['scraper']}: {entry['message']}")
    with open(os.path.join(resolve(cfg, cfg["paths"]["online_dir"]), "scrape_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
