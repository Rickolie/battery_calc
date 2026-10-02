# battery_calc

Energy contract & home battery payback analysis for a Dutch household with
solar panels, built from `Specs.txt`. It uses the 15-minute P1 meter data to
answer three questions:

1. **Objective 1:** for each battery, the break-even price at which selling
   stored energy (or using it at home) beats its losses and wear.
2. **Objective 2:** which contract (fixed or dynamic) is cheapest without a
   battery: under 2026 rules (saldering), 2027–2029 rules (no saldering, 50%
   feed-in minimum) and 2030+ rules.
3. **Objective 3:** which battery × strategy × contract pays back fastest.

The same Python package runs as a CLI and, through Pyodide, as a static
website on GitHub Pages.

## Quick start

```bash
pip install -r requirements.txt
python -m battery_calc                      # default connection 3x25
python -m battery_calc --connection 1x35 --connection-margin 0.2
python -m battery_calc --quick              # headline price years only
python -m pytest -q                         # acceptance-criteria tests
```

The report is written to `results/report.md` and `results/report.html`, with
CSVs in `results/csv/` and charts in `results/figures/`.
`results/breakeven.csv` holds the per-battery break-even values (Objective 1).

Options: `--p1`, `--pv`, `--no-current-contract`, `--price-variant`,
`--feed-in-2030 0.3` (2030+ sensitivity), `--no-plots`, `--out`, `--config`.

## Inputs

All paths, column names and tariffs live in `config.yaml`.

| File | Status |
| --- | --- |
| `data/Energy_house.csv` | HomeWizard P1 export, 2025-11-01 → 2026-10-01. No gaps, duplicates or resets. October 2025 is filled synthetically and flagged. |
| `data/vast.txt` | Current fixed contract, parsed from the "Exclusief btw" section. |
| `data/jeroen_punt_nl_dynamische_stroomprijzen_jaar_YYYY.csv` | Day-ahead prices 2013–2026 (15-min, €/kWh kale; 2026 runs to early October). Loaded by glob using the `datum_utc` column, so a new year needs no code change. The loader also handles other layouts (separators, €/MWh, incl. taxes). |
| `data/dynamic_prices_energieknl.csv` | Holds the yearly delivery costs (`leveringskost`) of 23 dynamic suppliers, **not** day-ahead prices. The price cross-check therefore can't run; the file is shown in the report for reference. |
| `data/pv_production.csv` | Optional. Enables gross consumption and the solar-forecast Charge goal. |
| `data/online/contracts.csv` | Contract schema (section 3.1). Has one **template** dynamic contract (`verified=no`) so the pipeline runs. Replace it with real supplier terms. |
| `data/online/batteries.csv` | Battery schema (section 3.2). Three models with headline specs, all `verified=no`. **Prices are empty**, so payback can't be computed until they are filled in by hand or by the scraper (`tweakers_url` / `idealo_url` columns). Savings and cycles are computed anyway. |
| `data/online/taxes.csv` | Written by the scraper. When present, it overrides `taxes.years` in the config. |

Missing battery specs are left empty and flagged. The run uses the documented
defaults from `battery_defaults` in the config and marks those batteries as
"estimated".

## How it works

- `p1.py`: loads the meter data, converts counters to per-interval kWh, handles DST, gaps (≤1 h interpolated,
  longer ones flagged) and counter resets, and builds a 12-month profile with synthetic months (blend of the two
  weeks either side).
- `prices.py`: loads the price history, normalises it to €/kWh kale at 15 minutes, and replays each price year
  against the profile (same ISO week, weekday and quarter-hour).
- `costs.py`: one cost engine for every contract. It handles saldering on yearly totals, tariff blocks
  (T1/T2), feed-in cost tiers, the 50% minimum, and the rule that net export payment can't go below zero.
- `battery.py`: the per-interval physics (SoC limits, power cap from `--connection`, √RTE each way,
  standby, linear degradation) and these strategies:
  - **Self-consumption**: charges from surplus, discharges to cover load.
  - **Timed**: windows are searched on history.
  - **Dynamic** and **Dynamic + Sell**: per-day plans from day-ahead prices, gated by the battery's own
    break-even, with a persistence solar forecast and energy reserved for the day's priciest load (tuned on 2023–2025: beats self-consumption under every rule set). The break-even can be scaled 0.5–1.5.
  - **hbc_default / hbc_pv_first**: Home Battery Control's Dynamic strategy reproduced exactly (Extreme-Pair Matching, `min_delta`, hour caps, Low/Neutral/High sub-strategies; presets in `config.yaml`). These results are achievable in Home Assistant as-is; `dynamic` needs custom control for its peak reservation.
  - **Perfect foresight**: an LP upper bound (HiGHS).
- `breakeven.py`: Objective 1. Wear cost uses the lifetime limit (cycles, warranty throughput or calendar
  life), and the result is recomputed with the simulated cycles.
- `analysis.py`: the workflow of section 10. It emits each report section as soon as it is computed.
- `scrape.py`: the online-data scrapers (EnergyZero day-ahead prices, Belastingdienst energy tax, battery
  prices). A failed scrape never touches existing files.

## Website

`web/` contains the static site. `tools/build_site.py` assembles `_site/`, which holds:

- the package, config and data;
- `manifest.json`, so new price years are found automatically;
- a vendored Pyodide runtime, so no CDN is needed at run time.

The analysis runs in a Web Worker, and uploads never leave the browser.

The steps are:

1. Choose Rick's data or your own.
2. Upload a P1 file (validated immediately), an optional PV file and an optional contract form.
3. Choose the settings.
4. Read the results. Sections appear as they are computed and can be downloaded as HTML, Markdown or CSV.

Workflows:

- `.github/workflows/pages.yml` runs the tests and deploys to GitHub Pages on every push to `main`.
- `.github/workflows/scrape.yml` scrapes weekly (or on demand), commits `data/online/` and redeploys.

**Privacy:** the deploy publishes Rick's P1 file and contract with the site. Add `--no-ricks-data` to the
build step in `pages.yml` to keep them out (an open question in the spec).

## Acceptance criteria status

| Criterion | Status |
| --- | --- |
| Current-contract cost within 3% of an annual bill | Open: no bill in the repo. Compare `total` (2026 rules) in section 2. |
| Energy balance closes every interval | Tested (`test_energy_balance_closes_every_interval`) and reported in section 6. |
| RTE 100%, standby 0, zero spread → zero Dynamic savings | Tested. |
| Every number traces to a source | Contracts, batteries and taxes carry `source_url` and `retrieved_at`. Hand-entered values are flagged `verify`/`verified=no`. |
| Pre- and post-2027 rules shown | All three regimes appear in every table. |
| RTE 100%, infinite life → break-even = charge price | Tested. |
| Lifetime limit shown per battery | `lifetime_limit` column. |
| `--connection 1x35` vs `3x25` changes caps and filters 3-phase | Tested. |
| Both price-year analyses shown | "Last 3 years (headline)" and "Full history" tables, plus year groups. |
| Synthetic October flagged, 11-real-months check next to totals | `real_months_only` columns. |
| Website = CLI | Same code. Verified in headless Chromium: identical numbers for Rick's data. |
| Example upload runs without external requests | Verified: 0 requests outside the site. |
| Missing month in an upload is shown and filled | Tested, and shown in the upload validation step. |

## Known limitations

- Battery prices are not in the repo yet, so payback can't be ranked; savings, cycles and break-even inputs are computed.
- Dynamic contracts use a template until real supplier terms are added to `contracts.csv`.
- Grid charges are configured only up to 3x25 (from `vast.txt`). Larger connections fall back with a warning.
  They are equal for every supplier, so rankings are unaffected.
- The peak-shave strategy is not simulated. It only pays with a capacity tariff, which is a scenario flag that
  is off by default.
- Contract scrapers per supplier are not implemented, because supplier pages differ too much to parse
  reliably without testing them. `contracts.csv` is edited by hand.
- Old price files that only contain all-in prices are converted back to kale with the configured energy tax.
  Add historic tax years to the config for exact conversion.
