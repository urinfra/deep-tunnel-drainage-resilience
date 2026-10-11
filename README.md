# Deep tunnel resilience framework - code repository

Python code for the manuscript **"Deep tunnel drainage systems redistribute urban flood resilience under climate extremes"** (Communications Sustainability, COMMSSUSTAIN-26-0508-A).

## Structure

- Repository root - the parameterized resilience modelling framework and analysis chain:
  - `resilience_framework.py` - the DTFRI four-dimensional resilience metric and framework classes.
  - `phase2_quantitative_analysis.py`, `phase2_sensitivity_analysis.py`, `expand_15cases.py`, `generate_ed_tables.py`, `run_resilience_analysis.py` - quantitative analysis and table generation.
  - `calibrate_gz_model.py`, `analyze_gz_data.py` - Guangzhou calibration against the operational monitoring record.
  - `cmip6_extreme_precipitation.py`, `verified_public_data.py`, `validate_synthetic_equity.py`, `fig3_spatial_nature.py`, `figure_editor.py`, `update_figures.py`, `export_for_submission.py`, `regenerate_sensitivity*.py` - climate projections, data validation, and figure/data export.
- `figures/` - scripts that generate Figs. 1-5 and Supplementary Figs. S1-S15 (one script per figure), plus `generate_all.py`.

## Data

The source data and numerical results underlying the figures are deposited in Zenodo (record link to be inserted; see the Data Availability statement of the manuscript). Download the data files from the Zenodo record into this repository before running the figure scripts (the scripts expect the xlsx/csv files in the same directory as the script or in a `data/` subdirectory).

**Guangzhou operational monitoring data**: event-level monitoring data for the Guangzhou Donghaochong deep tunnel (2022-2025) are confidential (Guangzhou Water Affairs Bureau data-sharing agreement) and are **not** included. Aggregate statistics (n = 579 events, P50/P95/P99 and maximum peak storage) are provided in the Zenodo deposit (`Fig1_RFC_Curves.xlsx`, sheet `Guangzhou_Observed_Summary`).

## Requirements

Python 3.14. See `requirements.txt`.

## Licence

MIT License (see LICENSE).
