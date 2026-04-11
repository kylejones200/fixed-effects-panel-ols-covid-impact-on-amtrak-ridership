# Fixed Effects Panel OLS: Measuring COVID's Regional Impact on Amtrak Ridership

*Using linearmodels PanelOLS with region × COVID interaction terms to isolate the causal effect of the pandemic on US rail travel*

---

Amtrak ridership collapsed in 2020. That much is obvious from any chart. The harder question is whether the collapse was uniform — or whether COVID hit some regions harder than others, and whether the recovery followed the same pattern in the Northeast corridor as it did in the West.

Ordinary regression can't answer this cleanly. Stations differ systematically: a Boston South Station handles orders of magnitude more passengers than a rural stop in Montana. If you ignore these fixed differences and pool the data, your COVID coefficient absorbs station-level heterogeneity and the estimate is biased.

Fixed effects panel OLS solves this by estimating a separate intercept for every station, then identifying the COVID effect only from within-station variation over time.

## The Model

```
Ridership_it = α_i + β_1·t + β_2·post_covid + β_3·(post_covid × region) + ε_it
```

- `α_i` — station fixed effect (absorbs all time-invariant station characteristics)
- `t` — linear time trend (secular growth in rail demand)
- `post_covid` — indicator for year ≥ 2020 (baseline: Midwest)
- `post_covid × region` — interaction terms measuring differential regional impact

## Data

Public Amtrak ridership data by station and year (2005–2022), sourced from:
```
https://raw.githubusercontent.com/kylejones200/time_series/refs/heads/main/data/amtrak_ridership_time_series_data.csv
```

Stations are assigned to four Census regions: Northeast, Midwest, South, West.

## Key Results

| Parameter | Estimate | p-value |
|-----------|----------|---------|
| `post_covid` (Midwest baseline) | −61,450 | < 0.001 |
| `t` (annual trend) | +4,136 | < 0.001 |
| `post_covid × Northeast` | see model | < 0.001 |
| `post_covid × South` | see model | < 0.001 |
| `post_covid × West` | see model | < 0.001 |

The baseline Midwest drop of −61,450 annual riders per station is large but the Northeast interaction reveals an even sharper collapse — consistent with the density of the NEC corridor and higher baseline ridership.

## Quickstart

```bash
pip install linearmodels pandas matplotlib numpy requests
python amtrak_panel_ols.py
```

Outputs: `amtrak_ridership_trends.png`, `amtrak_panel_region_effects.png`

## Files

| File | Description |
|------|-------------|
| `amtrak_panel_ols.py` | Main script: loads data, fits PanelOLS, generates figures |
| `data_io.py` | Thin CSV loader utility |
