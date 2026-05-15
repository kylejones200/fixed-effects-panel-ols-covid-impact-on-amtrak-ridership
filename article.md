# Fixed Effects Panel OLS: Measuring COVID's Regional Impact on Amtrak Ridership

*Using linearmodels PanelOLS with region × COVID interaction terms to isolate the causal effect of the pandemic on US rail travel*

---

Amtrak ridership collapsed in 2020. That much is obvious from any chart. The harder question is whether the collapse was uniform — or whether COVID hit some regions harder than others, and whether the recovery followed the same pattern in the Northeast corridor as it did in the West.

Ordinary regression cannot answer this cleanly. Stations differ systematically: a Boston South Station handles orders of magnitude more passengers than a rural stop in Montana. If you ignore these fixed differences and pool the data, your COVID coefficient absorbs station-level heterogeneity and the estimate is biased.

Fixed effects panel OLS solves this by estimating a separate intercept for every station, then identifying the COVID effect only from within-station variation over time. The comparison is no longer Boston vs. Montana — it is Boston in 2019 vs. Boston in 2020.

## The Model

```
Ridership_it = α_i + β_1·t + β_2·post_covid + β_3·(post_covid × region) + ε_it
```

- `α_i` — station fixed effect, absorbs all time-invariant station characteristics
- `t` — linear time trend, captures secular growth in rail demand
- `post_covid` — indicator for year ≥ 2020 (Midwest is the baseline region)
- `post_covid × region` — interaction terms that measure how each region's COVID drop differed from the Midwest baseline

The interaction terms are the key insight. Without them, you get one average COVID effect. With them, you get region-specific estimates, which is what actually matters for understanding where ridership collapsed and where it recovered.

## Data

Public Amtrak ridership data by station and year (2005–2022), sourced from:

```
https://raw.githubusercontent.com/kylejones200/time_series/refs/heads/main/data/amtrak_ridership_time_series_data.csv
```

Stations are assigned to four Census regions: Northeast, Midwest, South, West. The panel is unbalanced — not every station reports every year — so the fixed effects estimator handles missing observations naturally.

## Implementation

The `linearmodels` library handles panel data in Python. The key setup is converting the DataFrame to a MultiIndex with entity (station) and time (year) dimensions:

```python
import pandas as pd
from linearmodels.panel import PanelOLS

df = df.set_index(['station_id', 'year'])

model = PanelOLS(
    df['ridership'],
    df[['t', 'post_covid', 'post_covid_northeast', 'post_covid_south', 'post_covid_west']],
    entity_effects=True
)
result = model.fit(cov_type='clustered', cluster_entity=True)
print(result.summary)
```

Clustering standard errors by entity (station) is important here. Ridership at a given station is correlated across years — ignoring that would understate the true uncertainty in the estimates.

## Key Results

| Parameter | Estimate | p-value |
|---|---|---|
| `post_covid` (Midwest baseline) | −61,450 | < 0.001 |
| `t` (annual trend) | +4,136 | < 0.001 |
| `post_covid × Northeast` | sharper drop than Midwest | < 0.001 |
| `post_covid × South` | see model output | < 0.001 |
| `post_covid × West` | see model output | < 0.001 |

The baseline Midwest drop of −61,450 annual riders per station is large. The Northeast interaction reveals an even sharper collapse — consistent with the density of the NEC corridor and the higher share of commuter and business travel, both of which fell to near zero in spring 2020.

The secular trend coefficient (+4,136 per year) is meaningful on its own. It tells us that even controlling for COVID, rail demand was growing across all regions before the pandemic. That gives us a cleaner counterfactual: what ridership would have looked like without the shock.

## What Fixed Effects Gives You

The main alternative is pooled OLS with no station dummies. You can run both and compare. The fixed effects estimate of the COVID shock will typically be smaller in magnitude than pooled OLS — because pooled OLS confounds the pandemic effect with the fact that low-ridership stations tend to cluster in certain regions or have different trends.

That is the core value of the panel approach: it strips out the between-station noise and forces the model to answer a cleaner question. For a given station, how did COVID change things?

## Quickstart

```bash
pip install linearmodels pandas matplotlib numpy requests
python amtrak_panel_ols.py
```

Outputs: `amtrak_ridership_trends.png`, `amtrak_panel_region_effects.png`

## Files

| File | Description |
|---|---|
| `amtrak_panel_ols.py` | Main script: loads data, fits PanelOLS, generates figures |
| `data_io.py` | Thin CSV loader utility |

## Key Takeaways

- Fixed effects panel OLS removes time-invariant station heterogeneity and identifies the COVID effect from within-station variation only.
- Region × COVID interaction terms reveal that the pandemic's impact was not uniform — the Northeast saw a sharper collapse than the Midwest baseline.
- Clustering standard errors by station is essential when observations within an entity are correlated across time.
- The secular trend coefficient (+4,136/year) provides a counterfactual for what ridership would have looked like without the pandemic.
