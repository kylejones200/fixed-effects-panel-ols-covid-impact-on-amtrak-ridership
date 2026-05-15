"""
Fixed Effects Time Series Modeling with Panel OLS:
Region & COVID Interaction in Amtrak Ridership

Model:  Ridership_it = α_i + β_1*t + β_2*post_covid + β_3*(post_covid × region) + ε_it

Key results from full-sample run:
  post_covid:  -61,450  (p < 0.001)  — baseline drop (Midwest reference)
  t:           +4,136   (p < 0.001)  — secular annual growth trend

Usage:
    python amtrak_panel_ols.py
"""

import signalplot
import logging

import matplotlib

from pathlib import Path


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        return {}
    with open(config_path) as _f:
        import yaml as _yaml
        return _yaml.safe_load(_f) or {}

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from data_io import read_csv

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

DATA_URL = (
    "https://raw.githubusercontent.com/kylejones200/time_series/refs/heads/main"
    "/data/amtrak_ridership_time_series_data.csv"
)

REGION_DEF = {
    "Northeast": [
        "Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island",
        "Vermont", "New Jersey", "New York", "Pennsylvania",
    ],
    "Midwest": [
        "Indiana", "Illinois", "Michigan", "Ohio", "Wisconsin",
        "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska",
        "North Dakota", "South Dakota",
    ],
    "South": [
        "Delaware", "Florida", "Georgia", "Maryland", "North Carolina",
        "South Carolina", "Virginia", "West Virginia", "Alabama", "Kentucky",
        "Mississippi", "Tennessee", "Arkansas", "Louisiana", "Oklahoma",
        "Texas", "District of Columbia",
    ],
    "West": [
        "Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico",
        "Utah", "Wyoming", "Alaska", "California", "Hawaii", "Oregon", "Washington",
    ],
}

signalplot.apply(font_family='serif')


def assign_region(state: str) -> str:
    for region, states in REGION_DEF.items():
        if state in states:
            return region
    return "Other"


def load_and_prepare(url: str) -> pd.DataFrame:
    df = read_csv(url)
    df["Year"] = pd.to_datetime(df["Year"])
    df["year"] = df["Year"].dt.year
    df["Ridership"] = pd.to_numeric(df["Ridership"], errors="coerce")
    df = df.dropna(subset=["Ridership"])

    df["post_covid"] = (df["year"] >= 2020).astype(int)
    df["t"] = df["year"] - 2005

    station_meta = df.drop_duplicates("Station")[["Station", "State"]].copy()
    station_meta["region"] = station_meta["State"].apply(assign_region)
    df = df.merge(station_meta[["Station", "region"]], on="Station", how="left")

    df = df[df["region"] != "Other"]
    df = df.set_index(["Station", "year"]).sort_index()
    return df


def fit_panel_model(df: pd.DataFrame):
    formula = "Ridership ~ post_covid * region + t + EntityEffects"
    model = PanelOLS.from_formula(formula, data=df, drop_absorbed=True)
    res = model.fit(cov_type="clustered", cluster_entity=True)
    logger.info("\n%s", res.summary)
    return res


def extract_region_effects(res) -> tuple[dict, dict]:
    baseline = res.params["post_covid"]
    se_baseline = res.std_errors["post_covid"]

    regions = ["Midwest", "Northeast", "South", "West"]
    effects, ci = {}, {}
    for region in regions:
        key = f"post_covid:region[T.{region}]"
        delta = res.params.get(key, 0.0)
        se_delta = res.std_errors.get(key, se_baseline)
        effect = baseline + delta
        margin = 1.96 * np.sqrt(se_baseline**2 + se_delta**2)
        effects[region] = effect
        ci[region] = (effect - margin, effect + margin)

    # Midwest is the reference — no interaction term, use baseline SE directly
    effects["Midwest"] = baseline
    ci["Midwest"] = (baseline - 1.96 * se_baseline, baseline + 1.96 * se_baseline)
    return effects, ci


def plot_region_effects(effects: dict, ci: dict, out_path: str = "amtrak_panel_region_effects.png", plot: bool = False):
    regions = ["Northeast", "Midwest", "South", "West"]
    coefs = [effects[r] for r in regions]
    lower_err = [abs(ci[r][0] - effects[r]) for r in regions]
    upper_err = [abs(ci[r][1] - effects[r]) for r in regions]

    if plot:
        fig, ax = plt.subplots(figsize=tuple(config.get('output', {}).get('figsize', [8, 5])))
        ax.errorbar(
            regions, coefs,
            yerr=[lower_err, upper_err],
            fmt="o", color="black", capsize=5, linewidth=1.2,
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title("Estimated Drop in Ridership Post-COVID by Region", fontsize=13)
        ax.set_ylabel("Marginal Effect on Annual Ridership")
        ax.set_xlabel("Region")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    logger.info("Figure saved: %s", out_path)


def plot_ridership_trends(df: pd.DataFrame, out_path: str = "amtrak_ridership_trends.png", plot: bool = False):
    annual = (
        df.reset_index()
        .groupby(["year", "region"])["Ridership"]
        .sum()
        .reset_index()
    )
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        for region, grp in annual.groupby("region"):
            ax.plot(grp["year"], grp["Ridership"] / 1e6, label=region, linewidth=1.5)
        ax.axvline(2020, color="red", linestyle="--", linewidth=0.8, label="COVID-19 (2020)")
        ax.set_title("Annual Amtrak Ridership by Region (millions)", fontsize=13)
        ax.set_ylabel("Ridership (millions)")
        ax.set_xlabel("Year")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    logger.info("Figure saved: %s", out_path)


def main():
    logger.info("Loading Amtrak ridership data...")
    df = load_and_prepare(DATA_URL)
    logger.info("Panel shape: %s stations × years", df.shape)

    plot_ridership_trends(df)

    logger.info("Fitting Panel OLS with region × COVID interaction...")
    res = fit_panel_model(df)

    effects, ci = extract_region_effects(res)
    for region in sorted(effects):
        lo, hi = ci[region]
        logger.info("  %s: %.0f  [95%% CI: %.0f, %.0f]", region, effects[region], lo, hi)

    plot_region_effects(effects, ci)
    logger.info("\nOutputs: amtrak_ridership_trends.png, amtrak_panel_region_effects.png")


if __name__ == "__main__":
    main()
