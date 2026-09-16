"""
forecast_future.py
-------------------
Loads previously-saved SARIMAX models (produced by train_sarimax.py) and
generates fresh 30-day future forecasts without needing to refit the
models from scratch. Saves results to output/<region>_<product>_forecast_30d.csv.

This script is useful when you want to (re)generate the future forecast
on a schedule without re-running the full train/evaluate pipeline.

Run:
    python src/forecast_future.py
"""

import os
import glob
import warnings

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "fuel_sales.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FUTURE_HORIZON = 30
EXOG_COLS = ["is_holiday", "is_weekend", "festive_flag", "temperature_c", "rainfall_mm"]


def load_region_product_series(df: pd.DataFrame, region: str, product: str) -> pd.DataFrame:
    """Same aggregation logic used in train_sarimax.py, kept here so this
    script can run independently and reproduce identical history features
    needed to build future exogenous variables."""
    subset = df[(df["region"] == region) & (df["product"] == product)].copy()

    agg = subset.groupby("date").agg(
        volume_liters=("volume_liters", "sum"),
        is_holiday=("is_holiday", "max"),
        is_weekend=("is_weekend", "max"),
        festive_flag=("festive_flag", "max"),
        temperature_c=("temperature_c", "mean"),
        rainfall_mm=("rainfall_mm", "mean"),
    ).reset_index()

    agg = agg.sort_values("date").set_index("date")
    agg.index = pd.DatetimeIndex(agg.index)
    agg = agg.asfreq("D")

    if agg.isnull().any().any():
        agg["volume_liters"] = agg["volume_liters"].interpolate()
        for col in EXOG_COLS:
            agg[col] = agg[col].ffill().bfill()

    return agg


def build_future_exog(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Identical logic to train_sarimax.py's build_future_exog: deterministic
    calendar flags for weekend/holiday/festive, and a prior-year-same-day
    proxy for temperature/rainfall (falling back to a recent 30-day average)."""
    last_date = history.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    future = pd.DataFrame(index=future_dates)
    future.index.name = "date"
    future["is_weekend"] = future.index.dayofweek.isin([5, 6]).astype(int)
    future["is_holiday"] = 0
    future["festive_flag"] = 0

    recent_avg_temp = history["temperature_c"].tail(30).mean()
    recent_avg_rain = history["rainfall_mm"].tail(30).mean()

    temp_values, rain_values = [], []
    for d in future_dates:
        prior_year_date = d - pd.DateOffset(years=1)
        if prior_year_date in history.index:
            temp_values.append(history.loc[prior_year_date, "temperature_c"])
            rain_values.append(history.loc[prior_year_date, "rainfall_mm"])
        else:
            temp_values.append(recent_avg_temp)
            rain_values.append(recent_avg_rain)

    future["temperature_c"] = temp_values
    future["rainfall_mm"] = rain_values

    return future[EXOG_COLS]


def forecast_from_saved_model(model_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """Loads one saved (lightweight) model bundle -- just the fitted SARIMAX
    parameter vector plus its specification -- and produces a future
    forecast CSV-ready DataFrame.

    Rather than pickling the full statsmodels results object (which stores
    dense Kalman filter/smoother matrices and can be 100+ MB per model),
    train_sarimax.py saves only the small params vector and the model spec.
    Here we rebuild the SARIMAX model against the full historical data and
    call `mod.smooth(params)`, which re-applies the already-fitted
    parameters without any re-optimization (fast) and yields numerically
    identical results to the original fitted object."""
    bundle = joblib.load(model_path)
    region = bundle["region"]
    product = bundle["product"]
    order = bundle["order"]
    seasonal_order = bundle["seasonal_order"]
    exog_cols = bundle["exog_cols"]
    params = bundle["full_history_params"]

    print(f"Loaded model: {os.path.basename(model_path)} (region={region}, product={product})")

    history = load_region_product_series(df, region, product)
    y = history["volume_liters"]
    exog = history[exog_cols]

    mod = SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    full_fitted = mod.smooth(params)

    future_exog = build_future_exog(history, FUTURE_HORIZON)

    forecast_res = full_fitted.get_forecast(steps=FUTURE_HORIZON, exog=future_exog)
    forecast_mean = forecast_res.predicted_mean
    forecast_mean.index = future_exog.index
    forecast_mean = forecast_mean.clip(lower=0)

    out = pd.DataFrame(
        {
            "date": forecast_mean.index.strftime("%Y-%m-%d"),
            "product": product,
            "region": region,
            "forecast_volume_liters": np.round(forecast_mean.values, 1),
        }
    )
    return out, region, product


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run 'python src/generate_data.py' first."
        )

    model_paths = sorted(glob.glob(os.path.join(MODELS_DIR, "sarimax_*.joblib")))
    if not model_paths:
        raise FileNotFoundError(
            f"No saved models found in {MODELS_DIR}. Run 'python src/train_sarimax.py' first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    for model_path in model_paths:
        forecast_df, region, product = forecast_from_saved_model(model_path, df)
        out_filename = f"{region.lower()}_{product.lower()}_forecast_30d.csv"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        forecast_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path} ({len(forecast_df)} rows)")


if __name__ == "__main__":
    main()
