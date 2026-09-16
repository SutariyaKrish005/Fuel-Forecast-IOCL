"""
train_sarimax.py
-----------------
Loads the synthetic fuel sales data, aggregates it by region/product,
fits a SARIMAX model per product for a chosen region, evaluates forecast
accuracy on a held-out test period, saves the fitted models, produces
actual-vs-forecast plots, and writes a metrics summary CSV.

Run:
    python src/train_sarimax.py
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")  # SARIMAX convergence warnings are noisy but non-fatal

sns.set_style("whitegrid")

# ----------------------------------------------------------------------
# Paths (relative to this script, so it works when cloned anywhere)
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "fuel_sales.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
for d in (FIGURES_DIR, MODELS_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TARGET_REGION = "North"           # region to model in this script
PRODUCTS = ["Petrol", "Diesel"]
TEST_DAYS = 60                    # size of held-out test window
FUTURE_HORIZON = 30               # days to forecast beyond the full dataset

EXOG_COLS = ["is_holiday", "is_weekend", "festive_flag", "temperature_c", "rainfall_mm"]

SARIMAX_ORDER = (1, 1, 1)
SARIMAX_SEASONAL_ORDER = (1, 1, 1, 7)  # weekly seasonality on daily data

EPSILON = 1e-6  # used only if any actual value in MAPE calc is exactly 0


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_region_product_series(df: pd.DataFrame, region: str, product: str) -> pd.DataFrame:
    """Filters to a region/product, aggregates across outlets by date,
    and returns a DataFrame indexed by a continuous daily DatetimeIndex
    with the target column 'volume_liters' plus exogenous columns."""
    subset = df[(df["region"] == region) & (df["product"] == product)].copy()

    agg = subset.groupby("date").agg(
        volume_liters=("volume_liters", "sum"),
        is_holiday=("is_holiday", "max"),
        is_weekend=("is_weekend", "max"),
        festive_flag=("festive_flag", "max"),
        temperature_c=("temperature_c", "mean"),
        rainfall_mm=("rainfall_mm", "mean"),
    ).reset_index()

    agg = agg.sort_values("date").reset_index(drop=True)
    agg = agg.set_index("date")
    agg.index = pd.DatetimeIndex(agg.index)
    agg = agg.asfreq("D")  # enforce daily frequency; raises visibility on any gaps

    # If asfreq introduced any NaNs (shouldn't happen with synthetic data,
    # but guarded here for robustness), forward-fill exog and interpolate target.
    if agg.isnull().any().any():
        agg["volume_liters"] = agg["volume_liters"].interpolate()
        for col in EXOG_COLS:
            agg[col] = agg[col].ffill().bfill()

    return agg


def mean_absolute_percentage_error_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE that guards against division by zero: any zero actuals are
    excluded from the MAPE calculation (documented rather than silently
    biasing the metric with an epsilon substitution)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero_mask = np.abs(y_true) > EPSILON
    if nonzero_mask.sum() == 0:
        return float("nan")
    return float(
        np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    )


def build_future_exog(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Builds a simple future exogenous-variable DataFrame for the forecast
    horizon beyond the historical data. Weekend/holiday flags are computed
    from the calendar (deterministic); festive_flag is assumed 0 (no
    known festival in the immediate horizon) unless it coincides with a
    weekend; temperature/rainfall are carried forward using the same
    day-of-year value from the prior year as a simple seasonal proxy,
    falling back to the last 30-day average if unavailable."""
    last_date = history.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    future = pd.DataFrame(index=future_dates)
    future.index.name = "date"
    future["is_weekend"] = future.index.dayofweek.isin([5, 6]).astype(int)
    future["is_holiday"] = 0  # no specific holiday calendar assumed for the horizon
    future["festive_flag"] = 0  # simple assumption: no major festival in this window

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


def plot_actual_vs_forecast(
    train: pd.Series,
    test: pd.Series,
    test_forecast: pd.Series,
    future_forecast: pd.Series,
    region: str,
    product: str,
    save_path: str,
):
    """Creates and saves a plot of actual vs. SARIMAX forecast for the
    test period, plus the future forecast extension."""
    fig, ax = plt.subplots(figsize=(13, 6))

    # Show last 90 days of training history for context, plus full test period
    context = train.tail(90)
    ax.plot(context.index, context.values, label="Train (recent history)", color="#4C72B0")
    ax.plot(test.index, test.values, label="Actual (test)", color="#2C6E49", linewidth=2)
    ax.plot(
        test_forecast.index,
        test_forecast.values,
        label="Forecast (test)",
        color="#C44E52",
        linewidth=2,
        linestyle="--",
    )
    ax.plot(
        future_forecast.index,
        future_forecast.values,
        label=f"Future forecast (+{len(future_forecast)}d)",
        color="#DD8452",
        linewidth=2,
        linestyle=":",
    )

    ax.set_title(f"{region} Region - {product}: Actual vs SARIMAX Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume (liters)")
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def run_region(df: pd.DataFrame, region: str) -> list:
    """Runs the full train/evaluate/forecast/save pipeline for every
    product in the given region. Returns a list of metric-row dicts."""
    metrics_rows = []

    for product in PRODUCTS:
        print("=" * 70)
        print(f"Region: {region} | Product: {product}")

        series_df = load_region_product_series(df, region, product)
        y = series_df["volume_liters"]
        exog = series_df[EXOG_COLS]

        if len(y) <= TEST_DAYS + 30:
            raise ValueError(
                f"Not enough data ({len(y)} days) for a {TEST_DAYS}-day test split. "
                "Reduce TEST_DAYS or provide more history."
            )

        # ---- Train/test split ----
        train_y, test_y = y.iloc[:-TEST_DAYS], y.iloc[-TEST_DAYS:]
        train_exog, test_exog = exog.iloc[:-TEST_DAYS], exog.iloc[-TEST_DAYS:]

        print(f"Train range: {train_y.index.min().date()} to {train_y.index.max().date()} ({len(train_y)} days)")
        print(f"Test range : {test_y.index.min().date()} to {test_y.index.max().date()} ({len(test_y)} days)")
        print(f"SARIMAX order={SARIMAX_ORDER}, seasonal_order={SARIMAX_SEASONAL_ORDER}")

        # ---- Fit SARIMAX on training data ----
        model = SARIMAX(
            train_y,
            exog=train_exog,
            order=SARIMAX_ORDER,
            seasonal_order=SARIMAX_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)

        # ---- Forecast over the test horizon using actual test exog ----
        test_forecast_res = fitted.get_forecast(steps=len(test_y), exog=test_exog)
        test_forecast = test_forecast_res.predicted_mean
        test_forecast.index = test_y.index  # align index exactly

        # ---- Evaluation metrics on test set ----
        mae = mean_absolute_error(test_y, test_forecast)
        rmse = float(np.sqrt(mean_squared_error(test_y, test_forecast)))
        mape = mean_absolute_percentage_error_safe(test_y.values, test_forecast.values)

        print(f"MAE  = {mae:,.2f} liters")
        print(f"RMSE = {rmse:,.2f} liters")
        print(f"MAPE = {mape:,.2f}%")
        if mape < 8:
            interpretation = f"MAPE of ~{mape:.1f}% indicates highly accurate short-term forecasts for {product} in {region} region."
        elif mape < 15:
            interpretation = f"MAPE of ~{mape:.1f}% indicates reasonably accurate short-term forecasts for {product} in {region} region."
        else:
            interpretation = f"MAPE of ~{mape:.1f}% suggests forecast accuracy for {product} in {region} region could be improved (consider tuning SARIMAX orders or adding exogenous signals)."
        print(f"Interpretation: {interpretation}")

        # ---- Refit on FULL series (train+test) before producing the future forecast ----
        # This uses all available history for the most up-to-date future forecast,
        # while the metrics above remain a fair evaluation on unseen test data.
        full_model = SARIMAX(
            y,
            exog=exog,
            order=SARIMAX_ORDER,
            seasonal_order=SARIMAX_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        full_fitted = full_model.fit(disp=False)

        future_exog = build_future_exog(series_df, FUTURE_HORIZON)
        future_forecast_res = full_fitted.get_forecast(steps=FUTURE_HORIZON, exog=future_exog)
        future_forecast = future_forecast_res.predicted_mean
        future_forecast.index = future_exog.index
        future_forecast = future_forecast.clip(lower=0)  # volumes can't be negative

        # ---- Plot ----
        plot_filename = f"{region.lower()}_{product.lower()}_actual_vs_forecast.png"
        plot_path = os.path.join(FIGURES_DIR, plot_filename)
        plot_actual_vs_forecast(train_y, test_y, test_forecast, future_forecast, region, product, plot_path)
        print(f"Saved plot: {plot_path}")

        # ---- Save fitted models ----
        # NOTE ON PERSISTENCE STRATEGY: a full statsmodels SARIMAXResults
        # object carries dense Kalman filter/smoother matrices (state
        # covariances etc. for every time step), which can balloon to
        # 100+ MB per model even for a modest dataset. Since SARIMAX
        # parameters fully determine the model, we instead persist only
        # the small fitted parameter vector plus the model specification
        # (order, seasonal_order, exog columns). To forecast later, the
        # model is reconstructed from the same historical y/exog data and
        # "smoothed" at the saved params (mod.smooth(params)) -- this is
        # instant (no re-optimization) and produces numerically identical
        # results to the original fitted object, at a tiny fraction of
        # the file size.
        model_filename = f"sarimax_{region.lower()}_{product.lower()}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(
            {
                "test_split_params": fitted.params,
                "full_history_params": full_fitted.params,
                "order": SARIMAX_ORDER,
                "seasonal_order": SARIMAX_SEASONAL_ORDER,
                "exog_cols": EXOG_COLS,
                "region": region,
                "product": product,
                "last_history_date": y.index.max(),
                "train_end_date": train_y.index.max(),
            },
            model_path,
        )
        print(f"Saved model: {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")

        # ---- Save future forecast CSV ----
        forecast_filename = f"{region.lower()}_{product.lower()}_forecast_30d.csv"
        forecast_path = os.path.join(OUTPUT_DIR, forecast_filename)
        forecast_out = pd.DataFrame(
            {
                "date": future_forecast.index.strftime("%Y-%m-%d"),
                "product": product,
                "region": region,
                "forecast_volume_liters": np.round(future_forecast.values, 1),
            }
        )
        forecast_out.to_csv(forecast_path, index=False)
        print(f"Saved future forecast: {forecast_path}")

        metrics_rows.append(
            {
                "region": region,
                "product": product,
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "MAPE": round(mape, 2),
                "test_start_date": test_y.index.min().strftime("%Y-%m-%d"),
                "test_end_date": test_y.index.max().strftime("%Y-%m-%d"),
                "forecast_horizon_days": FUTURE_HORIZON,
            }
        )

    return metrics_rows


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run 'python src/generate_data.py' first to create the dataset."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    all_metrics = []
    all_metrics.extend(run_region(df, TARGET_REGION))

    # ------------------------------------------------------------------
    # Demonstration: the same logic extends to other regions.
    # Uncomment/modify to run additional regions in one go, e.g.:
    #
    #   for region in ["South", "East", "West", "Central"]:
    #       all_metrics.extend(run_region(df, region))
    #
    # This is left commented out by default so the primary run stays
    # focused on the North region as scoped, but the function is fully
    # general and reusable per-region without any code changes.
    # ------------------------------------------------------------------

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("=" * 70)
    print(f"Metrics summary saved to: {metrics_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
