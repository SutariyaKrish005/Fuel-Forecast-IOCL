# Fuel Demand Forecasting for Retail Outlets (IOCL) using SARIMAX

Short-term demand forecasting for petrol and diesel at retail fuel outlets,
built with SARIMAX (Seasonal ARIMA with eXogenous regressors) in Python.
Uses realistic **synthetic** data (no external downloads) so the project
runs fully offline and reproducibly.

## Business Problem

IOCL needs accurate short-term forecasts of petrol/diesel demand by
region/outlet to optimize inventory, reduce stockouts, and plan logistics.
This project trains a separate SARIMAX model per product (Petrol, Diesel)
for a chosen region, evaluates it on a held-out test window, and produces
a 30-day future forecast.

## Project Structure

```
.
├── data/
│   └── fuel_sales.csv                  # synthetic daily fuel sales data
├── src/
│   ├── generate_data.py                # generates the synthetic dataset
│   ├── train_sarimax.py                # trains, evaluates, plots, saves models
│   └── forecast_future.py              # reloads saved models -> fresh 30-day forecasts
├── figures/
│   ├── north_petrol_actual_vs_forecast.png
│   └── north_diesel_actual_vs_forecast.png
├── models/
│   ├── sarimax_north_petrol.joblib
│   └── sarimax_north_diesel.joblib
├── output/
│   ├── north_petrol_forecast_30d.csv
│   ├── north_diesel_forecast_30d.csv
│   └── metrics_summary.csv
└── README.md
```

## Requirements

- Python 3.10+
- pandas, numpy, matplotlib, seaborn, statsmodels, scikit-learn, joblib

Install everything with:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn joblib
```

## How to Run

Run the three scripts in order from the project root:

```bash
# 1. Generate ~3 years of synthetic daily fuel sales data (7 outlets, 5 regions)
python src/generate_data.py

# 2. Aggregate by region/product, fit SARIMAX, evaluate, plot, and save models
python src/train_sarimax.py

# 3. (Optional) Reload the saved models to regenerate the 30-day forecast
#    without retraining — useful for scheduled/repeated forecast refreshes
python src/forecast_future.py
```

All paths in the scripts are relative to the script location, so the
project works regardless of where it's cloned or from which directory
you invoke Python, as long as you run from the project root (or the
relative folder structure above is preserved).

## Data Generation (`generate_data.py`)

Produces a daily time series (2022-01-01 to 2024-12-31) for 7 outlets
across 5 regions (North, South, East, West, Central) and 2 products
(Petrol, Diesel), with:

- **Weekly seasonality**: Petrol demand rises on weekends (personal
  travel); Diesel dips on Sundays and has a smaller Saturday bump
  (commercial/freight pattern).
- **Yearly seasonality**: Petrol peaks around summer travel months;
  Diesel has a mild post-monsoon/winter freight bump.
- **Festival demand spikes**: a recurring set of Indian festival/holiday
  dates (illustrative, not an authoritative calendar) drives `is_holiday`
  and a ±1-day `festive_flag` window with elevated demand.
- **Slowly varying fuel prices** (`price_per_liter`) via a bounded random
  walk, different base prices for Petrol vs. Diesel.
- **Weather proxies**: seasonal `temperature_c` and monsoon-season
  `rainfall_mm`.
- Guaranteed **no missing dates** per outlet-product pair (asserted in code).

## Modeling Approach (`train_sarimax.py`)

For the **North** region (configurable via `TARGET_REGION`), for each
product:

1. Aggregate outlet-level volumes to a single daily regional series.
2. Set a `DatetimeIndex` with explicit daily frequency (`asfreq("D")`).
3. Split into train / test, holding out the **last 60 days** as test.
4. Fit `statsmodels.tsa.statespace.SARIMAX` with:
   - `order=(1,1,1)`
   - `seasonal_order=(1,1,1,7)` (weekly seasonality for daily data)
   - exogenous regressors: `is_holiday`, `is_weekend`, `festive_flag`,
     `temperature_c`, `rainfall_mm`
5. Forecast the test window using the actual test-period exogenous values,
   and compute **MAE**, **RMSE**, and **MAPE** (MAPE skips any exact-zero
   actuals to avoid division-by-zero — none occur in this synthetic data,
   but the guard is documented and in place).
6. Refit on the **full history** (train+test) to produce the most
   up-to-date **30-day future forecast**. Future exogenous inputs are
   built with simple, deterministic rules: weekend flags from the
   calendar, no assumed holidays/festivals in the immediate horizon, and
   temperature/rainfall proxied from the same calendar day one year
   earlier (falling back to a recent 30-day average).
7. Save actual-vs-forecast plots, the fitted model objects (both the
   test-split model and the full-history model, via `joblib`), the
   30-day forecast CSVs, and a consolidated `metrics_summary.csv`.

### Extending to other regions

`train_sarimax.py` is organized around a single `run_region(df, region)`
function that does the full fit/evaluate/plot/save pipeline for both
products in one region. To run additional regions, either change
`TARGET_REGION`, or loop over several regions, e.g.:

```python
for region in ["South", "East", "West", "Central"]:
    all_metrics.extend(run_region(df, region))
```

This is shown (commented out) at the bottom of `main()` in
`train_sarimax.py`.

## Key Results (North Region, this synthetic dataset)

| Region | Product | MAE (L) | RMSE (L) | MAPE  | Test Window            |
|--------|---------|---------|----------|-------|-------------------------|
| North  | Petrol  | ~199    | ~244     | ~3.4% | 2024-11-02 to 2024-12-31 |
| North  | Diesel  | ~305    | ~371     | ~3.6% | 2024-11-02 to 2024-12-31 |

A MAPE in the 3-4% range indicates the SARIMAX model captures the
weekly and festival-driven demand patterns in this dataset very well.
Exact numbers will vary slightly if you change the random seed or the
date range in `generate_data.py`.

> Note: results above reflect this synthetic dataset's known, encoded
> seasonality — real-world IOCL sales data will likely show higher
> MAPE and may benefit from additional exogenous signals (local events,
> competitor pricing, macroeconomic indicators) or SARIMAX order tuning
> (e.g., a grid search over `(p,d,q)` and `(P,D,Q,7)`).

## Notes & Assumptions

- The festival calendar in `generate_data.py` uses fixed (month, day)
  pairs as an approximation for demand-spike simulation; real festival
  dates (e.g., Holi, Diwali) shift each year on the lunar calendar. If
  you plug in real sales data, replace this with an accurate holiday
  calendar for the relevant years.
- Future exogenous variables for the 30-day forecast are necessarily
  assumptions (no holidays/festivals assumed, weather proxied from the
  prior year) since real future weather/holiday data isn't known in
  advance. Replace `build_future_exog()` with real forecasted values
  (e.g., a weather API, an official holiday calendar) for production use.
- `enforce_stationarity=False` and `enforce_invertibility=False` are set
  on the SARIMAX model for robustness against edge cases in the
  synthetic data; revisit these if you require strict stationarity
  guarantees on real data.

<img width="1950" height="900" alt="north_diesel_actual_vs_forecast" src="https://github.com/user-attachments/assets/ff307f05-7655-463d-a6c2-49d731c8e11a" />
<img width="1950" height="900" alt="north_petrol_actual_vs_forecast" src="https://github.com/user-attachments/assets/d049ea79-9634-463f-82b5-07e37036b35e" />

