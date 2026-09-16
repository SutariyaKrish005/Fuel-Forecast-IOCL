"""
generate_data.py
----------------
Generates a realistic synthetic daily fuel-sales dataset that mimics IOCL
retail outlet sales for Petrol and Diesel across multiple outlets/regions,
and saves it to ../data/fuel_sales.csv (relative to this script).

Run:
    python src/generate_data.py
"""

import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
RNG_SEED = 42
np.random.seed(RNG_SEED)

# ----------------------------------------------------------------------
# Paths (relative, so this works regardless of the current working dir)
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(DATA_DIR, "fuel_sales.csv")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"  # ~3 years of daily data

OUTLETS = [
    {"outlet_id": "OUT_001", "region": "North"},
    {"outlet_id": "OUT_002", "region": "North"},
    {"outlet_id": "OUT_003", "region": "South"},
    {"outlet_id": "OUT_004", "region": "South"},
    {"outlet_id": "OUT_005", "region": "East"},
    {"outlet_id": "OUT_006", "region": "West"},
    {"outlet_id": "OUT_007", "region": "Central"},
]

PRODUCTS = ["Petrol", "Diesel"]

# Base daily volume (liters) per outlet-product, before seasonality/noise
BASE_VOLUME = {
    "Petrol": 3200.0,
    "Diesel": 4100.0,
}

# Starting prices (INR/liter) - vary slowly over time via random walk
BASE_PRICE = {
    "Petrol": 96.5,
    "Diesel": 89.0,
}

# Indian festival / holiday-ish dates (month, day) recurring each year.
# These are illustrative/approximate dates used only to simulate demand
# spikes and are not meant to be an authoritative festival calendar.
FESTIVAL_MONTH_DAY = [
    (1, 14),   # Makar Sankranti / Pongal
    (3, 8),    # Holi (approx, varies by year in reality)
    (8, 15),   # Independence Day
    (8, 19),   # Raksha Bandhan (approx)
    (9, 7),    # Ganesh Chaturthi (approx)
    (10, 2),   # Gandhi Jayanti
    (10, 12),  # Dussehra (approx)
    (11, 1),   # Diwali (approx)
    (11, 2),   # Diwali +1 (approx)
    (12, 25),  # Christmas
]


def build_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """Builds the shared calendar features (holiday/weekend/festive/weather)
    that apply identically across outlets for a given date."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    cal = pd.DataFrame({"date": dates})

    cal["is_weekend"] = cal["date"].dt.dayofweek.isin([5, 6]).astype(int)

    # Festive flag: +/- 1 day window around each festival date, for each year
    festive_dates = set()
    holiday_dates = set()
    for year in range(dates.min().year, dates.max().year + 1):
        for month, day in FESTIVAL_MONTH_DAY:
            try:
                base = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                continue  # skip invalid dates (e.g., Feb 30)
            holiday_dates.add(base)
            for offset in (-1, 0, 1):
                festive_dates.add(base + pd.Timedelta(days=offset))

    cal["is_holiday"] = cal["date"].isin(holiday_dates).astype(int)
    cal["festive_flag"] = cal["date"].isin(festive_dates).astype(int)

    # Seasonal temperature (Celsius): warm summers, cool winters (India-ish)
    day_of_year = cal["date"].dt.dayofyear
    cal["temperature_c"] = (
        27
        + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        + np.random.normal(0, 1.5, size=len(cal))
    ).round(1)

    # Monsoon-season rainfall (roughly June-September gets most rain)
    month = cal["date"].dt.month
    monsoon_mask = month.isin([6, 7, 8, 9])
    rainfall = np.where(
        monsoon_mask,
        np.random.gamma(shape=2.0, scale=12.0, size=len(cal)),
        np.random.gamma(shape=0.6, scale=2.0, size=len(cal)),
    )
    cal["rainfall_mm"] = np.round(rainfall, 1)

    return cal


def simulate_prices(dates: pd.DatetimeIndex, base_price: float, seed_offset: int) -> np.ndarray:
    """Simulates a slowly-varying price series via a bounded random walk."""
    rng = np.random.RandomState(RNG_SEED + seed_offset)
    n = len(dates)
    steps = rng.normal(loc=0.0, scale=0.06, size=n)
    walk = np.cumsum(steps)
    # Gentle mean reversion so price doesn't drift unrealistically far
    walk = walk - np.linspace(0, walk[-1], n) * 0.6
    price = base_price + walk
    price = np.clip(price, base_price * 0.85, base_price * 1.20)
    return np.round(price, 2)


def simulate_volume(
    cal: pd.DataFrame,
    product: str,
    outlet_scale: float,
    seed_offset: int,
) -> np.ndarray:
    """Simulates daily volume_liters for one outlet-product combination,
    including weekly seasonality, yearly seasonality, festival spikes,
    and random noise."""
    rng = np.random.RandomState(RNG_SEED + seed_offset)
    n = len(cal)
    day_of_year = cal["date"].dt.dayofyear.values
    dow = cal["date"].dt.dayofweek.values  # 0=Mon ... 6=Sun

    base = BASE_VOLUME[product] * outlet_scale

    # Weekly seasonality: weekends higher (private vehicle travel).
    # Diesel (commercial/freight heavy) dips a bit more on Sundays.
    if product == "Petrol":
        weekly_effect = np.where(np.isin(dow, [5, 6]), 1.18, 1.0)
    else:
        weekday_effect = np.where(dow == 6, 0.85, 1.0)  # lower on Sunday (less freight)
        weekend_boost = np.where(dow == 5, 1.05, 1.0)   # slight Saturday bump
        weekly_effect = weekday_effect * weekend_boost

    # Yearly seasonality: summer travel bump for Petrol (Apr-Jun),
    # slight post-monsoon/winter freight bump for Diesel (Oct-Dec).
    if product == "Petrol":
        yearly_effect = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25)
    else:
        yearly_effect = 1.0 + 0.10 * np.sin(2 * np.pi * (day_of_year - 280) / 365.25)

    # Long-term mild growth trend (~4% per year)
    years_elapsed = (cal["date"] - cal["date"].min()).dt.days.values / 365.25
    trend_effect = 1.0 + 0.04 * years_elapsed

    # Festival demand spikes (Petrol reacts more due to personal travel)
    festive_boost = np.where(
        cal["festive_flag"].values == 1,
        1.25 if product == "Petrol" else 1.12,
        1.0,
    )

    # Random noise (multiplicative, small)
    noise = rng.normal(loc=1.0, scale=0.05, size=n)

    volume = base * weekly_effect * yearly_effect * trend_effect * festive_boost * noise
    volume = np.clip(volume, a_min=base * 0.3, a_max=None)  # avoid unrealistic near-zero days
    return np.round(volume, 1)


def main():
    cal = build_calendar(START_DATE, END_DATE)

    all_rows = []
    seed_counter = 0

    for outlet_idx, outlet in enumerate(OUTLETS):
        outlet_id = outlet["outlet_id"]
        region = outlet["region"]
        # Each outlet has a slightly different baseline size (scale factor)
        outlet_scale = 0.8 + 0.4 * ((outlet_idx % 5) / 4.0)  # range ~0.8-1.2

        for product in PRODUCTS:
            seed_counter += 1
            volume = simulate_volume(cal, product, outlet_scale, seed_offset=seed_counter)
            price = simulate_prices(cal["date"], BASE_PRICE[product], seed_offset=seed_counter + 100)

            revenue_noise = np.random.RandomState(RNG_SEED + seed_counter + 200).normal(
                loc=0.0, scale=50.0, size=len(cal)
            )
            revenue = volume * price + revenue_noise

            df = pd.DataFrame(
                {
                    "date": cal["date"],
                    "outlet_id": outlet_id,
                    "region": region,
                    "product": product,
                    "volume_liters": volume,
                    "price_per_liter": price,
                    "revenue_inr": np.round(revenue, 2),
                    "is_holiday": cal["is_holiday"].values,
                    "is_weekend": cal["is_weekend"].values,
                    "festive_flag": cal["festive_flag"].values,
                    "temperature_c": cal["temperature_c"].values,
                    "rainfall_mm": cal["rainfall_mm"].values,
                }
            )
            all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True)

    # Sanity checks: no missing dates per outlet-product, no missing values
    n_expected_days = len(cal)
    counts = full_df.groupby(["outlet_id", "product"])["date"].count()
    assert (counts == n_expected_days).all(), "Missing dates detected for some outlet-product pair!"
    assert full_df.isnull().sum().sum() == 0, "Unexpected missing values in generated data!"
    assert (full_df["volume_liters"] > 0).all(), "Non-positive volume detected!"

    # Reorder columns to match spec
    column_order = [
        "date",
        "outlet_id",
        "region",
        "product",
        "volume_liters",
        "revenue_inr",
        "price_per_liter",
        "is_holiday",
        "is_weekend",
        "festive_flag",
        "temperature_c",
        "rainfall_mm",
    ]
    full_df = full_df[column_order].sort_values(["date", "outlet_id", "product"]).reset_index(drop=True)
    full_df["date"] = full_df["date"].dt.strftime("%Y-%m-%d")

    full_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Synthetic fuel sales data generated: {OUTPUT_CSV}")
    print(f"Rows: {len(full_df):,} | Date range: {START_DATE} to {END_DATE}")
    print(f"Outlets: {len(OUTLETS)} | Regions: {sorted(set(o['region'] for o in OUTLETS))}")
    print(full_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
