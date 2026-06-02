import requests
import pandas as pd
from pathlib import Path
import time

# Chicago coordinates
LATITUDE = 41.8781
LONGITUDE = -87.6298

# Date range matching our crime dataset
START_YEAR = 2008
END_YEAR = 2025

# Output path
OUTPUT_PATH = Path("../../data/raw/weather_chicago.parquet")


def fetch_weather_for_year(year: int) -> pd.DataFrame:
    """Fetch daily weather data for Chicago for a given year."""

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": [
            "temperature_2m_mean",
            "precipitation_sum",
            "relative_humidity_2m_mean",
            "wind_speed_10m_mean"
        ],
        "timezone": "America/Chicago"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"API error for year {year}: {response.status_code} — {response.text}")

    data = response.json()["daily"]

    df = pd.DataFrame({
        "date": data["time"],
        "temperature": data["temperature_2m_mean"],
        "precipitation": data["precipitation_sum"],
        "humidity": data["relative_humidity_2m_mean"],
        "wind_speed": data["wind_speed_10m_mean"]
    })

    df["date"] = pd.to_datetime(df["date"])

    return df


def fetch_all_weather() -> pd.DataFrame:
    """Fetch weather data for all years and combine into one DataFrame."""

    all_years = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Fetching weather data for {year}...")

        try:
            df = fetch_weather_for_year(year)
            all_years.append(df)
            print(f"  ✓ {year} — {len(df)} days fetched")

        except Exception as e:
            print(f"  ✗ {year} failed — {e}")

        # Be polite to the API — small delay between requests
        time.sleep(0.5)

    combined = pd.concat(all_years, ignore_index=True)
    return combined


def save_weather(df: pd.DataFrame) -> None:
    """Save weather DataFrame to parquet."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    print("Starting Chicago weather data fetch...\n")
    weather_df = fetch_all_weather()

    print(f"\nFetch complete.")
    print(f"Total rows: {len(weather_df)}")
    print(f"Date range: {weather_df['date'].min()} → {weather_df['date'].max()}")
    print(f"Missing values:\n{weather_df.isnull().sum()}")

    save_weather(weather_df)