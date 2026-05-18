from pathlib import Path
import zipfile
import tempfile

import numpy as np
import pandas as pd
import xarray as xr


# =========================
# 0. PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_ERA5_DIR = BASE_DIR / "data_raw" / "era5"
METADATA_FILE = BASE_DIR / "metadata" / "suburbs_12_mvp.csv"
PROCESSED_DIR = BASE_DIR / "data_processed"

OUTPUT_FILE = PROCESSED_DIR / "sydney_era5_hourly_2020.csv"
OUTPUT_FILE_12 = PROCESSED_DIR / "sydney_12_suburbs_era5_hourly_2020.csv"

PROCESSED_DIR.mkdir(exist_ok=True)


# =========================
# 1. LOAD METADATA
# =========================

if not METADATA_FILE.exists():
    raise FileNotFoundError(f"Missing metadata file: {METADATA_FILE}")

suburbs = pd.read_csv(METADATA_FILE)

required_cols = [
    "suburb",
    "suburb_lat",
    "suburb_lon",
]

missing = [c for c in required_cols if c not in suburbs.columns]
if missing:
    raise ValueError(f"Missing required columns in metadata: {missing}")

print("Loaded suburbs:")
print(suburbs[["suburb", "suburb_lat", "suburb_lon"]])


# =========================
# 2. HELPERS
# =========================

def safe_name(name):
    return str(name).lower().replace(" ", "_")


def open_era5_dataset(file_path):
    """
    Opens a normal NetCDF file.
    If the file is actually a ZIP file with .nc extension, extract and open the first .nc.
    """
    if zipfile.is_zipfile(file_path):
        print(f"File is a ZIP disguised as NetCDF, extracting: {file_path.name}")

        temp_dir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        nc_files = list(temp_dir.rglob("*.nc"))

        if not nc_files:
            raise FileNotFoundError(f"No .nc file found inside zipped file: {file_path}")

        extracted_nc = nc_files[0]
        print(f"Using extracted file: {extracted_nc}")
        return xr.open_dataset(extracted_nc)

    return xr.open_dataset(file_path)


def find_time_column(df):
    possible_time_cols = [
        "valid_time",
        "time",
        "datetime",
        "date",
    ]

    for col in possible_time_cols:
        if col in df.columns:
            return col

    datetime_like = [
        col for col in df.columns
        if np.issubdtype(df[col].dtype, np.datetime64)
    ]

    if datetime_like:
        return datetime_like[0]

    raise ValueError(
        "Could not find time column. Columns were: "
        + str(df.columns.tolist())
    )


def kelvin_to_celsius(series):
    return series - 273.15


def calculate_relative_humidity(temp_c, dewpoint_c):
    """
    Approximate relative humidity from temperature and dewpoint.
    Formula uses Magnus approximation.
    """
    es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    e = 6.112 * np.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
    rh = 100 * (e / es)
    return rh.clip(lower=0, upper=100)


# =========================
# 3. COMBINE FILES
# =========================

all_frames = []

for _, row in suburbs.iterrows():
    suburb = str(row["suburb"])
    lat = float(row["suburb_lat"])
    lon = float(row["suburb_lon"])

    filename = row.get("era5_filename", f"{safe_name(suburb)}_era5_hourly_2020.nc")
    nc_file = RAW_ERA5_DIR / filename

    if not nc_file.exists():
        fallback = RAW_ERA5_DIR / f"{safe_name(suburb)}_era5_hourly_2020.nc"
        if fallback.exists():
            nc_file = fallback
        else:
            raise FileNotFoundError(f"Missing NetCDF file for {suburb}: {nc_file}")

    print("\n" + "=" * 70)
    print(f"Processing {suburb}")
    print(nc_file)
    print("=" * 70)

    ds = open_era5_dataset(nc_file)

    print("Dataset variables:")
    print(list(ds.data_vars))

    print("Dataset coordinates:")
    print(list(ds.coords))

    # Convert to dataframe.
    temp_df = ds.to_dataframe().reset_index()

    time_col = find_time_column(temp_df)

    # Standardise time.
    temp_df["datetime_utc"] = pd.to_datetime(temp_df[time_col], errors="coerce", utc=True)

    # Convert UTC to Sydney local time.
    temp_df["datetime"] = (
        temp_df["datetime_utc"]
        .dt.tz_convert("Australia/Sydney")
        .dt.tz_localize(None)
    )

    # Filter to Sydney local calendar year 2020.
    temp_df = temp_df[
        (temp_df["datetime"] >= "2020-01-01 00:00:00")
        & (temp_df["datetime"] <= "2020-12-31 23:00:00")
    ].copy()

    # Remove duplicated times if xarray produced extra coordinate rows.
    temp_df = temp_df.drop_duplicates(subset=["datetime"]).copy()

    # Variable name mapping from ERA5 short names.
    variable_map = {
        "t2m": "temperature",
        "d2m": "dewpoint",
        "u10": "u10",
        "v10": "v10",
        "sp": "surface_pressure",
        "tp": "total_precipitation",
    }

    missing_vars = [v for v in variable_map.keys() if v not in temp_df.columns]
    if missing_vars:
        raise ValueError(
            f"Missing ERA5 variables for {suburb}: {missing_vars}. "
            f"Available columns: {temp_df.columns.tolist()}"
        )

    out = pd.DataFrame()
    out["datetime"] = temp_df["datetime"]
    out["suburb"] = suburb
    out["suburb_lat"] = lat
    out["suburb_lon"] = lon
    out["source"] = "ERA5 hourly time-series"

    out["temperature"] = kelvin_to_celsius(pd.to_numeric(temp_df["t2m"], errors="coerce"))
    out["dewpoint"] = kelvin_to_celsius(pd.to_numeric(temp_df["d2m"], errors="coerce"))

    u10 = pd.to_numeric(temp_df["u10"], errors="coerce")
    v10 = pd.to_numeric(temp_df["v10"], errors="coerce")

    out["humidity"] = calculate_relative_humidity(
        out["temperature"],
        out["dewpoint"],
    )

    out["wind_speed"] = np.sqrt(u10 ** 2 + v10 ** 2)

    # Pa to hPa.
    out["air_pressure"] = pd.to_numeric(temp_df["sp"], errors="coerce") / 100.0

    # m to mm.
    out["precipitation"] = pd.to_numeric(temp_df["tp"], errors="coerce") * 1000.0

    out = out.sort_values("datetime").reset_index(drop=True)

    print(f"Rows for {suburb}: {len(out)}")
    print(f"Datetime range: {out['datetime'].min()} to {out['datetime'].max()}")
    print(out.head())

    all_frames.append(out)


# =========================
# 4. SAVE OUTPUT
# =========================

combined = pd.concat(all_frames, ignore_index=True)
combined = combined.sort_values(["suburb", "datetime"]).reset_index(drop=True)

combined.to_csv(OUTPUT_FILE, index=False)
combined.to_csv(OUTPUT_FILE_12, index=False)

print("\n" + "=" * 70)
print("COMBINED ERA5 FILE CREATED")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)
print(OUTPUT_FILE_12)

print("\nRows per suburb:")
print(combined["suburb"].value_counts().sort_index())

print("\nDatetime range by suburb:")
print(
    combined
    .groupby("suburb")["datetime"]
    .agg(["min", "max", "count"])
    .sort_index()
)

print("\nColumns:")
print(combined.columns.tolist())
