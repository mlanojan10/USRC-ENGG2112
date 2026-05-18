import os
import xarray as xr
import pandas as pd


# =========================
# 0. FILE PATHS
# =========================

RAW_DIR = "data_raw"
PROCESSED_DIR = "data_processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

INPUT_FILE = os.path.join(RAW_DIR, "parramatta_humidity_2020_2039_raw.nc")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "parramatta_humidity_2020.csv")


# =========================
# 1. LOAD HUMIDITY NETCDF
# =========================

ds = xr.open_dataset(INPUT_FILE)

print("NetCDF dataset:")
print(ds)

print("\nAvailable variables:")
print(list(ds.data_vars))

print("\nAvailable coordinates:")
print(list(ds.coords))


# =========================
# 2. IDENTIFY VARIABLE NAMES
# =========================

humidity_var = list(ds.data_vars)[0]

if "lat" in ds.coords:
    lat_name = "lat"
elif "latitude" in ds.coords:
    lat_name = "latitude"
else:
    raise ValueError("Could not find latitude coordinate.")

if "lon" in ds.coords:
    lon_name = "lon"
elif "longitude" in ds.coords:
    lon_name = "longitude"
else:
    raise ValueError("Could not find longitude coordinate.")

if "time" in ds.coords:
    time_name = "time"
else:
    raise ValueError("Could not find time coordinate.")

print("\nUsing humidity variable:", humidity_var)
print("Using latitude coordinate:", lat_name)
print("Using longitude coordinate:", lon_name)
print("Using time coordinate:", time_name)


# =========================
# 3. CHECK FILE IS NOT EMPTY
# =========================

if ds.sizes.get(lat_name, 0) == 0 or ds.sizes.get(lon_name, 0) == 0:
    raise ValueError(
        "Downloaded humidity NetCDF has zero lat/lon grid cells. "
        "Redownload using a larger Parramatta bounding box."
    )


# =========================
# 4. EXTRACT NEAREST PARRAMATTA GRID CELL
# =========================

parramatta_lat = -33.82
parramatta_lon = 151.00

humidity_point = ds[humidity_var].sel(
    {
        lat_name: parramatta_lat,
        lon_name: parramatta_lon,
    },
    method="nearest"
)

print("\nSelected nearest grid cell:")
print(humidity_point)


# =========================
# 5. CONVERT TO DATAFRAME
# =========================

humidity_df = humidity_point.to_dataframe().reset_index()

print("\nRaw extracted dataframe:")
print(humidity_df.head())
print(humidity_df.columns)


# =========================
# 6. CLEAN COLUMN NAMES
# =========================

humidity_df = humidity_df.rename(columns={
    time_name: "date",
    humidity_var: "humidity_mean",
})

humidity_df = humidity_df[["date", "humidity_mean"]].copy()

humidity_df["date"] = pd.to_datetime(humidity_df["date"], errors="coerce")
humidity_df["humidity_mean"] = pd.to_numeric(
    humidity_df["humidity_mean"],
    errors="coerce"
)

humidity_df = humidity_df.dropna(subset=["date", "humidity_mean"])


# =========================
# 7. FILTER ONLY 2020
# =========================

humidity_df = humidity_df[
    (humidity_df["date"] >= "2020-01-01") &
    (humidity_df["date"] < "2021-01-01")
].copy()


# =========================
# 8. CONVERT TO DAILY MEAN
# =========================

humidity_df = (
    humidity_df
    .groupby(humidity_df["date"].dt.date)["humidity_mean"]
    .mean()
    .reset_index()
)

humidity_df["date"] = pd.to_datetime(humidity_df["date"])


# =========================
# 9. SAVE CLEAN CSV
# =========================

humidity_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved clean humidity file:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(humidity_df.head())

print("\nLast 5 rows:")
print(humidity_df.tail())

print("\nRows saved:")
print(len(humidity_df))