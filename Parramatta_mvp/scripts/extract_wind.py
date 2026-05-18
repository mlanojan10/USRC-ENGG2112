import xarray as xr
import pandas as pd


# =========================
# 1. LOAD WIND NETCDF
# =========================

nc_file = "parramatta_wind_2020_2039_raw.nc"

ds = xr.open_dataset(nc_file)

print("NetCDF dataset:")
print(ds)

print("\nAvailable variables:")
print(list(ds.data_vars))

print("\nAvailable coordinates:")
print(list(ds.coords))


# =========================
# 2. IDENTIFY VARIABLE NAMES
# =========================

# Common wind variable names:
# sfcWind = near-surface wind speed
# uas = eastward wind
# vas = northward wind

wind_var = list(ds.data_vars)[0]

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

print("\nUsing wind variable:", wind_var)
print("Using latitude coordinate:", lat_name)
print("Using longitude coordinate:", lon_name)
print("Using time coordinate:", time_name)


# =========================
# 3. CHECK FILE IS NOT EMPTY
# =========================

if ds.sizes.get(lat_name, 0) == 0 or ds.sizes.get(lon_name, 0) == 0:
    raise ValueError(
        "Downloaded NetCDF has zero lat/lon grid cells. "
        "Redownload using the larger Parramatta bounding box."
    )


# =========================
# 4. EXTRACT NEAREST PARRAMATTA GRID CELL
# =========================

parramatta_lat = -33.82
parramatta_lon = 151.00

wind_point = ds[wind_var].sel(
    {
        lat_name: parramatta_lat,
        lon_name: parramatta_lon
    },
    method="nearest"
)

print("\nSelected nearest grid cell:")
print(wind_point)


# =========================
# 5. CONVERT TO DATAFRAME
# =========================

wind_df = wind_point.to_dataframe().reset_index()

print("\nRaw extracted dataframe:")
print(wind_df.head())
print(wind_df.columns)


# =========================
# 6. CLEAN COLUMN NAMES
# =========================

wind_df = wind_df.rename(columns={
    time_name: "date",
    wind_var: "wind_speed_mean"
})

wind_df = wind_df[["date", "wind_speed_mean"]].copy()

wind_df["date"] = pd.to_datetime(wind_df["date"], errors="coerce")
wind_df["wind_speed_mean"] = pd.to_numeric(wind_df["wind_speed_mean"], errors="coerce")

wind_df = wind_df.dropna(subset=["date", "wind_speed_mean"])


# =========================
# 7. FILTER ONLY 2020
# =========================

wind_df = wind_df[
    (wind_df["date"] >= "2020-01-01") &
    (wind_df["date"] < "2021-01-01")
].copy()


# =========================
# 8. CONVERT TO DAILY MEAN
# =========================

wind_df = (
    wind_df
    .groupby(wind_df["date"].dt.date)["wind_speed_mean"]
    .mean()
    .reset_index()
)

wind_df["date"] = pd.to_datetime(wind_df["date"])


# =========================
# 9. SAVE CLEAN CSV
# =========================

wind_df.to_csv("parramatta_wind_2020.csv", index=False)

print("\nSaved clean wind file:")
print("parramatta_wind_2020.csv")

print("\nFirst 5 rows:")
print(wind_df.head())

print("\nLast 5 rows:")
print(wind_df.tail())

print("\nRows saved:")
print(len(wind_df))