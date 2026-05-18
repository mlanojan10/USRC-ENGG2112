from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path(__file__).resolve().parents[1]

NC_DIR = BASE_DIR / "data_raw" / "parramatta_era5_unzipped"
SUBURB_FILE = BASE_DIR / "metadata" / "suburbs_3_mvp.csv"
OUT_FILE = BASE_DIR / "data_processed" / "sydney_era5_hourly_2020.csv"

nc_files = list(NC_DIR.glob("*.nc"))

if not nc_files:
    raise FileNotFoundError(f"No .nc files found in {NC_DIR}")

NC_FILE = nc_files[0]

print("Using NetCDF file:")
print(NC_FILE)

suburbs = pd.read_csv(SUBURB_FILE)
ds = xr.open_dataset(NC_FILE)

print("\nDataset variables:")
print(list(ds.data_vars))

print("\nDataset coordinates:")
print(list(ds.coords))

# Common ERA5 variable names.
var_map = {
    "temperature": ["temperature", "t2m", "2m_temperature"],
    "dewpoint": ["dewpoint", "d2m", "2m_dewpoint_temperature"],
    "u_wind": ["u10", "10m_u_component_of_wind"],
    "v_wind": ["v10", "10m_v_component_of_wind"],
    "air_pressure": ["air_pressure", "sp", "surface_pressure"],
    "precipitation": ["precipitation", "tp", "total_precipitation"],
}

def find_var(options):
    for name in options:
        if name in ds.data_vars:
            return name
    return None

found = {key: find_var(options) for key, options in var_map.items()}

print("\nMatched variables:")
print(found)

temp_var = found["temperature"]
dew_var = found["dewpoint"]
u_var = found["u_wind"]
v_var = found["v_wind"]
pressure_var = found["air_pressure"]
precip_var = found["precipitation"]

if temp_var is None:
    raise ValueError("Could not find temperature variable. Check the printed dataset variables.")

# Detect coordinate names.
if "latitude" in ds.coords:
    lat_name = "latitude"
elif "lat" in ds.coords:
    lat_name = "lat"
else:
    raise ValueError("Could not find latitude coordinate.")

if "longitude" in ds.coords:
    lon_name = "longitude"
elif "lon" in ds.coords:
    lon_name = "lon"
else:
    raise ValueError("Could not find longitude coordinate.")

if "time" in ds.coords:
    time_name = "time"
elif "valid_time" in ds.coords:
    time_name = "valid_time"
else:
    raise ValueError("Could not find time coordinate.")

rows = []

for _, s in suburbs.iterrows():
    suburb = s["suburb"]
    lat = float(s["suburb_lat"])
    lon = float(s["suburb_lon"])

    print(f"\nExtracting nearest ERA5 grid cell for {suburb}: {lat}, {lon}")

    point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")

    df = pd.DataFrame()
    df["datetime"] = pd.to_datetime(point[time_name].values)
    df["suburb"] = suburb
    df["suburb_lat"] = lat
    df["suburb_lon"] = lon
    df["source"] = "ERA5 nearest grid cell"

    temp = point[temp_var].values
    if np.nanmean(temp) > 100:
        temp = temp - 273.15
    df["temperature"] = temp

    if dew_var is not None:
        dew = point[dew_var].values
        if np.nanmean(dew) > 100:
            dew = dew - 273.15
        df["dewpoint"] = dew
    else:
        df["dewpoint"] = np.nan

    if dew_var is not None:
        T = df["temperature"]
        Td = df["dewpoint"]
        rh = 100 * (
            np.exp((17.625 * Td) / (243.04 + Td)) /
            np.exp((17.625 * T) / (243.04 + T))
        )
        df["humidity"] = rh.clip(0, 100)
    else:
        df["humidity"] = np.nan

    if u_var is not None and v_var is not None:
        u = point[u_var].values
        v = point[v_var].values
        df["wind_speed"] = np.sqrt(u**2 + v**2)
    else:
        df["wind_speed"] = np.nan

    if pressure_var is not None:
        pressure = point[pressure_var].values
        if np.nanmean(pressure) > 2000:
            pressure = pressure / 100.0
        df["air_pressure"] = pressure
    else:
        df["air_pressure"] = np.nan

    if precip_var is not None:
        precip = point[precip_var].values
        if np.nanmax(precip) < 10:
            precip = precip * 1000.0
        df["precipitation"] = precip
    else:
        df["precipitation"] = 0.0

    rows.append(df)

out = pd.concat(rows, ignore_index=True)

out["datetime"] = pd.to_datetime(out["datetime"])
out = out[(out["datetime"] >= "2020-01-01") & (out["datetime"] < "2021-01-01")].copy()

out = out[
    [
        "datetime",
        "suburb",
        "suburb_lat",
        "suburb_lon",
        "source",
        "temperature",
        "dewpoint",
        "humidity",
        "wind_speed",
        "air_pressure",
        "precipitation",
    ]
]

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_FILE, index=False)

print("\nSaved:")
print(OUT_FILE)

print("\nFirst rows:")
print(out.head())

print("\nRows per suburb:")
print(out["suburb"].value_counts())
