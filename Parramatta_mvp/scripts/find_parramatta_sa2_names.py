from pathlib import Path
import zipfile
import geopandas as gpd

BASE_DIR = Path(__file__).resolve().parents[1]

BOUNDARIES_DIR = BASE_DIR / "data_raw" / "boundaries"
SA2_ZIP = BOUNDARIES_DIR / "SA2_2021_AUST_SHP_GDA2020.zip"
SA2_UNZIPPED_DIR = BOUNDARIES_DIR / "SA2_2021_AUST_SHP_GDA2020"

SA2_UNZIPPED_DIR.mkdir(exist_ok=True)

with zipfile.ZipFile(SA2_ZIP, "r") as zip_ref:
    zip_ref.extractall(SA2_UNZIPPED_DIR)

shp_files = list(SA2_UNZIPPED_DIR.rglob("*.shp"))

if not shp_files:
    raise FileNotFoundError("No SA2 shapefile found.")

sa2 = gpd.read_file(shp_files[0])

print("SA2 names containing 'Parramatta':")
matches = sa2[sa2["SA2_NAME21"].astype(str).str.contains("Parramatta", case=False, na=False)]

print(matches[["SA2_CODE21", "SA2_NAME21", "SA3_NAME21", "SA4_NAME21", "AREASQKM21"]].to_string(index=False))

print("\nSA2 names containing 'Rosehill':")
matches = sa2[sa2["SA2_NAME21"].astype(str).str.contains("Rosehill", case=False, na=False)]

print(matches[["SA2_CODE21", "SA2_NAME21", "SA3_NAME21", "SA4_NAME21", "AREASQKM21"]].to_string(index=False))
