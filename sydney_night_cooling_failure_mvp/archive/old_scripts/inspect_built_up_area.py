from pathlib import Path
import geopandas as gpd
import fiona

BASE_DIR = Path(__file__).resolve().parents[1]
BUILT_UP_FILE = BASE_DIR / "data_raw" / "built_up_area" / "Built_Up_Areas.gpkg"

if not BUILT_UP_FILE.exists():
    raise FileNotFoundError(f"Missing file: {BUILT_UP_FILE}")

print("Built-up area file:")
print(BUILT_UP_FILE)

print("\nAvailable layers:")
layers = fiona.listlayers(BUILT_UP_FILE)
print(layers)

layer = layers[0]
print(f"\nReading first layer: {layer}")

gdf = gpd.read_file(BUILT_UP_FILE, layer=layer)

print("\nShape:")
print(gdf.shape)

print("\nCRS:")
print(gdf.crs)

print("\nColumns:")
print(gdf.columns.tolist())

print("\nFirst 5 rows:")
print(gdf.head())

print("\nGeometry types:")
print(gdf.geometry.geom_type.value_counts())

print("\nBounds:")
print(gdf.total_bounds)

print("\nArea-like columns:")
area_cols = [c for c in gdf.columns if "area" in c.lower() or "sq" in c.lower() or "sqm" in c.lower()]
print(area_cols)

if area_cols:
    print("\nArea-like column preview:")
    print(gdf[area_cols].head())
