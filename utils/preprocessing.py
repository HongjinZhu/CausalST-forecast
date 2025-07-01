import zipfile
import xarray as xr
import pandas as pd
import numpy as np
import os

# === File paths ===
PRESSURE_FILE = "/mnt/e/CERA/CausalST-forecast/53b532235fc47f346e14025392cd450f.nc"
SINGLE_NETCDF = "/mnt/e/CERA/CausalST-forecast/data_stream-oper_stepType-instant.nc"
ACCUM_FILE = "/mnt/e/CERA/CausalST-forecast/data_stream-oper_stepType-accum.nc"  # from zip

# === Solar site metadata ===
SOLAR_FIELDS = [
    {"name": "Bhadla", "lat": 27.5833, "lon": 71.4333},
    {"name": "Tengger", "lat": 37.3333, "lon": 103.8558},
    {"name": "Benban", "lat": 24.5561, "lon": 32.9016},
    {"name": "Cestas", "lat": 44.7255, "lon": -0.8157},
    {"name": "Balboa", "lat": 38.4533, "lon": -6.2260},
    {"name": "Topaz", "lat": 35.2426, "lon": -120.0096},
    {"name": "DesertSun", "lat": 33.8214, "lon": -115.3939},
    {"name": "Pirapora", "lat": -17.0891, "lon": -44.9878},
    {"name": "Nyngan", "lat": -31.5575, "lon": 147.2031},
]

# === Load and process datasets ===
print("📂 Loading NetCDF datasets...")
ds_accum = xr.open_dataset(ACCUM_FILE)  # cumulative ssrd
ssrd_hourly = ds_accum["ssrd"].diff(dim="time") / 3600  # J/m² → W/m²

ds_single = xr.open_dataset(SINGLE_NETCDF)  # t2m, tcc
ds_pressure = xr.open_dataset(PRESSURE_FILE)  # t, q, u, v

# === Extract site data ===
records = []
for field in SOLAR_FIELDS:
    sel = dict(latitude=field["lat"], longitude=field["lon"], method="nearest")
    try:
        df = xr.merge([
            ds_single["t2m"].sel(**sel),
            ds_single["tcc"].sel(**sel),
            ssrd_hourly.sel(**sel),  # ✅ corrected version
            ds_pressure["t"].sel(**sel),
            ds_pressure["q"].sel(**sel),
            ds_pressure["u"].sel(**sel),
            ds_pressure["v"].sel(**sel),
        ]).to_dataframe().reset_index()
    except KeyError as e:
        print(f"⚠️ Skipping {field['name']}: missing {e}")
        continue

    df["wind_speed"] = np.sqrt(df["u"]**2 + df["v"]**2)
    df["field"] = field["name"]
    df["lat"] = field["lat"]
    df["lon"] = field["lon"]
    df["hour"] = df["time"].dt.hour
    df["day_of_year"] = df["time"].dt.dayofyear

    records.append(df)

# === Combine and save ===
df_all = pd.concat(records, ignore_index=True)
df_all = df_all.rename(columns={"ssrd": "ssrd_hourly"})

# Create output folder
os.makedirs("data", exist_ok=True)

# Save features
df_feat = df_all[[
    "time", "field", "lat", "lon",
    "t2m", "t", "q", "tcc", "u", "v", "wind_speed",
    "hour", "day_of_year"
]]
df_feat.to_csv("data/era5_features.csv", index=False)
print("✅ Features saved to 'data/era5_features.csv'")

# Save target
df_target = df_all[["time", "field", "ssrd_hourly"]]
df_target.to_csv("data/ssrd_target.csv", index=False)
print("✅ Target saved to 'data/ssrd_target.csv'")
