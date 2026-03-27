#!/usr/bin/env python
# coding: utf-8

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import geopandas as gpd
from shapely.geometry import Point
from fastapi.middleware.cors import CORSMiddleware
from .gee import get_gee_features
from .rasters import (
    sample_raster,
    sample_climate_classes,
    validate_raster_crs,
    SAND_RASTER,
    SILT_RASTER,
    CLIMATE_RASTER,
)
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://[::1]:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SOC_DIR = r"F:\SOC_WEB_APP\backend\models\SOC"
BD_DIR = r"F:\SOC_WEB_APP\backend\models\BD"
SOIL_SHP = r"F:\SOC_WEB_APP\data\soil type map\DSMW_with_soil_type_epsg4326_defined.shp"

DEPTH_CM = 30.0

validate_raster_crs()

# ============================================================
# LOAD SOIL MAP
# ============================================================
SOIL_GDF = gpd.read_file(SOIL_SHP)

if SOIL_GDF.crs is None:
    SOIL_GDF = SOIL_GDF.set_crs("EPSG:4326", allow_override=True)
else:
    SOIL_GDF = SOIL_GDF.to_crs("EPSG:4326")

SOIL_GDF = SOIL_GDF[["soil_type", "geometry"]].copy()
SOIL_GDF = SOIL_GDF[SOIL_GDF.geometry.notnull()].copy()

# Build spatial index once
_ = SOIL_GDF.sindex

print(f"✅ Soil shapefile loaded: {len(SOIL_GDF)} polygons")

# ============================================================
# LOAD MODELS
# ============================================================
SOC_MODELS = {}
BD_MODELS = {}

for f in os.listdir(SOC_DIR):
    if f.endswith(".pkl"):
        obj = joblib.load(os.path.join(SOC_DIR, f))
        # expected stored key: obj["soil_type"]
        SOC_MODELS[str(obj["soil_type"]).strip()] = obj

for f in os.listdir(BD_DIR):
    if f.endswith(".joblib"):
        soil = os.path.splitext(f)[0].replace("BD_model_", "").strip()
        BD_MODELS[soil] = joblib.load(os.path.join(BD_DIR, f))

SUPPORTED_SOC_SOILS = set(SOC_MODELS.keys())
SUPPORTED_BD_SOILS = set(BD_MODELS.keys())

print("✅ SOC models loaded:", sorted(SUPPORTED_SOC_SOILS))
print("✅ BD models loaded:", sorted(SUPPORTED_BD_SOILS))

# ============================================================
# REQUEST SCHEMA
# ============================================================
class PredictRequest(BaseModel):
    polygon: dict
    date: str

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6

    df["NDVI"] = (df["B8"] - df["B4"]) / (df["B8"] + df["B4"] + eps)
    df["EVI"] = 2.5 * (df["B8"] - df["B4"]) / (
        df["B8"] + 6 * df["B4"] - 7.5 * df["B2"] + 1
    )
    df["SAVI"] = 1.5 * (df["B8"] - df["B4"]) / (
        df["B8"] + df["B4"] + 0.5
    )

    df["NDRE"] = (df["B8"] - df["B5"]) / (df["B8"] + df["B5"] + eps)
    df["CI_RE"] = (df["B8"] / (df["B5"] + eps)) - 1

    df["BSI"] = (
        (df["B11"] + df["B4"] - df["B8"] - df["B2"]) /
        (df["B11"] + df["B4"] + df["B8"] + df["B2"] + eps)
    )

    df["NDTI"] = (df["B11"] - df["B12"]) / (df["B11"] + df["B12"] + eps)
    df["BI"] = np.sqrt(df["B4"] ** 2 + df["B3"] ** 2)

    df["NDWI"] = (df["B8"] - df["B11"]) / (df["B8"] + df["B11"] + eps)
    df["LSWI"] = (df["B8"] - df["B12"]) / (df["B8"] + df["B12"] + eps)

    df["VV_VH_ratio"] = df["VV"] / (df["VH"] + eps)
    df["VV_minus_VH"] = df["VV"] - df["VH"]
    df["log_VV"] = np.log(np.abs(df["VV"]) + 1)
    df["log_VH"] = np.log(np.abs(df["VH"]) + 1)

    df["slope_norm"] = np.log1p(df["slope"])
    df["elev_range_90m"] = df["elev_max_90m"] - df["elev_min_90m"]

    return df.replace([np.inf, -np.inf], np.nan)

# ============================================================
# DATE HELPERS
# ============================================================
def _gee_window(date_str: str, days_back: int = 30):
    d = datetime.fromisoformat(date_str).date()
    start = (d - timedelta(days=days_back)).isoformat()
    end = (d + timedelta(days=1)).isoformat()
    return start, end

def _last_12_calendar_month_windows(date_str: str):
    req_ts = pd.Timestamp(date_str).normalize()
    req_month_start = req_ts.replace(day=1)

    windows = []

    for i in range(11, -1, -1):
        month_start = (req_month_start - pd.DateOffset(months=i)).normalize()

        if i == 0:
            month_end_exclusive = (req_ts + pd.Timedelta(days=1)).normalize()
        else:
            month_end_exclusive = (month_start + pd.DateOffset(months=1)).normalize()

        windows.append({
            "label": month_start.strftime("%b %Y"),
            "month": month_start.strftime("%Y-%m"),
            "start_date": month_start.date().isoformat(),
            "end_date": month_end_exclusive.date().isoformat()
        })

    return windows

# ============================================================
# SOIL LOOKUP
# ============================================================
def assign_soil_type_from_shapefile(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Input dataframe is empty while assigning soil type.")

    if "long" not in df.columns or "lat" not in df.columns:
        raise ValueError("Columns 'long' and 'lat' are required for soil lookup.")

    point_gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) for xy in zip(df["long"], df["lat"])],
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(
        point_gdf,
        SOIL_GDF,
        how="left",
        predicate="within"
    )

    # Boundary fallback: nearest polygon for any unmatched points
    missing_mask = joined["soil_type"].isna()
    if missing_mask.any():
        nearest = gpd.sjoin_nearest(
            joined.loc[missing_mask, ["geometry"]],
            SOIL_GDF,
            how="left",
            distance_col="dist_deg"
        )
        joined.loc[missing_mask, "soil_type"] = nearest["soil_type"].values

    joined["SOIL_CLASS"] = joined["soil_type"].astype(str).str.strip()

    missing_final = joined["SOIL_CLASS"].isna() | joined["SOIL_CLASS"].eq("") | joined["SOIL_CLASS"].eq("nan")
    if missing_final.any():
        n_missing = int(missing_final.sum())
        raise ValueError(f"Could not assign soil class to {n_missing} sampled points.")

    drop_cols = [c for c in ["index_right", "geometry"] if c in joined.columns]
    joined = joined.drop(columns=drop_cols)

    return pd.DataFrame(joined)

# ============================================================
# CORE HELPERS
# ============================================================

def _validate_encoder_categories(encoder, climate_values):
    """
    Check whether current CLIMATE_ZONE values are compatible with the
    categories seen during training.
    """
    if not hasattr(encoder, "categories_"):
        return

    known = {str(v) for v in encoder.categories_[0]}
    found = {str(v) for v in pd.Series(climate_values).dropna().unique()}
    unknown = sorted(found - known)

    if unknown:
        raise ValueError(
            f"Encountered CLIMATE_ZONE values not seen during training: {unknown}. "
            f"Encoder knows: {sorted(known)}. "
            "Either map Köppen classes to the training categories, or retrain the SOC models."
        )


def _prepare_feature_df_from_window(
    polygon_geojson: dict,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    gee_data = get_gee_features(polygon_geojson, start_date, end_date)

    features = gee_data.get("features", [])
    if not features:
        raise ValueError(f"No valid GEE samples found for window {start_date} to {end_date}.")

    rows = [f["properties"] for f in features]
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"Empty feature table for window {start_date} to {end_date}.")

    coords = list(zip(df["long"], df["lat"]))

    df["sand"] = sample_raster(SAND_RASTER, coords)
    df["silt"] = sample_raster(SILT_RASTER, coords)

    # Sample Köppen-Geiger climate raster
    kg_ids, kg_classes, climate_zones = sample_climate_classes(coords)

    df["KG_ID"] = kg_ids
    df["KG_CLASS"] = kg_classes
    df["CLIMATE_ZONE"] = climate_zones

    missing_climate = df["CLIMATE_ZONE"].isna() | df["KG_CLASS"].isna()
    if missing_climate.any():
        n_missing = int(missing_climate.sum())
        raise ValueError(
            f"Could not assign Köppen-Geiger climate class to {n_missing} sampled points "
            f"from climate raster: {CLIMATE_RASTER}"
        )

    # Assign actual soil class from DSMW shapefile
    df = assign_soil_type_from_shapefile(df)

    # Feature engineering after soil assignment
    df = feature_engineering(df)

    return df

def _predict_soc_from_window(
    polygon_geojson: dict,
    start_date: str,
    end_date: str,
    return_pixels: bool = False
):
    df = _prepare_feature_df_from_window(polygon_geojson, start_date, end_date)

    pixel_outputs = []
    all_stock_values = []

    soils_in_request = set(df["SOIL_CLASS"].dropna().astype(str).str.strip().unique())
    missing_soc = sorted(soils_in_request - SUPPORTED_SOC_SOILS)
    missing_bd = sorted(soils_in_request - SUPPORTED_BD_SOILS)

    if missing_soc:
        raise ValueError(
            "SOC models not available for soil classes found in the selected polygon: "
            f"{missing_soc}. Available SOC models: {sorted(SUPPORTED_SOC_SOILS)}"
        )

    if missing_bd:
        raise ValueError(
            "BD models not available for soil classes found in the selected polygon: "
            f"{missing_bd}. Available BD models: {sorted(SUPPORTED_BD_SOILS)}"
        )

    for soil, group in df.groupby("SOIL_CLASS"):
        soc_bundle = SOC_MODELS[soil]
        bd_model = BD_MODELS[soil]

        encoder = soc_bundle["encoder"]
        model = soc_bundle["model"]
        features = soc_bundle["features"]
        
        _validate_encoder_categories(encoder, group["CLIMATE_ZONE"])

        X_cat = encoder.transform(group[["CLIMATE_ZONE"]])
        X_cat = pd.DataFrame(
            X_cat,
            columns=encoder.get_feature_names_out(["CLIMATE_ZONE"]),
            index=group.index
        )

        # Keep only model-relevant inputs
        X_num = group.drop(columns=["CLIMATE_ZONE"], errors="ignore")
        X = pd.concat([X_num, X_cat], axis=1)
        X = X.reindex(columns=features, fill_value=0).astype(np.float32)

        soc_pred = model.predict(X)

        bd_input = pd.DataFrame({
            "orgc_value_avg": soc_pred,
            "sand_value_avg": group["sand"].values,
            "silt_value_avg": group["silt"].values
        })

        bd_pred = bd_model.predict(bd_input)

        stock = soc_pred * bd_pred * DEPTH_CM * 0.1
        stock = np.asarray(stock, dtype=float)

        all_stock_values.extend(stock.tolist())

        if return_pixels:
            g = group.reset_index(drop=True)
            for i, val in enumerate(stock):
                pixel_outputs.append({
                    "lat": float(g.loc[i, "lat"]),
                    "long": float(g.loc[i, "long"]),
                    "soil_class": str(soil),
                    "soc_stock": float(val)
                })

    if not all_stock_values:
        raise ValueError(f"No SOC stock values produced for window {start_date} to {end_date}.")

    farm_stock_mean = float(np.mean(all_stock_values))

    result = {
        "soc_stock_mean": round(farm_stock_mean, 2),
        "soc_stock_label": f"{farm_stock_mean:.2f} t/ha"
    }

    if return_pixels:
        result["pixels"] = pixel_outputs

    return result

def _predict_soc_snapshot_for_date(
    polygon_geojson: dict,
    date_str: str,
    return_pixels: bool = False
):
    """
    Single-date map logic:
    trailing 30-day window ending at requested date.
    """
    start_date, end_date = _gee_window(date_str, days_back=30)
    return _predict_soc_from_window(
        polygon_geojson=polygon_geojson,
        start_date=start_date,
        end_date=end_date,
        return_pixels=return_pixels
    )

# ============================================================
# ENDPOINT
# ============================================================
@app.post("/predict_soc")
def predict_soc(req: PredictRequest):
    if not req.date:
        raise HTTPException(status_code=400, detail="date is required (YYYY-MM-DD)")

    try:
        # Current selected-date map
        current = _predict_soc_snapshot_for_date(req.polygon, req.date, return_pixels=True)

        # Monthly calendar-window time series
        monthly_time_series = []
        month_windows = _last_12_calendar_month_windows(req.date)

        for w in month_windows:
            try:
                snap = _predict_soc_from_window(
                    polygon_geojson=req.polygon,
                    start_date=w["start_date"],
                    end_date=w["end_date"],
                    return_pixels=False
                )

                monthly_time_series.append({
                    "label": w["label"],
                    "month": w["month"],
                    "start_date": w["start_date"],
                    "end_date": w["end_date"],
                    "soc_stock_mean": snap["soc_stock_mean"]
                })

            except Exception as e:
                monthly_time_series.append({
                    "label": w["label"],
                    "month": w["month"],
                    "start_date": w["start_date"],
                    "end_date": w["end_date"],
                    "soc_stock_mean": None,
                    "error": str(e)
                })

        return {
            "soc_stock": current["soc_stock_label"],
            "soc_stock_mean": current["soc_stock_mean"],
            "pixels": current["pixels"],
            "monthly_time_series": monthly_time_series
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))