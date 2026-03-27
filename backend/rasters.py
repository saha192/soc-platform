#!/usr/bin/env python
# coding: utf-8

import rasterio
import numpy as np
from pyproj import Transformer, CRS

TARGET_CRS = "EPSG:4326"

SAND_RASTER = r"F:\SOC_WEB_APP\data\sand.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif"
SILT_RASTER = r"F:\SOC_WEB_APP\data\silt.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif"
CLIMATE_RASTER = r"F:\SOC_WEB_APP\data\climate map\koppen_geiger_0p00833333.tif"

KOPPEN_GEIGER_CLASSES = {
    1: "Af",  2: "Am",  3: "Aw",
    4: "BWh", 5: "BWk", 6: "BSh", 7: "BSk",
    8: "Csa", 9: "Csb", 10: "Csc",
    11: "Cwa", 12: "Cwb", 13: "Cwc",
    14: "Cfa", 15: "Cfb", 16: "Cfc",
    17: "Dsa", 18: "Dsb", 19: "Dsc", 20: "Dsd",
    21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd",
    25: "Dfa", 26: "Dfb", 27: "Dfc", 28: "Dfd",
    29: "ET", 30: "EF"
}

KOPPEN_MAJOR_GROUPS = {
    "A": "Tropical",
    "B": "Arid",
    "C": "Temperate",
    "D": "Cold",
    "E": "Polar",
}

def _normalize_crs(crs_like):
    if crs_like is None:
        return None
    return CRS.from_user_input(crs_like)

def _transform_coords(coords, src_crs, dst_crs):
    """
    coords: list of (x, y) in src_crs
    returns coords in dst_crs
    """
    src = _normalize_crs(src_crs)
    dst = _normalize_crs(dst_crs)

    if src is None or dst is None:
        raise ValueError("Source CRS or destination CRS is undefined.")

    if src == dst:
        return coords

    transformer = Transformer.from_crs(src, dst, always_xy=True)
    xs, ys = zip(*coords)
    tx, ty = transformer.transform(xs, ys)
    return list(zip(tx, ty))

def _sample_raster_masked(path, coords, input_crs=TARGET_CRS):
    """
    Samples raster safely after transforming coords from input_crs
    to raster CRS if needed.
    """
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS defined: {path}")

        coords_for_raster = _transform_coords(coords, input_crs, src.crs)
        values = list(src.sample(coords_for_raster, masked=True))
        return values, src.crs

def sample_raster(path, coords, input_crs=TARGET_CRS):
    values, raster_crs = _sample_raster_masked(path, coords, input_crs=input_crs)

    out = []
    for v in values:
        val = v[0]
        if np.ma.is_masked(val):
            out.append(np.nan)
        else:
            out.append(float(val))

    return np.array(out, dtype=np.float32)

def sample_climate_classes(coords, input_crs=TARGET_CRS):
    """
    Sample Köppen-Geiger raster and return:
      - numeric class ids
      - detailed KG class (Af, BSh, Cwa, ...)
      - broad climate zone (Tropical, Arid, Temperate, Cold, Polar)
    """
    values, raster_crs = _sample_raster_masked(CLIMATE_RASTER, coords, input_crs=input_crs)

    kg_ids = []
    kg_classes = []
    climate_zones = []

    for v in values:
        val = v[0]

        if np.ma.is_masked(val):
            kg_ids.append(np.nan)
            kg_classes.append(None)
            climate_zones.append(None)
            continue

        class_id = int(val)
        kg_code = KOPPEN_GEIGER_CLASSES.get(class_id)

        kg_ids.append(class_id)
        kg_classes.append(kg_code)

        if kg_code is None:
            climate_zones.append(None)
        else:
            climate_zones.append(KOPPEN_MAJOR_GROUPS.get(kg_code[0]))

    return (
        np.array(kg_ids, dtype=object),
        np.array(kg_classes, dtype=object),
        np.array(climate_zones, dtype=object)
    )

def validate_raster_crs():
    """
    Optional startup check to confirm all expected rasters use TARGET_CRS.
    """
    for path in [SAND_RASTER, SILT_RASTER, CLIMATE_RASTER]:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"Raster has no CRS defined: {path}")

            raster_crs = CRS.from_user_input(src.crs)
            target_crs = CRS.from_user_input(TARGET_CRS)

            print(f"Raster: {path}")
            print(f"  CRS: {raster_crs}")

            if raster_crs != target_crs:
                print(f"  Warning: raster CRS differs from {TARGET_CRS}. Coordinates will be transformed before sampling.")
            else:
                print(f"  OK: raster CRS matches {TARGET_CRS}")