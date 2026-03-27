#!/usr/bin/env python
# coding: utf-8

import ee
import math

ee.Initialize()

# ============================================================
# GLOBAL TARGET OUTPUT CRS
# ============================================================
TARGET_CRS = "EPSG:4326"
TARGET_SCALE = 10  # output sampling scale in meters

# ============================================================
# HELPERS
# ============================================================
def reproject_to_target(img, scale=TARGET_SCALE):
    """Reproject final output image to target CRS."""
    return img.toFloat().reproject(crs=TARGET_CRS, scale=scale)

def cast_float(img):
    return img.toFloat()

# ============================================================
# Sentinel-2 Cloud Mask (IDENTICAL TO TRAINING)
# ============================================================
def mask_s2(img):
    scl = img.select('SCL')

    mask = (
        scl.eq(2)
        .Or(scl.eq(4))
        .Or(scl.eq(5))
        .Or(scl.eq(6))
        .Or(scl.eq(7))
    )

    return (
        img.updateMask(mask)
        .select([
            'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
            'B8', 'B8A', 'B11', 'B12'
        ])
        .multiply(0.0001)
        .toFloat()
    )

# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================
def get_gee_features(polygon_geojson, start_date, end_date):

    geometry = ee.Geometry(polygon_geojson["geometry"])

    # ========================================================
    # Sentinel-1
    # ========================================================
    s1 = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .select(['VH', 'VV'])
        .map(cast_float)
    )

    s1Med_native = s1.median().toFloat()

    # ========================================================
    # Sentinel-2
    # ========================================================
    s2 = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
        .map(mask_s2)
    )

    s2Med_native = s2.median().toFloat()

    s2Bands_native = s2Med_native.select([
        'B11', 'B12', 'B2', 'B3', 'B4',
        'B5', 'B6', 'B7', 'B8', 'B8A'
    ]).toFloat()

    # Use Sentinel-2 projection as reference output grid
    s2_projection = s2.first().select('B2').projection()

    # ========================================================
    # DEM & Terrain (compute in native DEM projection)
    # ========================================================
    dem_native = ee.Image('NASA/NASADEM_HGT/001').select('elevation').toFloat()
    dem_proj = dem_native.projection()

    terrain_native = ee.Terrain.products(dem_native)

    elevation_native = dem_native.rename('elevation').toFloat()
    slope_native = terrain_native.select('slope').rename('slope').toFloat()
    aspect_native = terrain_native.select('aspect').rename('aspect').toFloat()

    slopeRad_native = slope_native.multiply(math.pi / 180.0).rename('slope_rad').toFloat()
    tanSlope_native = slopeRad_native.tan().rename('tan_slope').toFloat()

    aspectRad_native = aspect_native.multiply(math.pi / 180.0).rename('aspect_rad').toFloat()
    aspectSin_native = aspectRad_native.sin().rename('aspect_sin').toFloat()
    aspectCos_native = aspectRad_native.cos().rename('aspect_cos').toFloat()

    # ========================================================
    # Flow Accumulation
    # ========================================================
    flowAcc_native = ee.Image('MERIT/Hydro/v1_0_1').select('upa').rename('flow_acc').toFloat()
    logFlowAcc_native = flowAcc_native.add(1).log().rename('log_flow_acc').toFloat()

    # ========================================================
    # TWI & SPI
    # ========================================================
    twi_native = (
        flowAcc_native.add(1)
        .divide(tanSlope_native.add(0.001))
        .log()
        .rename('TWI')
        .toFloat()
    )

    spi_native = (
        flowAcc_native.multiply(tanSlope_native)
        .rename('SPI')
        .toFloat()
    )

    # ========================================================
    # Curvature
    # ========================================================
    laplacian_native = dem_native.convolve(ee.Kernel.laplacian8()).rename('laplacian').toFloat()

    profileCurv_native = (
        laplacian_native.divide(tanSlope_native.add(0.001))
        .rename('profile_curvature')
        .toFloat()
    )

    planCurv_native = (
        laplacian_native.multiply(tanSlope_native)
        .rename('plan_curvature')
        .toFloat()
    )

    # ========================================================
    # TPI
    # ========================================================
    def make_tpi(radius, name):
        meanElev = dem_native.reduceNeighborhood(
            reducer=ee.Reducer.mean(),
            kernel=ee.Kernel.circle(radius, 'meters')
        )
        return dem_native.subtract(meanElev).rename(name).toFloat()

    TPI_300m_native = make_tpi(300, 'TPI_300m')
    TPI_600m_native = make_tpi(600, 'TPI_600m')

    # ========================================================
    # Elevation Stats
    # ========================================================
    elevStats90_native = dem_native.reduceNeighborhood(
        reducer=(
            ee.Reducer.mean()
            .combine(ee.Reducer.max(), '', True)
            .combine(ee.Reducer.min(), '', True)
            .combine(ee.Reducer.stdDev(), '', True)
        ),
        kernel=ee.Kernel.circle(90, 'meters')
    ).rename([
        'elev_mean_90m',
        'elev_max_90m',
        'elev_min_90m',
        'elev_std_90m'
    ]).toFloat()

    # ========================================================
    # LS Factor
    # ========================================================
    LS_factor_native = (
        flowAcc_native.pow(0.4)
        .multiply(slopeRad_native.sin().pow(1.3))
        .rename('LS_factor')
        .toFloat()
    )

    # ========================================================
    # CHIRPS Rainfall → R Factor
    # ========================================================
    chirps = (
        ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
        .filterDate('2000-01-01', '2023-12-31')
        .filterBounds(geometry)
        .map(cast_float)
    )

    meanAnnualRain_native = (
        chirps.sum()
        .divide(24)
        .rename('mean_annual_rain_mm')
        .toFloat()
    )

    R_factor_native = (
        meanAnnualRain_native.multiply(0.5)
        .rename('R_factor')
        .toFloat()
    )

    # ========================================================
    # Lat / Lon
    # ========================================================
    latLong_native = ee.Image.pixelLonLat().select(
        ['longitude', 'latitude'],
        ['long', 'lat']
    ).toFloat()

    # ========================================================
    # REPROJECT ONLY FINAL BANDS TO TARGET CRS
    # ========================================================
    s2Bands = reproject_to_target(s2Bands_native)
    s1Med = reproject_to_target(s1Med_native)

    aspectCos = reproject_to_target(aspectCos_native)
    aspectSin = reproject_to_target(aspectSin_native)
    elevation = reproject_to_target(elevation_native)
    slope = reproject_to_target(slope_native)

    twi = reproject_to_target(twi_native)
    spi = reproject_to_target(spi_native)
    logFlowAcc = reproject_to_target(logFlowAcc_native)

    planCurv = reproject_to_target(planCurv_native)
    profileCurv = reproject_to_target(profileCurv_native)

    TPI_300m = reproject_to_target(TPI_300m_native)
    TPI_600m = reproject_to_target(TPI_600m_native)

    elevStats90 = reproject_to_target(elevStats90_native)

    LS_factor = reproject_to_target(LS_factor_native)
    R_factor = reproject_to_target(R_factor_native)

    latLong = reproject_to_target(latLong_native)

    # ========================================================
    # Final Stack
    # ========================================================
    finalImage = ee.Image.cat([
        s2Bands,
        s1Med,
        aspectCos,
        aspectSin,
        elevation,
        slope,
        twi,
        spi,
        logFlowAcc,
        planCurv,
        profileCurv,
        TPI_300m,
        TPI_600m,
        elevStats90,
        LS_factor,
        R_factor,
        latLong
    ]).toFloat()

    # ========================================================
    # Sampling in EPSG:4326
    # ========================================================
    samples = finalImage.sample(
        region=geometry,
        scale=TARGET_SCALE,
        projection=TARGET_CRS,
        geometries=False
    )

    return samples.getInfo()