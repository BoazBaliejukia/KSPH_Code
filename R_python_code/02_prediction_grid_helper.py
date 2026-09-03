"""Prediction-grid helpers for the event ENM pipeline.

This module wraps the training covariate helper without changing it. The
prediction-grid workflow uses the same covariate imagery/reducer logic as the
training workflow, but owns its own temporary buffer distances, simplification
tolerances, and prediction-grid buffer asset names.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Sequence

import ee


_BASE_HELPER_PATH = Path(__file__).with_name("02_drivehelper_covariates.py")
_BASE_MODULE_NAME = "_event_enm_drive_covariates"
_SPEC = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _BASE_HELPER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load covariate helper from {_BASE_HELPER_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_BASE_MODULE_NAME] = _BASE
_SPEC.loader.exec_module(_BASE)


LAT_LONG_CRS = "EPSG:4326"
APPROX_KM_PER_DEGREE = 111.32

# Prediction-grid settings in approximate latitude/longitude degrees.
SCALED_RINGS = [
    (0.000, 0.090, "0_10km"),
    (0.090, 0.225, "10_25km"),
    (0.225, 0.449, "25_50km"),
]

# Degree equivalents of the simplification tolerances requested for the
# prediction-grid workflow: ~200 m, 500 m, and 3000 m at the equator.
BUFFER_GEOMETRY_ERROR_BY_DISTANCE_DEG = {
    0.090: 0.0018,
    0.225: 0.0045,
    0.449: 0.0270,
}

# Approximate degree cell sizes for Hansen reductions. These correspond to
# ~200 m, 500 m, and 2000 m at the equator.
RING_EXTRACTION_SCALES_DEG = {
    "0_10km": 0.0018,
    "10_25km": 0.0045,
    "25_50km": 0.0180,
}

HANSEN_PREDICTION_BANDS = [
    "forest_cover_prop",
    "flsy_prop",
    "fl1yp_prop",
    "fl2yp_prop",
    "frag_edge_prop",
]


def __getattr__(name: str):
    return getattr(_BASE, name)


def selected_scaled_rings(
    scaled_ring_suffixes: Sequence[str] | None = None,
) -> list[tuple[float, float, str]]:
    """Return prediction-grid ring definitions selected by suffix."""
    if scaled_ring_suffixes is None:
        return list(SCALED_RINGS)

    requested = [str(suffix).strip() for suffix in scaled_ring_suffixes if str(suffix).strip()]
    if not requested:
        return list(SCALED_RINGS)

    valid_suffixes = {suffix for _, _, suffix in SCALED_RINGS}
    invalid = sorted(set(requested).difference(valid_suffixes))
    if invalid:
        raise ValueError(f"Unknown scaled ring suffixes: {invalid}. Valid suffixes: {sorted(valid_suffixes)}")

    requested_set = set(requested)
    return [ring for ring in SCALED_RINGS if ring[2] in requested_set]


def degree_projection(scale_degrees: float | None = None) -> ee.Projection:
    """Return an EPSG:4326 projection, optionally with a degree grid scale."""
    if scale_degrees is None:
        return ee.Projection(LAT_LONG_CRS)
    return ee.Projection(
        LAT_LONG_CRS,
        [float(scale_degrees), 0, -180, 0, -float(scale_degrees), 90],
    )


def degree_grid_transform(scale_degrees: float) -> list[float]:
    """Return a stable global lon/lat transform for degree-spaced rasters."""
    return [float(scale_degrees), 0, -180, 0, -float(scale_degrees), 90]


def approx_degrees_to_km(degrees: float) -> float:
    """Convert approximate equatorial degrees to kilometers."""
    return float(degrees) * APPROX_KM_PER_DEGREE


def approx_degrees_to_m(degrees: float) -> int:
    """Convert approximate equatorial degrees to meters for display only."""
    return int(round(approx_degrees_to_km(degrees) * 1000))


def buffer_geometry_error_degrees(outer_degrees: float) -> float:
    """Return prediction-grid buffer simplification tolerance in degrees."""
    distances = sorted(BUFFER_GEOMETRY_ERROR_BY_DISTANCE_DEG)
    for distance in distances:
        if float(outer_degrees) <= distance:
            return BUFFER_GEOMETRY_ERROR_BY_DISTANCE_DEG[distance]
    return BUFFER_GEOMETRY_ERROR_BY_DISTANCE_DEG[distances[-1]]


def buffer_geometry_error_m(outer_m: int | float) -> int:
    """Return the approximate simplification tolerance in meters for display."""
    return approx_degrees_to_m(buffer_geometry_error_degrees(float(outer_m)))


def ring_scale_degrees(suffix: str) -> float:
    """Return the prediction-grid Hansen extraction scale in degrees."""
    return RING_EXTRACTION_SCALES_DEG.get(suffix, 0.0090)


def ring_scale_m(suffix: str) -> int:
    """Return the approximate Hansen extraction scale in meters for display."""
    return approx_degrees_to_m(ring_scale_degrees(suffix))


def scaled_family_scale_m(family: str, ring_suffix: str | None = None) -> int | float:
    """Return the reduction scale for one prediction-grid scaled covariate family."""
    if family in {"forest_cover", "forest_loss", "fragmentation"}:
        if ring_suffix is not None:
            return ring_scale_degrees(ring_suffix)
        return _BASE.HANSEN_BUFFER_SCALE_M
    if family == "population":
        return _BASE.LANDSCAN_NATIVE_SCALE_M

    return _BASE.EXTRACTION_SCALE_M


def land_mask_image(study_region: ee.Geometry) -> ee.Image:
    """Image mask with valid pixels only inside the mainland/no-lakes study region."""
    return ee.Image.constant(1).clip(study_region).selfMask().rename("land_mask")


def mask_image_to_region(image: ee.Image, study_region: ee.Geometry) -> ee.Image:
    """Mask an image outside the mainland/no-lakes study region."""
    return image.updateMask(land_mask_image(study_region))


def zero_fill_hansen_land_pixels(image: ee.Image, study_region: ee.Geometry) -> ee.Image:
    """Fill masked Hansen pixels with zero on land while keeping water masked."""
    land_mask = land_mask_image(study_region)
    return image.unmask(0).updateMask(land_mask).clip(study_region)


def simplify_study_region(study_region: ee.Geometry, tolerance_m: int) -> ee.Geometry:
    """Simplify a study-region geometry using a meter tolerance."""
    return study_region.simplify(ee.ErrorMargin(int(tolerance_m)))


def safe_asset_name_component(value: str) -> str:
    """Return a readable Earth Engine asset-name component."""
    cleaned = "".join(character if character.isalnum() else "_" for character in str(value).strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "study_area"


def simplified_study_region_asset_id(
    asset_root: str,
    tolerance_m: int,
    study_area_name: str = "study_area",
) -> str:
    """Return the table-asset id for the simplified prediction study region."""
    safe_name = safe_asset_name_component(study_area_name)
    return f"{asset_root.rstrip('/')}/{safe_name}_simplified_{int(tolerance_m)}m"


def export_study_region_to_asset(
    study_region: ee.Geometry,
    description: str,
    asset_id: str,
    tolerance_m: int,
    study_area_name: str = "study_area",
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Export one simplified study-region geometry as a table asset."""
    safe_name = safe_asset_name_component(study_area_name)
    feature = ee.Feature(
        study_region,
        {
            "name": study_area_name,
            "asset_name": safe_name,
            "simplify_tolerance_m": int(tolerance_m),
        },
    )
    return _BASE.export_table_to_asset(
        collection=ee.FeatureCollection([feature]),
        description=description,
        asset_id=asset_id,
        dry_run=dry_run,
        overwrite=overwrite,
        max_vertices=1_000_000,
    )


def ring_geometry(feature: ee.Feature, inner_degrees: float, outer_degrees: float) -> ee.Geometry:
    """Create a simplified approximate-degree polygon buffer or donut buffer."""
    point = feature.geometry()
    max_error = ee.ErrorMargin(buffer_geometry_error_degrees(outer_degrees), "projected")
    projection = degree_projection()
    outer = point.buffer(float(outer_degrees), max_error, projection)
    if inner_degrees == 0:
        return outer.simplify(max_error, projection)
    inner = point.buffer(float(inner_degrees), max_error, projection)
    return outer.difference(inner, max_error, projection).simplify(max_error, projection)


def make_prediction_buffer_collection(
    grid_points: ee.FeatureCollection,
    ring_suffix: str,
    study_region: ee.Geometry | None = None,
) -> ee.FeatureCollection:
    """Create one simplified donut-buffer FeatureCollection from prediction grid points."""
    ring_lookup = {suffix: (inner_degrees, outer_degrees) for inner_degrees, outer_degrees, suffix in SCALED_RINGS}
    if ring_suffix not in ring_lookup:
        raise ValueError(f"Unknown ring suffix: {ring_suffix}")

    inner_degrees, outer_degrees = ring_lookup[ring_suffix]

    def buffer_feature(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        max_error = ee.ErrorMargin(buffer_geometry_error_degrees(outer_degrees), "projected")
        projection = degree_projection()
        geometry = ring_geometry(feature, inner_degrees, outer_degrees)
        if study_region is not None:
            geometry = geometry.intersection(study_region, max_error, projection).simplify(max_error, projection)
        return feature.set(
            {
                "buffer_ring": ring_suffix,
                "buffer_inner_degrees": inner_degrees,
                "buffer_outer_degrees": outer_degrees,
                "buffer_inner_km_approx": approx_degrees_to_km(inner_degrees),
                "buffer_outer_km_approx": approx_degrees_to_km(outer_degrees),
            }
        ).setGeometry(geometry)

    return grid_points.map(buffer_feature)


def prediction_hansen_image_asset_id(asset_root: str, year: int, ring_suffix: str) -> str:
    """Return the image-asset id for one precomputed annual Hansen prediction raster."""
    scale_label = format_degree_label(ring_scale_degrees(ring_suffix))
    return f"{asset_root.rstrip('/')}/hansen_{int(year)}_{ring_suffix}_{scale_label}deg"


def format_degree_label(value: float) -> str:
    """Format a degree value so it is safe in an Earth Engine asset name."""
    return f"{float(value):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def aggregate_hansen_band_for_degrees(
    image: ee.Image,
    name: str,
    source_projection: ee.Projection,
    clip_region: ee.Geometry | None = None,
    fill_masked_land_with_zero: bool = False,
) -> ee.Image:
    """Aggregate native Hansen pixels by mean for later degree-grid output."""
    if clip_region is not None:
        if fill_masked_land_with_zero:
            image = zero_fill_hansen_land_pixels(image, clip_region)
        else:
            image = image.updateMask(land_mask_image(clip_region)).clip(clip_region)

    return (
        image.setDefaultProjection(source_projection)
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            bestEffort=True,
            maxPixels=_BASE.HANSEN_REDUCE_MAX_PIXELS,
        )
        .rename(name)
        .toFloat()
    )


def annual_forest_cover_image_degrees(
    year: int | ee.Number,
    clip_region: ee.Geometry | None = None,
) -> ee.Image:
    """Annual Hansen forest-cover proportion prepared for degree-grid output."""
    components = _BASE.hansen_base_components(ee.Number(year))
    hansen_projection = ee.Projection(components["hansen_projection"])
    tree_cover_2000 = ee.Image(components["tree_cover_2000"])
    loss_year = ee.Image(components["loss_year"])
    years_since_2000 = ee.Number(components["years_since_2000"])

    lost_through_year = loss_year.gt(0).And(loss_year.lte(years_since_2000))
    forest_cover = tree_cover_2000.where(lost_through_year, 0).rename("forest_cover_prop").toFloat()
    return aggregate_hansen_band_for_degrees(
        forest_cover,
        "forest_cover_prop",
        hansen_projection,
        clip_region=clip_region,
        fill_masked_land_with_zero=True,
    )


def annual_forest_loss_images_degrees(
    year: int | ee.Number,
    clip_region: ee.Geometry | None = None,
) -> ee.Image:
    """Annual Hansen forest-loss indicators prepared for degree-grid output."""
    year = ee.Number(year)
    components = _BASE.hansen_base_components(year)
    hansen_projection = ee.Projection(components["hansen_projection"])
    loss_year = ee.Image(components["loss_year"])
    years_since_2000 = ee.Number(components["years_since_2000"])

    flsy = loss_year.eq(years_since_2000).unmask(0).rename("flsy_prop").toFloat()
    fl1yp = ee.Image(
        ee.Algorithms.If(
            year.lt(2002),
            _BASE.masked_band("fl1yp_prop"),
            loss_year.eq(years_since_2000.subtract(1)).unmask(0).rename("fl1yp_prop").toFloat(),
        )
    )
    fl2yp = ee.Image(
        ee.Algorithms.If(
            year.lt(2003),
            _BASE.masked_band("fl2yp_prop"),
            loss_year.eq(years_since_2000.subtract(2)).unmask(0).rename("fl2yp_prop").toFloat(),
        )
    )

    flsy = aggregate_hansen_band_for_degrees(flsy, "flsy_prop", hansen_projection, clip_region=clip_region)
    fl1yp = aggregate_hansen_band_for_degrees(fl1yp, "fl1yp_prop", hansen_projection, clip_region=clip_region)
    fl2yp = aggregate_hansen_band_for_degrees(fl2yp, "fl2yp_prop", hansen_projection, clip_region=clip_region)
    return ee.Image.cat([flsy, fl1yp, fl2yp]).toFloat()


def annual_fragmentation_image_degrees(
    year: int | ee.Number,
    clip_region: ee.Geometry | None = None,
) -> ee.Image:
    """Annual edge proportion prepared for degree-grid output."""
    components = _BASE.hansen_base_components(ee.Number(year))
    hansen_projection = ee.Projection(components["hansen_projection"])
    tree_cover_2000 = ee.Image(components["tree_cover_2000"])
    loss_year = ee.Image(components["loss_year"])
    years_since_2000 = ee.Number(components["years_since_2000"])
    lost_through_year = loss_year.gt(0).And(loss_year.lte(years_since_2000))
    forest_cover = tree_cover_2000.where(lost_through_year, 0).rename("forest_cover_prop").toFloat()
    dense_forest = forest_cover.gte(_BASE.DENSE_FOREST_THRESHOLD).unmask(0)
    neighbor_min = dense_forest.focal_min(radius=1, units="pixels")
    edge_prop = dense_forest.eq(1).And(neighbor_min.eq(0)).rename("frag_edge_prop").toFloat()
    return aggregate_hansen_band_for_degrees(
        edge_prop,
        "frag_edge_prop",
        hansen_projection,
        clip_region=clip_region,
        fill_masked_land_with_zero=True,
    )


def prediction_hansen_image(
    year: int | ee.Number,
    ring_suffix: str,
    study_region: ee.Geometry | None = None,
) -> ee.Image:
    """Build one annual multi-band Hansen image for degree-grid prediction."""
    image = ee.Image.cat(
        [
            annual_forest_cover_image_degrees(year, clip_region=study_region),
            annual_forest_loss_images_degrees(year, clip_region=study_region),
            annual_fragmentation_image_degrees(year, clip_region=study_region),
        ]
    )
    return image.select(HANSEN_PREDICTION_BANDS).toFloat()


def prediction_buffer_asset_id(asset_root: str, ring_suffix: str, batch_id: int | None = None) -> str:
    """Return the table-asset id for one prediction-grid buffer scale/batch."""
    asset_name = f"prediction_grid_buffers_{ring_suffix}"
    if batch_id is not None:
        asset_name = f"{asset_name}_batch_{int(batch_id):03d}"
    return f"{asset_root.rstrip('/')}/{asset_name}"


def prediction_buffer_asset_ids(
    asset_root: str,
    ring_suffixes: Sequence[str] | None = None,
    batch_id: int | None = None,
) -> dict[str, str]:
    """Return prediction buffer suffix -> table-asset id mappings."""
    return {
        suffix: prediction_buffer_asset_id(asset_root, suffix, batch_id=batch_id)
        for _, _, suffix in selected_scaled_rings(ring_suffixes)
    }


def merge_ring_feature_collections(
    ring_collections: Sequence[tuple[ee.FeatureCollection, Sequence[str]]],
    join_key: str = "id",
) -> ee.FeatureCollection:
    """Merge FeatureCollections into one wide table using a configurable key."""
    if not ring_collections:
        raise ValueError("At least one ring FeatureCollection is required.")

    combined = ring_collections[0][0]
    for collection, property_names in ring_collections[1:]:
        combined = _BASE.join_feature_collection_properties(
            combined,
            collection,
            property_names,
            join_key=join_key,
        )

    return combined


def reduce_image_over_buffer_features(
    image: ee.Image,
    features: ee.FeatureCollection,
    band_names: Sequence[str],
    ring_suffix: str,
    scale_m: int | float | None = None,
) -> ee.FeatureCollection:
    """Extract mean image values over precomputed buffer polygons with reduceRegions."""
    scale_m = scale_m or ring_scale_degrees(ring_suffix)
    band_names = list(band_names)
    output_names = [f"{name}_{ring_suffix}" for name in band_names]
    renamed = image.select(band_names).rename(output_names)

    reduction_kwargs = {
        "collection": features,
        "reducer": ee.Reducer.mean(),
        "tileScale": 4,
        "maxPixelsPerRegion": int(1e13),
    }
    if float(scale_m) < 1:
        reduction_kwargs["crs"] = LAT_LONG_CRS
        reduction_kwargs["crsTransform"] = degree_grid_transform(float(scale_m))
    else:
        reduction_kwargs["scale"] = int(scale_m)

    reduced = renamed.reduceRegions(**reduction_kwargs)
    if len(output_names) == 1:
        output_name = output_names[0]
        reduced = reduced.map(
            lambda feature: ee.Feature(feature).set(output_name, ee.Feature(feature).get("mean"))
        )
    return reduced.map(lambda feature: ee.Feature(feature).setGeometry(None))


def export_image_to_asset(
    image: ee.Image,
    description: str,
    asset_id: str,
    region: ee.Geometry,
    scale_m: int | float | None = None,
    scale_degrees: float | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    max_pixels: int = int(1e13),
) -> dict:
    """Start an Earth Engine image-asset export and return manifest metadata."""
    description = description[:100]
    row = {
        "description": description,
        "asset_id": asset_id,
        "scale_m": int(scale_m) if scale_m is not None and float(scale_m) >= 1 else None,
        "scale_degrees": float(scale_degrees) if scale_degrees is not None else (
            float(scale_m) if scale_m is not None and float(scale_m) < 1 else None
        ),
        "task_id": None,
        "state": "DRY_RUN" if dry_run else None,
    }

    if dry_run:
        print(f"DRY RUN image asset export: {description} -> {asset_id}")
        return row

    if not overwrite:
        try:
            ee.data.getAsset(asset_id)
            row["state"] = "EXISTS"
            print(f"Skipping existing image asset: {asset_id}")
            return row
        except Exception:
            pass

    export_kwargs = {
        "image": image,
        "description": description,
        "assetId": asset_id,
        "region": region,
        "maxPixels": max_pixels,
        "overwrite": overwrite,
    }
    if scale_degrees is not None:
        export_kwargs["crs"] = LAT_LONG_CRS
        export_kwargs["crsTransform"] = degree_grid_transform(float(scale_degrees))
    elif scale_m is not None and float(scale_m) < 1:
        export_kwargs["crs"] = LAT_LONG_CRS
        export_kwargs["crsTransform"] = degree_grid_transform(float(scale_m))
    else:
        export_kwargs["scale"] = int(scale_m or _BASE.EXTRACTION_SCALE_M)

    task = ee.batch.Export.image.toAsset(**export_kwargs)
    task.start()
    status = task.status()
    row["task_id"] = status.get("id")
    row["state"] = status.get("state")
    print(f"Started image asset export: {description} | state={row['state']} | task_id={row['task_id']}")
    return row
