"""Google Earth Engine covariate helpers for the event ENM pipeline.

Python orchestrates Earth Engine. The heavy raster operations run in GEE.
Local rasters must be uploaded to Earth Engine first, then listed in
config/external_rasters.csv as asset IDs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import ee
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CODE_DIR / "data"
CONFIG_DIR = CODE_DIR / "config"
OUTPUT_DIR = CODE_DIR / "outputs"

STUDY_AREA_FILE = CONFIG_DIR / "africacountries_nolakes.shp"
EXTERNAL_RASTERS_FILE = CONFIG_DIR / "external_rasters.csv"
PREDICTOR_LIST_FILE = CONFIG_DIR / "predictor_list.csv"

# Current project default: mainland Africa clipped to the equatorial band.
# Pass bbox=None to use the full provided shapefile/polygon instead.
DEFAULT_STUDY_AREA_BBOX = (-15.5, -10.0, 51.0, 10.0)

EXPORT_FOLDER = "Event_ENM_Exports"
BASELINE_YEARS = list(range(2000, 2025))
PREDICTION_YEARS = list(range(2020, 2026))

HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
LANDSCAN_COLLECTION = "projects/sat-io/open-datasets/ORNL/LANDSCAN_GLOBAL"
LANDSCAN_LATEST_YEAR = 2023
ERA5_LAND_MONTHLY_AGGR_COLLECTION = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
ERA5_LAND_DAILY_AGGR_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
MODIS_NDVI_COLLECTION = "MODIS/061/MOD13A2"
ELEVATION_ASSET = "USGS/SRTMGL1_003"

# The active training/prediction workflows use approximate lon/lat degree
# buffers near the equator instead of projected-meter buffers.
LAT_LONG_CRS = "EPSG:4326"
APPROX_KM_PER_DEGREE = 111.32
ANALYSIS_CRS = LAT_LONG_CRS
EXTRACTION_SCALE_M = 1000
PREDICTION_GRID_SCALE_M = 10000

# Native or intended source scales used for reductions. Hansen-derived
# variables are intentionally coarsened before buffer extraction because the
# native 30 m product is too expensive for repeated 100 km summaries.
ERA5_NATIVE_SCALE_M = 11132
ERA5_DAILY_NATIVE_SCALE_M = 11132
MODIS_NDVI_NATIVE_SCALE_M = 1000
SRTM_EXTRACTION_SCALE_M = 5000
LANDSCAN_NATIVE_SCALE_M = 1000
HANSEN_BUFFER_SCALE_M = 1000
HANSEN_REDUCE_MAX_PIXELS = 65535
DENSE_FOREST_THRESHOLD = 0.40
NO_DATA_VALUE = -9999
MAX_CONTEXT_BUFFER_M = 100000
EXTRACTION_LAT_BUFFER_DEGREES = 1.0
BUFFER_GEOMETRY_ERROR_M = 100

# Approximate ring radii in EPSG:4326 degrees. The 50_100km label is held
# stable while the active test geometry uses a 75 km outer radius.
SCALED_RINGS = [
    (0.000, 0.090, "0_10km"),
    (0.090, 0.225, "10_25km"),
    (0.225, 0.449, "25_50km"),
    (0.449, 0.674, "50_100km"),
]
NONSCALED_RINGS = [(0.000, 0.090, "0_10km")]

# Degree equivalents of the simplification tolerances used in the prediction
# workflow: ~200 m, 500 m, 3000 m, and 8000 m at the equator.
BUFFER_GEOMETRY_ERROR_BY_DISTANCE_DEG = {
    0.090: 0.0018,
    0.225: 0.0045,
    0.449: 0.0270,
    0.674: 0.0719,
}

BASE_SCALED_BANDS = [
    "forest_cover_prop",
    "flsy_prop",
    "fl1yp_prop",
    "fl2yp_prop",
    "frag_edge_prop",
    "pop_density",
]

BASE_NONSCALED_BANDS = [
    "precip_mm",
    "precip_anom_mm",
    "precip_z",
    "temp_c",
    "temp_anom_c",
    "temp_z",
    "pet_mm",
    "ndvi",
    "ndvi_anom",
    "ndvi_z",
    "elevation_m",
]

TRAINING_BASE_COLUMNS = [
    "id",
    "year",
    "latitude",
    "longitude",
    "outcome",
    "type",
    "country",
]

SCALED_COVARIATE_FAMILIES = {
    "forest_cover": ["forest_cover_prop"],
    "forest_loss": ["flsy_prop", "fl1yp_prop", "fl2yp_prop"],
    "fragmentation": ["frag_edge_prop"],
    "population": ["pop_density"],
}

RING_EXTRACTION_SCALES_DEG = {
    "0_10km": 0.0018,
    "10_25km": 0.0045,
    "25_50km": 0.0180,
    "50_100km": 0.0270,
}

_BASELINE_CACHE: dict[str, tuple[ee.Image, ee.Image]] = {}


@dataclass(frozen=True)
class ExternalRasterSpec:
    covariate_name: str
    gee_asset_id: str
    buffer_mode: str = "nonscaled"
    reducer: str = "mean"
    scale_m: int = EXTRACTION_SCALE_M
    enabled: bool = False


@dataclass(frozen=True)
class CovariateImageGroup:
    name: str
    image: ee.Image
    band_names: list[str]
    scale_m: int


def initialize_earth_engine(project: str | None = None) -> None:
    """Initialize Earth Engine, authenticating interactively if needed."""
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()


def normalize_study_area_bbox(
    bbox: Sequence[float] | dict[str, float] | None,
) -> tuple[float, float, float, float] | None:
    """Return bbox as xmin/ymin/xmax/ymax, or None to use the full input polygon."""
    if bbox is None:
        return None

    if isinstance(bbox, dict):
        values = [bbox[key] for key in ("xmin", "ymin", "xmax", "ymax")]
    else:
        values = list(bbox)

    if len(values) != 4:
        raise ValueError("Study-area bbox must contain xmin, ymin, xmax, ymax, or be None.")

    xmin, ymin, xmax, ymax = [float(value) for value in values]
    if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
        raise ValueError("Study-area bbox values must be finite numbers.")
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Study-area bbox must have xmin < xmax and ymin < ymax.")

    return xmin, ymin, xmax, ymax


def expand_study_area_bbox(
    bbox: Sequence[float] | dict[str, float] | None,
    lat_buffer_degrees: float = 0.0,
    lon_buffer_degrees: float = 0.0,
) -> tuple[float, float, float, float] | None:
    """Expand a configured bbox for extraction context; None remains full polygon."""
    normalized = normalize_study_area_bbox(bbox)
    if normalized is None:
        return None

    xmin, ymin, xmax, ymax = normalized
    lat_buffer = float(lat_buffer_degrees)
    lon_buffer = float(lon_buffer_degrees)
    return (
        max(-180.0, xmin - lon_buffer),
        max(-90.0, ymin - lat_buffer),
        min(180.0, xmax + lon_buffer),
        min(90.0, ymax + lat_buffer),
    )


def bbox_label(bbox: Sequence[float] | dict[str, float] | None) -> str:
    """Human-readable study-area bbox label for messages/errors."""
    normalized = normalize_study_area_bbox(bbox)
    if normalized is None:
        return "full provided polygon extent"

    xmin, ymin, xmax, ymax = normalized
    return f"xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}"


def load_study_region(
    path: Path = STUDY_AREA_FILE,
    bbox: Sequence[float] | dict[str, float] | None = DEFAULT_STUDY_AREA_BBOX,
    lat_buffer_degrees: float = 0.0,
    lon_buffer_degrees: float = 0.0,
) -> ee.Geometry:
    """Load the active study-area polygon as an ee.Geometry.

    The full provided shapefile/GeoJSON is used when bbox is None. When bbox is
    supplied, the input polygon is clipped to that lon/lat box. Optional
    latitude/longitude buffers expand the bbox for raster masking/extraction
    context around edge locations.
    """
    path = Path(path)
    expanded_bbox = expand_study_area_bbox(
        bbox,
        lat_buffer_degrees=lat_buffer_degrees,
        lon_buffer_degrees=lon_buffer_degrees,
    )
    if path.suffix.lower() == ".shp":
        import geopandas as gpd
        from shapely.geometry import box, mapping

        gdf = gpd.read_file(path).to_crs("EPSG:4326")
        if expanded_bbox is not None:
            band = gpd.GeoDataFrame(
                geometry=[box(*expanded_bbox)],
                crs="EPSG:4326",
            )
            clipped = gpd.clip(gdf, band)
        else:
            clipped = gdf
        clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty]

        if clipped.empty:
            raise ValueError(f"No study-area geometry remains after applying bbox {bbox_label(expanded_bbox)} to {path}.")

        if hasattr(clipped.geometry, "union_all"):
            unioned = clipped.geometry.union_all()
        else:
            unioned = clipped.geometry.unary_union
        simplified = unioned.simplify(0.005, preserve_topology=True)
        return ee.Geometry(mapping(simplified), None, False)

    with open(path, "r", encoding="utf-8") as handle:
        geojson = json.load(handle)

    if geojson["type"] == "FeatureCollection":
        geometry = geojson["features"][0]["geometry"]
    elif geojson["type"] == "Feature":
        geometry = geojson["geometry"]
    else:
        geometry = geojson

    if expanded_bbox is not None:
        from shapely.geometry import box, mapping, shape

        clipped_geometry = shape(geometry).intersection(box(*expanded_bbox))
        if clipped_geometry.is_empty:
            raise ValueError(f"No study-area geometry remains after applying bbox {bbox_label(expanded_bbox)} to {path}.")
        geometry = mapping(clipped_geometry.simplify(0.005, preserve_topology=True))

    return ee.Geometry(geometry, None, False)


def read_external_raster_specs(path: Path = EXTERNAL_RASTERS_FILE) -> list[ExternalRasterSpec]:
    """Read enabled external raster assets from the config CSV."""
    if not path.exists():
        return []

    df = pd.read_csv(path).fillna("")
    specs: list[ExternalRasterSpec] = []

    for row in df.to_dict("records"):
        enabled = str(row.get("enabled", "")).strip().lower() == "true"
        asset_id = str(row.get("gee_asset_id", "")).strip()
        if not enabled or not asset_id:
            continue

        specs.append(
            ExternalRasterSpec(
                covariate_name=str(row["covariate_name"]).strip(),
                gee_asset_id=asset_id,
                buffer_mode=str(row.get("buffer_mode", "nonscaled")).strip().lower(),
                reducer=str(row.get("reducer", "mean")).strip().lower(),
                scale_m=int(row.get("scale_m", EXTRACTION_SCALE_M) or EXTRACTION_SCALE_M),
                enabled=enabled,
            )
        )

    return specs


def expected_predictor_names(
    external_specs: Sequence[ExternalRasterSpec] | None = None,
    covariate_group: str = "all",
    scaled_ring_suffixes: Sequence[str] | None = None,
) -> list[str]:
    """Return predictor names in the order expected by the R model."""
    if covariate_group not in {"all", "scaled", "nonscaled"}:
        raise ValueError("covariate_group must be 'all', 'scaled', or 'nonscaled'.")

    names: list[str] = []
    scaled_rings = selected_scaled_rings(scaled_ring_suffixes)

    if covariate_group in {"all", "scaled"}:
        for band in BASE_SCALED_BANDS:
            for _, _, suffix in scaled_rings:
                names.append(f"{band}_{suffix}")

    if covariate_group in {"all", "nonscaled"}:
        for band in BASE_NONSCALED_BANDS:
            names.append(f"{band}_0_10km")

    for spec in external_specs or []:
        if spec.buffer_mode == "scaled" and covariate_group in {"all", "scaled"}:
            for _, _, suffix in scaled_rings:
                names.append(f"{spec.covariate_name}_{suffix}")
        elif spec.buffer_mode != "scaled" and covariate_group in {"all", "nonscaled"}:
            names.append(f"{spec.covariate_name}_0_10km")

    return names


def selected_scaled_rings(
    scaled_ring_suffixes: Sequence[str] | None = None,
) -> list[tuple[float, float, str]]:
    """Return scaled buffer definitions selected by suffix, preserving order."""
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
    """Return geometry simplification tolerance in EPSG:4326 degrees."""
    distances = sorted(BUFFER_GEOMETRY_ERROR_BY_DISTANCE_DEG)
    for distance in distances:
        if float(outer_degrees) <= distance:
            return BUFFER_GEOMETRY_ERROR_BY_DISTANCE_DEG[distance]
    return BUFFER_GEOMETRY_ERROR_BY_DISTANCE_DEG[distances[-1]]


def ring_scale_degrees(suffix: str) -> float:
    """Return the Hansen extraction scale in EPSG:4326 degrees."""
    return RING_EXTRACTION_SCALES_DEG.get(suffix, 0.0090)


def ring_scale_m(suffix: str) -> int:
    """Return the approximate Hansen extraction scale in meters for display."""
    return approx_degrees_to_m(ring_scale_degrees(suffix))


def buffer_geometry_error_m(outer_m: int | float) -> int:
    """Return the approximate simplification tolerance in meters for display."""
    return approx_degrees_to_m(buffer_geometry_error_degrees(float(outer_m)))


def training_buffer_asset_id(asset_root: str, ring_suffix: str) -> str:
    """Return the table-asset id for one training buffer scale."""
    return f"{asset_root.rstrip('/')}/training_buffers_{ring_suffix}"


def training_buffer_asset_ids(
    asset_root: str,
    ring_suffixes: Sequence[str] | None = None,
) -> dict[str, str]:
    """Return buffer suffix -> table-asset id mappings."""
    return {
        suffix: training_buffer_asset_id(asset_root, suffix)
        for _, _, suffix in selected_scaled_rings(ring_suffixes)
    }


def scaled_family_band_names(family: str) -> list[str]:
    """Return source band names for one scaled covariate family."""
    if family not in SCALED_COVARIATE_FAMILIES:
        valid = sorted(SCALED_COVARIATE_FAMILIES)
        raise ValueError(f"Unknown scaled covariate family: {family}. Valid families: {valid}")

    return list(SCALED_COVARIATE_FAMILIES[family])


def scaled_family_scale_m(family: str, ring_suffix: str | None = None) -> int | float:
    """Return the reduction scale for one scaled covariate family."""
    if family in {"forest_cover", "forest_loss", "fragmentation"}:
        if ring_suffix is not None:
            return ring_scale_degrees(ring_suffix)
        return HANSEN_BUFFER_SCALE_M
    if family == "population":
        return LANDSCAN_NATIVE_SCALE_M

    return EXTRACTION_SCALE_M


def export_group_predictor_names(
    export_group: str,
    external_specs: Sequence[ExternalRasterSpec] | None = None,
    scaled_ring_suffixes: Sequence[str] | None = None,
) -> list[str]:
    """Return output predictor names for one asset-based export group."""
    if export_group == "nonscaled":
        return expected_predictor_names(external_specs, covariate_group="nonscaled")

    band_names = scaled_family_band_names(export_group)
    return [
        f"{band}_{suffix}"
        for band in band_names
        for _, _, suffix in selected_scaled_rings(scaled_ring_suffixes)
    ]


def training_export_description(
    year: int,
    export_group: str,
    ring_suffixes: Sequence[str] | None = None,
) -> str:
    """Return a stable Drive export description for one training export group."""
    if export_group == "nonscaled":
        return f"dataset2_{int(year)}_nonscaled"

    selected = [suffix for _, _, suffix in selected_scaled_rings(ring_suffixes)]
    all_rings = [suffix for _, _, suffix in selected_scaled_rings(None)]
    if selected == all_rings:
        return f"dataset2_{int(year)}_{export_group}"

    return f"dataset2_{int(year)}_{export_group}_{'_'.join(selected)}"


def training_export_description_all_years(
    export_group: str,
    ring_suffixes: Sequence[str] | None = None,
) -> str:
    """Return a stable Drive export description for one all-years export."""
    if export_group == "nonscaled":
        return "dataset2_all_years_nonscaled"

    selected = [suffix for _, _, suffix in selected_scaled_rings(ring_suffixes)]
    all_rings = [suffix for _, _, suffix in selected_scaled_rings(None)]
    if selected == all_rings:
        return f"dataset2_all_years_{export_group}"

    return f"dataset2_all_years_{export_group}_{'_'.join(selected)}"


def clean_feature_property(value):
    """Convert pandas/numpy values to JSON-safe Earth Engine properties."""
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def dataframe_to_feature_collection(df: pd.DataFrame) -> ee.FeatureCollection:
    """Convert dataset1-style rows into an Earth Engine FeatureCollection."""
    required = {"id", "year", "latitude", "longitude", "outcome"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")

    features = []

    for row in df.to_dict("records"):
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        props = {}

        for key, value in row.items():
            if key in {"longitude", "latitude"}:
                continue
            clean_value = clean_feature_property(value)
            if clean_value is not None:
                props[key] = clean_value

        props["longitude"] = lon
        props["latitude"] = lat
        props["year"] = int(props["year"])
        props["outcome"] = int(props["outcome"])

        features.append(ee.Feature(ee.Geometry.Point([lon, lat]), props))

    return ee.FeatureCollection(features)


def masked_band(name: str) -> ee.Image:
    """Create a fully masked numeric band so reductions return null."""
    zero = ee.Image.constant(0)
    return zero.updateMask(zero).rename(name).toFloat()


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


def coarsen_hansen_band(
    image: ee.Image,
    name: str,
    source_projection: ee.Projection,
    scale_m: int = HANSEN_BUFFER_SCALE_M,
    clip_region: ee.Geometry | None = None,
    fill_masked_land_with_zero: bool = False,
) -> ee.Image:
    """Aggregate native Hansen pixels to the buffer extraction scale.

    The resulting bands remain proportions: for binary loss/edge bands this is
    the fraction of native pixels in each coarse cell; for tree cover it is mean
    proportional canopy cover.
    """
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
            maxPixels=HANSEN_REDUCE_MAX_PIXELS,
        )
        .reproject(source_projection.atScale(scale_m))
        .rename(name)
        .toFloat()
    )


def aggregate_hansen_band_for_degrees(
    image: ee.Image,
    name: str,
    source_projection: ee.Projection,
    clip_region: ee.Geometry | None = None,
    fill_masked_land_with_zero: bool = False,
) -> ee.Image:
    """Aggregate native Hansen pixels by mean for later EPSG:4326 degree output."""
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
            maxPixels=HANSEN_REDUCE_MAX_PIXELS,
        )
        .rename(name)
        .toFloat()
    )


def hansen_base_components(year: ee.Number) -> dict[str, ee.Image | ee.Number | ee.Projection]:
    year = ee.Number(year)
    hansen = ee.Image(HANSEN_ASSET)
    hansen_projection = hansen.select("treecover2000").projection()
    tree_cover_2000 = hansen.select("treecover2000").divide(100)
    loss_year = hansen.select("lossyear")
    years_since_2000 = year.subtract(2000)
    return {
        "hansen_projection": hansen_projection,
        "tree_cover_2000": tree_cover_2000,
        "loss_year": loss_year,
        "years_since_2000": years_since_2000,
    }


def annual_forest_cover_image(
    year: ee.Number,
    scale_m: int = HANSEN_BUFFER_SCALE_M,
    clip_region: ee.Geometry | None = None,
) -> ee.Image:
    """Annual Hansen forest-cover proportion aggregated to analysis scale."""
    components = hansen_base_components(year)
    tree_cover_2000 = ee.Image(components["tree_cover_2000"])
    loss_year = ee.Image(components["loss_year"])
    years_since_2000 = ee.Number(components["years_since_2000"])
    hansen_projection = ee.Projection(components["hansen_projection"])

    lost_through_year = loss_year.gt(0).And(loss_year.lte(years_since_2000))
    forest_cover = tree_cover_2000.where(lost_through_year, 0).rename("forest_cover_prop")
    return coarsen_hansen_band(
        forest_cover,
        "forest_cover_prop",
        hansen_projection,
        scale_m,
        clip_region=clip_region,
        fill_masked_land_with_zero=True,
    )


def annual_forest_loss_images(
    year: ee.Number,
    scale_m: int = HANSEN_BUFFER_SCALE_M,
    clip_region: ee.Geometry | None = None,
) -> ee.Image:
    """Annual Hansen forest-loss indicators aggregated to analysis scale."""
    components = hansen_base_components(year)
    loss_year = ee.Image(components["loss_year"])
    years_since_2000 = ee.Number(components["years_since_2000"])
    hansen_projection = ee.Projection(components["hansen_projection"])

    flsy = loss_year.eq(years_since_2000).unmask(0).rename("flsy_prop").toFloat()
    # Hansen lossyear starts in 2001, so each lag is missing only before
    # its prior-year target exists.
    fl1yp = ee.Image(
        ee.Algorithms.If(
            ee.Number(year).lt(2002),
            masked_band("fl1yp_prop"),
            loss_year.eq(years_since_2000.subtract(1)).unmask(0).rename("fl1yp_prop").toFloat(),
        )
    )
    fl2yp = ee.Image(
        ee.Algorithms.If(
            ee.Number(year).lt(2003),
            masked_band("fl2yp_prop"),
            loss_year.eq(years_since_2000.subtract(2)).unmask(0).rename("fl2yp_prop").toFloat(),
        )
    )

    flsy = coarsen_hansen_band(flsy, "flsy_prop", hansen_projection, scale_m, clip_region=clip_region)
    fl1yp = coarsen_hansen_band(fl1yp, "fl1yp_prop", hansen_projection, scale_m, clip_region=clip_region)
    fl2yp = coarsen_hansen_band(fl2yp, "fl2yp_prop", hansen_projection, scale_m, clip_region=clip_region)
    return ee.Image.cat([flsy, fl1yp, fl2yp]).toFloat()


def annual_fragmentation_image(
    year: ee.Number,
    scale_m: int = HANSEN_BUFFER_SCALE_M,
    clip_region: ee.Geometry | None = None,
) -> ee.Image:
    """Annual edge proportion derived from coarsened Hansen forest cover."""
    forest_cover = annual_forest_cover_image(year, scale_m, clip_region=clip_region)

    dense_forest_coarse = forest_cover.gte(DENSE_FOREST_THRESHOLD).unmask(0)
    neighbor_min = dense_forest_coarse.focal_min(radius=1, units="pixels")
    edge_prop = (
        dense_forest_coarse.eq(1)
        .And(neighbor_min.eq(0))
        .rename("frag_edge_prop")
        .toFloat()
    )
    if clip_region is not None:
        edge_prop = mask_image_to_region(edge_prop, clip_region).clip(clip_region)

    return edge_prop.toFloat()


def hansen_forest_images(
    year: ee.Number,
    scale_m: int = HANSEN_BUFFER_SCALE_M,
    clip_region: ee.Geometry | None = None,
) -> ee.Image:
    """Annual forest cover, forest-loss lags, and edge fragmentation."""
    forest_cover = annual_forest_cover_image(year, scale_m, clip_region=clip_region)
    loss_images = annual_forest_loss_images(year, scale_m, clip_region=clip_region)
    edge_prop = annual_fragmentation_image(year, scale_m, clip_region=clip_region)
    return ee.Image.cat([forest_cover, loss_images, edge_prop]).toFloat()


def annual_forest_cover_image_degrees(
    year: ee.Number,
    clip_region: ee.Geometry | None = None,
) -> ee.Image:
    """Annual Hansen forest-cover proportion prepared for degree-grid output."""
    components = hansen_base_components(year)
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
    year: ee.Number,
    clip_region: ee.Geometry | None = None,
) -> ee.Image:
    """Annual Hansen forest-loss indicators prepared for degree-grid output."""
    year = ee.Number(year)
    components = hansen_base_components(year)
    hansen_projection = ee.Projection(components["hansen_projection"])
    loss_year = ee.Image(components["loss_year"])
    years_since_2000 = ee.Number(components["years_since_2000"])

    flsy = loss_year.eq(years_since_2000).unmask(0).rename("flsy_prop").toFloat()
    fl1yp = ee.Image(
        ee.Algorithms.If(
            year.lt(2002),
            masked_band("fl1yp_prop"),
            loss_year.eq(years_since_2000.subtract(1)).unmask(0).rename("fl1yp_prop").toFloat(),
        )
    )
    fl2yp = ee.Image(
        ee.Algorithms.If(
            year.lt(2003),
            masked_band("fl2yp_prop"),
            loss_year.eq(years_since_2000.subtract(2)).unmask(0).rename("fl2yp_prop").toFloat(),
        )
    )

    flsy = aggregate_hansen_band_for_degrees(flsy, "flsy_prop", hansen_projection, clip_region=clip_region)
    fl1yp = aggregate_hansen_band_for_degrees(fl1yp, "fl1yp_prop", hansen_projection, clip_region=clip_region)
    fl2yp = aggregate_hansen_band_for_degrees(fl2yp, "fl2yp_prop", hansen_projection, clip_region=clip_region)
    return ee.Image.cat([flsy, fl1yp, fl2yp]).toFloat()


def annual_fragmentation_image_degrees(
    year: ee.Number,
    clip_region: ee.Geometry | None = None,
) -> ee.Image:
    """Annual edge proportion prepared for degree-grid output."""
    components = hansen_base_components(year)
    hansen_projection = ee.Projection(components["hansen_projection"])
    tree_cover_2000 = ee.Image(components["tree_cover_2000"])
    loss_year = ee.Image(components["loss_year"])
    years_since_2000 = ee.Number(components["years_since_2000"])
    lost_through_year = loss_year.gt(0).And(loss_year.lte(years_since_2000))
    forest_cover = tree_cover_2000.where(lost_through_year, 0).rename("forest_cover_prop").toFloat()
    dense_forest = forest_cover.gte(DENSE_FOREST_THRESHOLD).unmask(0)
    neighbor_min = dense_forest.focal_min(radius=1, units="pixels")
    edge_prop = dense_forest.eq(1).And(neighbor_min.eq(0)).rename("frag_edge_prop").toFloat()
    return aggregate_hansen_band_for_degrees(
        edge_prop,
        "frag_edge_prop",
        hansen_projection,
        clip_region=clip_region,
        fill_masked_land_with_zero=True,
    )


def hansen_forest_images_degrees(
    year: ee.Number,
    clip_region: ee.Geometry | None = None,
) -> ee.Image:
    """Annual forest cover, forest loss, and fragmentation for degree-grid output."""
    return ee.Image.cat(
        [
            annual_forest_cover_image_degrees(year, clip_region=clip_region),
            annual_forest_loss_images_degrees(year, clip_region=clip_region),
            annual_fragmentation_image_degrees(year, clip_region=clip_region),
        ]
    ).toFloat()


def landscan_population_density(year: ee.Number) -> ee.Image:
    """Population density, using latest LandScan year when requested year is newer."""
    pop_year = ee.Number(year).min(LANDSCAN_LATEST_YEAR)
    pop_image = (
        ee.ImageCollection(LANDSCAN_COLLECTION)
        .filter(ee.Filter.calendarRange(pop_year, pop_year, "year"))
        .first()
    )
    people_per_pixel = ee.Image(pop_image).select(0)
    pixel_area_km2 = ee.Image.pixelArea().divide(1e6)
    return people_per_pixel.divide(pixel_area_km2).rename("pop_density").toFloat()


def scaled_covariate_family_image(
    year: ee.Number,
    family: str,
    scale_m: int | float | None = None,
) -> ee.Image:
    """Return the source image for one scaled covariate family."""
    use_degree_hansen = scale_m is not None and float(scale_m) < 1
    if family == "forest_cover":
        if use_degree_hansen:
            return annual_forest_cover_image_degrees(year)
        return annual_forest_cover_image(year, scale_m or HANSEN_BUFFER_SCALE_M)
    if family == "forest_loss":
        if use_degree_hansen:
            return annual_forest_loss_images_degrees(year)
        return annual_forest_loss_images(year, scale_m or HANSEN_BUFFER_SCALE_M)
    if family == "fragmentation":
        if use_degree_hansen:
            return annual_fragmentation_image_degrees(year)
        return annual_fragmentation_image(year, scale_m or HANSEN_BUFFER_SCALE_M)
    if family == "population":
        return landscan_population_density(year)

    valid = sorted(SCALED_COVARIATE_FAMILIES)
    raise ValueError(f"Unknown scaled covariate family: {family}. Valid families: {valid}")


def annual_era5_land_monthly_collection(year: ee.Number) -> ee.ImageCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    return ee.ImageCollection(ERA5_LAND_MONTHLY_AGGR_COLLECTION).filterDate(start, end)


def annual_era5_land_daily_collection(year: ee.Number) -> ee.ImageCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    return ee.ImageCollection(ERA5_LAND_DAILY_AGGR_COLLECTION).filterDate(start, end)


def annual_precip_image(year: ee.Number) -> ee.Image:
    # ERA5-Land monthly aggregate precipitation is stored as monthly sums in meters.
    collection = annual_era5_land_monthly_collection(year).select("total_precipitation_sum")
    image = ee.Image(
        ee.Algorithms.If(
            collection.size().gt(0),
            collection.sum().multiply(1000).rename("precip_mm"),
            masked_band("precip_mm"),
        )
    )
    return image.toFloat()


def annual_temp_image(year: ee.Number) -> ee.Image:
    # ERA5-Land 2m temperature is stored in Kelvin; use annual mean Celsius.
    collection = annual_era5_land_monthly_collection(year).select("temperature_2m")
    image = ee.Image(
        ee.Algorithms.If(
            collection.size().gt(0),
            collection.mean().subtract(273.15).rename("temp_c"),
            masked_band("temp_c"),
        )
    )
    return image.toFloat()


def annual_pet_image(year: ee.Number) -> ee.Image:
    # ERA5-Land daily aggregate potential_evaporation_sum is stored in meters.
    # ECMWF evaporation accumulations are typically negative upward fluxes, so
    # multiply by -1000 to keep pet_mm positive and consistent with prior output.
    collection = annual_era5_land_daily_collection(year).select("potential_evaporation_sum")
    image = ee.Image(
        ee.Algorithms.If(
            collection.size().gt(0),
            collection.sum().multiply(-1000).rename("pet_mm"),
            masked_band("pet_mm"),
        )
    )
    return image.toFloat()


def annual_ndvi_image(year: ee.Number) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    collection = ee.ImageCollection(MODIS_NDVI_COLLECTION).filterDate(start, end).select("NDVI")
    image = ee.Image(
        ee.Algorithms.If(
            collection.size().gt(0),
            collection.mean().multiply(0.0001).rename("ndvi"),
            masked_band("ndvi"),
        )
    )
    return image.toFloat()


def baseline_mean_sd(image_func, band_name: str) -> tuple[ee.Image, ee.Image]:
    """Build each 2000-2024 baseline mean/sd pair once per Python session."""
    if band_name not in _BASELINE_CACHE:
        images = ee.ImageCollection([image_func(ee.Number(year)) for year in BASELINE_YEARS])
        mean = images.mean().rename(f"{band_name}_baseline_mean")
        sd = images.reduce(ee.Reducer.stdDev()).rename(f"{band_name}_baseline_sd")
        _BASELINE_CACHE[band_name] = (mean, sd)

    return _BASELINE_CACHE[band_name]


def with_anomaly(
    annual_image: ee.Image,
    image_func,
    value_band: str,
    anomaly_band: str,
    z_band: str,
) -> ee.Image:
    mean, sd = baseline_mean_sd(image_func, value_band)
    anomaly = annual_image.subtract(mean).rename(anomaly_band)
    z = anomaly.divide(sd).rename(z_band)
    return ee.Image.cat([annual_image.rename(value_band), anomaly, z]).toFloat()


def annual_nonscaled_image(
    year: ee.Number,
    external_specs: Sequence[ExternalRasterSpec] | None = None,
) -> ee.Image:
    """Annual covariates extracted only at the 0-10 km scale."""
    return ee.Image.cat(
        [
            group.image.select(group.band_names)
            for group in nonscaled_covariate_image_groups(year, external_specs)
        ]
    ).toFloat()


def nonscaled_covariate_image_groups(
    year: ee.Number,
    external_specs: Sequence[ExternalRasterSpec] | None = None,
) -> list[CovariateImageGroup]:
    """Build nonscaled covariates grouped by source-native reduction scale."""
    precip = with_anomaly(
        annual_precip_image(year),
        annual_precip_image,
        "precip_mm",
        "precip_anom_mm",
        "precip_z",
    )
    temp = with_anomaly(
        annual_temp_image(year),
        annual_temp_image,
        "temp_c",
        "temp_anom_c",
        "temp_z",
    )
    pet = annual_pet_image(year)
    ndvi = with_anomaly(
        annual_ndvi_image(year),
        annual_ndvi_image,
        "ndvi",
        "ndvi_anom",
        "ndvi_z",
    )
    elevation = ee.Image(ELEVATION_ASSET).select("elevation").rename("elevation_m").toFloat()

    groups = [
        CovariateImageGroup(
            name="era5_precip",
            image=precip.toFloat(),
            band_names=[
                "precip_mm",
                "precip_anom_mm",
                "precip_z",
            ],
            scale_m=ERA5_NATIVE_SCALE_M,
        ),
        CovariateImageGroup(
            name="era5_temp",
            image=temp.toFloat(),
            band_names=[
                "temp_c",
                "temp_anom_c",
                "temp_z",
            ],
            scale_m=ERA5_NATIVE_SCALE_M,
        ),
        CovariateImageGroup(
            name="era5_pet",
            image=pet.toFloat(),
            band_names=["pet_mm"],
            scale_m=ERA5_DAILY_NATIVE_SCALE_M,
        ),
        CovariateImageGroup(
            name="modis_ndvi",
            image=ndvi.toFloat(),
            band_names=["ndvi", "ndvi_anom", "ndvi_z"],
            scale_m=MODIS_NDVI_NATIVE_SCALE_M,
        ),
        CovariateImageGroup(
            name="elevation",
            image=elevation.toFloat(),
            band_names=["elevation_m"],
            scale_m=SRTM_EXTRACTION_SCALE_M,
        ),
    ]

    for spec in external_specs or []:
        if spec.buffer_mode != "scaled":
            groups.append(
                CovariateImageGroup(
                    name=f"external_{spec.covariate_name}",
                    image=ee.Image(spec.gee_asset_id)
                    .select(0)
                    .rename(spec.covariate_name)
                    .toFloat(),
                    band_names=[spec.covariate_name],
                    scale_m=spec.scale_m,
                )
            )

    return groups


def annual_scaled_image(
    year: ee.Number,
    external_specs: Sequence[ExternalRasterSpec] | None = None,
) -> ee.Image:
    """Annual covariates extracted for all spatial buffer scales."""
    return ee.Image.cat(
        [
            group.image.select(group.band_names)
            for group in scaled_covariate_image_groups(year, external_specs)
        ]
    ).toFloat()


def scaled_covariate_image_groups(
    year: ee.Number,
    external_specs: Sequence[ExternalRasterSpec] | None = None,
) -> list[CovariateImageGroup]:
    """Build scaled covariates grouped by source reduction scale."""
    groups = [
        CovariateImageGroup(
            name=family,
            image=scaled_covariate_family_image(year, family),
            band_names=scaled_family_band_names(family),
            scale_m=scaled_family_scale_m(family),
        )
        for family in SCALED_COVARIATE_FAMILIES
    ]

    for spec in external_specs or []:
        if spec.buffer_mode == "scaled":
            groups.append(
                CovariateImageGroup(
                    name=f"external_{spec.covariate_name}",
                    image=ee.Image(spec.gee_asset_id)
                    .select(0)
                    .rename(spec.covariate_name)
                    .toFloat(),
                    band_names=[spec.covariate_name],
                    scale_m=spec.scale_m,
                )
            )

    return groups


def rename_with_suffix(image: ee.Image, base_names: Sequence[str], suffix: str) -> ee.Image:
    return image.select(list(base_names)).rename([f"{name}_{suffix}" for name in base_names])


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


def make_training_buffer_collection(
    points: ee.FeatureCollection,
    ring_suffix: str,
    study_region: ee.Geometry | None = None,
) -> ee.FeatureCollection:
    """Create one simplified donut-buffer FeatureCollection from dataset1 points."""
    ring_lookup = {suffix: (inner_degrees, outer_degrees) for inner_degrees, outer_degrees, suffix in SCALED_RINGS}
    if ring_suffix not in ring_lookup:
        raise ValueError(f"Unknown ring suffix: {ring_suffix}")

    inner_degrees, outer_degrees = ring_lookup[ring_suffix]

    def buffer_feature(feature: ee.Feature) -> ee.Feature:
        geometry = ring_geometry(feature, inner_degrees, outer_degrees)
        max_error = ee.ErrorMargin(buffer_geometry_error_degrees(outer_degrees), "projected")
        projection = degree_projection()
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

    return points.map(buffer_feature)


def reduction_geometry(
    feature: ee.Feature,
    inner_m: int | float,
    outer_m: int | float,
    study_region: ee.Geometry | None = None,
) -> ee.Geometry:
    geometry = ring_geometry(feature, inner_m, outer_m)
    if study_region is None:
        return geometry
    max_error = ee.ErrorMargin(buffer_geometry_error_degrees(float(outer_m)), "projected")
    projection = degree_projection()
    return geometry.intersection(study_region, max_error, projection).simplify(max_error, projection)


def reduce_image_at_feature(
    image: ee.Image,
    feature: ee.Feature,
    inner_m: int,
    outer_m: int,
    scale_m: int | float = EXTRACTION_SCALE_M,
    study_region: ee.Geometry | None = None,
) -> ee.Dictionary:
    reduction_kwargs = {
        "reducer": ee.Reducer.mean(),
        "geometry": reduction_geometry(feature, inner_m, outer_m, study_region),
        "maxPixels": 1e13,
        "bestEffort": True,
        "tileScale": 4,
    }
    if float(scale_m) < 1:
        reduction_kwargs["crs"] = LAT_LONG_CRS
        reduction_kwargs["crsTransform"] = degree_grid_transform(float(scale_m))
    else:
        reduction_kwargs["scale"] = int(scale_m)

    values = image.reduceRegion(**reduction_kwargs)
    return values


def reduce_image_over_buffer_features(
    image: ee.Image,
    features: ee.FeatureCollection,
    band_names: Sequence[str],
    ring_suffix: str,
    scale_m: int | float | None = None,
) -> ee.FeatureCollection:
    """Extract mean image values over precomputed buffer polygon features."""
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


def join_feature_collection_properties(
    left: ee.FeatureCollection,
    right: ee.FeatureCollection,
    property_names: Sequence[str],
    join_key: str = "id",
) -> ee.FeatureCollection:
    """Join right-side properties onto left-side features by id."""
    joined = ee.Join.saveFirst("_matched").apply(
        left,
        right,
        ee.Filter.equals(leftField=join_key, rightField=join_key),
    )

    def copy_properties(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        matched = ee.Feature(feature.get("_matched"))
        keep_names = feature.propertyNames().remove("_matched")
        clean = ee.Feature(None, feature.toDictionary(keep_names))
        return clean.copyProperties(matched, list(property_names))

    return ee.FeatureCollection(joined).map(copy_properties)


def merge_ring_feature_collections(
    ring_collections: Sequence[tuple[ee.FeatureCollection, Sequence[str]]],
) -> ee.FeatureCollection:
    """Merge ring-specific FeatureCollections into one wide table."""
    if not ring_collections:
        raise ValueError("At least one ring FeatureCollection is required.")

    combined = ring_collections[0][0]
    for collection, property_names in ring_collections[1:]:
        combined = join_feature_collection_properties(combined, collection, property_names)

    return combined


def analysis_projection(scale_m: int = EXTRACTION_SCALE_M) -> ee.Projection:
    return ee.Projection(ANALYSIS_CRS).atScale(scale_m)


def ring_mean_image(
    image: ee.Image,
    band_names: Sequence[str],
    inner_m: int,
    outer_m: int,
    suffix: str,
    scale_m: int = EXTRACTION_SCALE_M,
) -> ee.Image:
    """Create image bands representing mean values inside a circular buffer.

    This is the efficient prediction-grid path: calculate buffer summaries as
    raster operations, then sample those bands at point/grid-cell centers.
    """
    projection = analysis_projection(scale_m)
    image = image.select(list(band_names)).reproject(projection)
    valid = image.mask().toFloat().unmask(0).reproject(projection)
    filled = image.unmask(0).reproject(projection)

    outer_kernel = ee.Kernel.circle(radius=outer_m, units="meters", normalize=False)
    outer_sum = filled.reduceNeighborhood(
        reducer=ee.Reducer.sum(),
        kernel=outer_kernel,
        skipMasked=False,
    )
    outer_count = valid.reduceNeighborhood(
        reducer=ee.Reducer.sum(),
        kernel=outer_kernel,
        skipMasked=False,
    )

    if inner_m > 0:
        inner_kernel = ee.Kernel.circle(radius=inner_m, units="meters", normalize=False)
        inner_sum = filled.reduceNeighborhood(
            reducer=ee.Reducer.sum(),
            kernel=inner_kernel,
            skipMasked=False,
        )
        inner_count = valid.reduceNeighborhood(
            reducer=ee.Reducer.sum(),
            kernel=inner_kernel,
            skipMasked=False,
        )
        numerator = outer_sum.subtract(inner_sum)
        denominator = outer_count.subtract(inner_count)
    else:
        numerator = outer_sum
        denominator = outer_count

    out = numerator.divide(denominator).updateMask(denominator.gt(0))
    return out.rename([f"{name}_{suffix}" for name in band_names]).toFloat()


def scaled_band_names(external_specs: Sequence[ExternalRasterSpec] | None = None) -> list[str]:
    names = list(BASE_SCALED_BANDS)
    for spec in external_specs or []:
        if spec.buffer_mode == "scaled":
            names.append(spec.covariate_name)
    return names


def nonscaled_band_names(external_specs: Sequence[ExternalRasterSpec] | None = None) -> list[str]:
    names = list(BASE_NONSCALED_BANDS)
    for spec in external_specs or []:
        if spec.buffer_mode != "scaled":
            names.append(spec.covariate_name)
    return names


def annual_buffered_covariate_image(
    year: int,
    external_specs: Sequence[ExternalRasterSpec] | None = None,
    study_region: ee.Geometry | None = None,
) -> ee.Image:
    """Build all model covariates as buffer-mean image bands for one year."""
    year_number = ee.Number(int(year))
    images: list[ee.Image] = []

    for family in SCALED_COVARIATE_FAMILIES:
        band_names = scaled_family_band_names(family)

        for inner_degrees, outer_degrees, suffix in SCALED_RINGS:
            scale_m = scaled_family_scale_m(family, suffix)
            source = scaled_covariate_family_image(year_number, family, scale_m=scale_m)
            if study_region is not None:
                source = source.clip(study_region)

            images.append(
                ring_mean_image(
                    source,
                    band_names,
                    inner_degrees,
                    outer_degrees,
                    suffix,
                    scale_m=scale_m,
                )
            )

    for spec in external_specs or []:
        if spec.buffer_mode == "scaled":
            source = ee.Image(spec.gee_asset_id).select(0).rename(spec.covariate_name).toFloat()
            if study_region is not None:
                source = source.clip(study_region)

            for inner_degrees, outer_degrees, suffix in SCALED_RINGS:
                images.append(
                    ring_mean_image(
                        source,
                        [spec.covariate_name],
                        inner_degrees,
                        outer_degrees,
                        suffix,
                        scale_m=spec.scale_m,
                    )
                )

    nonscaled_groups = nonscaled_covariate_image_groups(year_number, external_specs)
    for inner_degrees, outer_degrees, suffix in NONSCALED_RINGS:
        for group in nonscaled_groups:
            source = group.image
            if study_region is not None:
                source = source.clip(study_region)

            images.append(
                ring_mean_image(
                    source,
                    group.band_names,
                    inner_degrees,
                    outer_degrees,
                    suffix,
                    scale_m=group.scale_m,
                )
            )

    return ee.Image.cat(images).toFloat()


def sample_buffered_covariates(
    features: ee.FeatureCollection,
    year: int,
    external_specs: Sequence[ExternalRasterSpec] | None = None,
    study_region: ee.Geometry | None = None,
) -> ee.FeatureCollection:
    """Sample annual buffered covariate image bands at feature locations."""
    image = annual_buffered_covariate_image(year, external_specs, study_region).unmask(NO_DATA_VALUE)

    return image.sampleRegions(
        collection=features,
        scale=EXTRACTION_SCALE_M,
        geometries=False,
        tileScale=4,
    )


def add_covariates_to_feature_with_images(
    feature: ee.Feature,
    scaled: ee.Image,
    nonscaled: ee.Image,
    external_specs: Sequence[ExternalRasterSpec] | None = None,
    study_region: ee.Geometry | None = None,
    covariate_group: str = "all",
    scaled_ring_suffixes: Sequence[str] | None = None,
) -> ee.Feature:
    if covariate_group not in {"all", "scaled", "nonscaled"}:
        raise ValueError("covariate_group must be 'all', 'scaled', or 'nonscaled'.")

    out = feature
    scaled_names = scaled_band_names(external_specs)
    nonscaled_names = nonscaled_band_names(external_specs)
    scaled_rings = selected_scaled_rings(scaled_ring_suffixes)

    if covariate_group in {"all", "scaled"}:
        for inner_degrees, outer_degrees, suffix in scaled_rings:
            image = rename_with_suffix(scaled, scaled_names, suffix)
            out = out.set(
                reduce_image_at_feature(image, feature, inner_degrees, outer_degrees, study_region=study_region)
            )

    if covariate_group in {"all", "nonscaled"}:
        for inner_degrees, outer_degrees, suffix in NONSCALED_RINGS:
            image = rename_with_suffix(nonscaled, nonscaled_names, suffix)
            out = out.set(
                reduce_image_at_feature(image, feature, inner_degrees, outer_degrees, study_region=study_region)
            )

    return out


def add_covariates_to_feature_with_groups(
    feature: ee.Feature,
    scaled_groups: Sequence[CovariateImageGroup],
    nonscaled_groups: Sequence[CovariateImageGroup],
    year: ee.Number | None = None,
    study_region: ee.Geometry | None = None,
    scaled_ring_suffixes: Sequence[str] | None = None,
) -> ee.Feature:
    """Add buffered covariate means using each source group's reduction scale."""
    out = feature
    scaled_rings = selected_scaled_rings(scaled_ring_suffixes)

    for group in scaled_groups:
        for inner_degrees, outer_degrees, suffix in scaled_rings:
            scale_m = (
                scaled_family_scale_m(group.name, suffix)
                if group.name in SCALED_COVARIATE_FAMILIES
                else group.scale_m
            )
            year_number = year if year is not None else ee.Number(feature.get("year"))
            source = (
                scaled_covariate_family_image(year_number, group.name, scale_m=scale_m)
                if group.name in SCALED_COVARIATE_FAMILIES
                else group.image
            )
            image = rename_with_suffix(source, group.band_names, suffix)
            out = out.set(
                reduce_image_at_feature(
                    image,
                    feature,
                    inner_degrees,
                    outer_degrees,
                    scale_m=scale_m,
                    study_region=study_region,
                )
            )

    for group in nonscaled_groups:
        for inner_degrees, outer_degrees, suffix in NONSCALED_RINGS:
            image = rename_with_suffix(group.image, group.band_names, suffix)
            out = out.set(
                reduce_image_at_feature(
                    image,
                    feature,
                    inner_degrees,
                    outer_degrees,
                    scale_m=group.scale_m,
                    study_region=study_region,
                )
            )

    return out


def extract_point_buffer_covariates_for_year(
    features: ee.FeatureCollection,
    year: int,
    external_specs: Sequence[ExternalRasterSpec] | None = None,
    study_region: ee.Geometry | None = None,
    covariate_group: str = "all",
    scaled_ring_suffixes: Sequence[str] | None = None,
) -> ee.FeatureCollection:
    """Extract training covariates with per-feature buffer reductions.

    This path is safer for the training table than raster neighborhood kernels:
    it avoids creating 100 km moving-window rasters across the full study area.
    """
    if covariate_group not in {"all", "scaled", "nonscaled"}:
        raise ValueError("covariate_group must be 'all', 'scaled', or 'nonscaled'.")

    year_number = ee.Number(int(year))
    scaled_groups: list[CovariateImageGroup] = []
    nonscaled_groups: list[CovariateImageGroup] = []

    if covariate_group in {"all", "scaled"}:
        scaled_groups = scaled_covariate_image_groups(year_number, external_specs)

    if covariate_group in {"all", "nonscaled"}:
        nonscaled_groups = nonscaled_covariate_image_groups(year_number, external_specs)

    return features.map(
        lambda feature: add_covariates_to_feature_with_groups(
            feature=feature,
            scaled_groups=scaled_groups,
            nonscaled_groups=nonscaled_groups,
            year=year_number,
            study_region=None,
            scaled_ring_suffixes=scaled_ring_suffixes,
        ).setGeometry(None)
    )


def extract_covariates_for_year(
    features: ee.FeatureCollection,
    year: int,
    external_specs: Sequence[ExternalRasterSpec] | None = None,
    study_region: ee.Geometry | None = None,
) -> ee.FeatureCollection:
    """Extract covariates using raster neighborhood images for dense grids."""
    return sample_buffered_covariates(features, year, external_specs, study_region)


def extract_training_covariates(
    points: ee.FeatureCollection,
    years: Sequence[int],
    external_specs: Sequence[ExternalRasterSpec] | None = None,
    study_region: ee.Geometry | None = None,
    covariate_group: str = "all",
    scaled_ring_suffixes: Sequence[str] | None = None,
) -> ee.FeatureCollection:
    combined = ee.FeatureCollection([])

    for year in sorted({int(y) for y in years}):
        year_points = points.filter(ee.Filter.eq("year", year))
        combined = combined.merge(
            extract_point_buffer_covariates_for_year(
                year_points,
                year,
                external_specs,
                study_region,
                covariate_group,
                scaled_ring_suffixes,
            )
        )

    return combined


def extract_scaled_family_from_buffer_assets(
    year: int,
    family: str,
    buffer_asset_ids: dict[str, str],
    ring_suffixes: Sequence[str] | None = None,
) -> ee.FeatureCollection:
    """Extract one scaled covariate family from saved buffer assets."""
    year_int = int(year)
    year_number = ee.Number(year_int)
    band_names = scaled_family_band_names(family)

    ring_collections: list[tuple[ee.FeatureCollection, Sequence[str]]] = []
    for _, _, suffix in selected_scaled_rings(ring_suffixes):
        if suffix not in buffer_asset_ids:
            raise ValueError(f"Missing buffer asset id for ring {suffix}.")

        scale_m = scaled_family_scale_m(family, suffix)
        image = scaled_covariate_family_image(year_number, family, scale_m=scale_m)
        buffer_features = ee.FeatureCollection(buffer_asset_ids[suffix]).filter(
            ee.Filter.eq("year", year_int)
        )
        properties = [f"{name}_{suffix}" for name in band_names]
        stats = reduce_image_over_buffer_features(
            image=image,
            features=buffer_features,
            band_names=band_names,
            ring_suffix=suffix,
            scale_m=scale_m,
        )
        ring_collections.append((stats, properties))

    return merge_ring_feature_collections(ring_collections)


def extract_nonscaled_from_buffer_asset(
    year: int,
    buffer_asset_id: str,
    external_specs: Sequence[ExternalRasterSpec] | None = None,
) -> ee.FeatureCollection:
    """Extract all nonscaled covariates from the saved 0-10 km buffer asset."""
    year_int = int(year)
    buffer_features = ee.FeatureCollection(buffer_asset_id).filter(ee.Filter.eq("year", year_int))

    group_collections: list[tuple[ee.FeatureCollection, Sequence[str]]] = []
    for group in nonscaled_covariate_image_groups(ee.Number(year_int), external_specs):
        properties = [f"{name}_0_10km" for name in group.band_names]
        stats = reduce_image_over_buffer_features(
            image=group.image,
            features=buffer_features,
            band_names=group.band_names,
            ring_suffix="0_10km",
            scale_m=group.scale_m,
        )
        group_collections.append((stats, properties))

    return merge_ring_feature_collections(group_collections)


def extract_training_export_group_from_buffer_assets(
    year: int,
    export_group: str,
    buffer_asset_ids: dict[str, str],
    external_specs: Sequence[ExternalRasterSpec] | None = None,
    ring_suffixes: Sequence[str] | None = None,
) -> ee.FeatureCollection:
    """Extract one asset-based training export group."""
    if export_group == "nonscaled":
        if "0_10km" not in buffer_asset_ids:
            raise ValueError("The nonscaled export requires the 0_10km buffer asset.")
        return extract_nonscaled_from_buffer_asset(
            year,
            buffer_asset_ids["0_10km"],
            external_specs,
        )

    return extract_scaled_family_from_buffer_assets(
        year,
        export_group,
        buffer_asset_ids,
        ring_suffixes,
    )


def extract_training_export_group_all_years_from_buffer_assets(
    years: Sequence[int],
    export_group: str,
    buffer_asset_ids: dict[str, str],
    external_specs: Sequence[ExternalRasterSpec] | None = None,
    ring_suffixes: Sequence[str] | None = None,
) -> ee.FeatureCollection:
    """Extract one asset-based training export group for all selected years."""
    combined = ee.FeatureCollection([])
    for year in sorted({int(year) for year in years}):
        combined = combined.merge(
            extract_training_export_group_from_buffer_assets(
                year=year,
                export_group=export_group,
                buffer_asset_ids=buffer_asset_ids,
                external_specs=external_specs,
                ring_suffixes=ring_suffixes,
            )
        )

    return combined


def make_prediction_grid(region: ee.Geometry, scale_m: int = PREDICTION_GRID_SCALE_M) -> ee.FeatureCollection:
    projection = ee.Projection(ANALYSIS_CRS).atScale(scale_m)
    grid = (
        ee.Image.pixelLonLat()
        .reproject(projection)
        .sample(
            region=region,
            projection=projection,
            scale=scale_m,
            geometries=True,
            tileScale=4,
        )
    )

    def add_grid_fields(feature: ee.Feature) -> ee.Feature:
        coords = feature.geometry().coordinates()
        lon = ee.Number(coords.get(0))
        lat = ee.Number(coords.get(1))
        grid_id = lon.format("%.5f").cat("_").cat(lat.format("%.5f"))
        return feature.set({"grid_id": grid_id, "longitude": lon, "latitude": lat})

    return grid.map(add_grid_fields).select(["grid_id", "longitude", "latitude"])


def extract_prediction_grid_covariates(
    grid: ee.FeatureCollection,
    year: int,
    external_specs: Sequence[ExternalRasterSpec] | None = None,
    study_region: ee.Geometry | None = None,
) -> ee.FeatureCollection:
    year_int = int(year)
    grid_for_year = grid.map(lambda feature: feature.set({"year": year_int, "outcome": None}))
    return extract_covariates_for_year(grid_for_year, year_int, external_specs, study_region)


def ensure_ee_folder(asset_root: str, dry_run: bool = False) -> list[dict]:
    """Create an Earth Engine folder path if needed.

    Parent project asset roots must already exist and the active account must
    have permission to create assets there.
    """
    asset_root = asset_root.rstrip("/")
    parts = asset_root.split("/")
    if "assets" not in parts:
        raise ValueError("asset_root must look like 'projects/<project>/assets/<folder>'.")

    asset_index = parts.index("assets")
    current = "/".join(parts[: asset_index + 1])
    rows = []

    for part in parts[asset_index + 1 :]:
        current = f"{current}/{part}"
        row = {"asset_id": current, "state": None}

        try:
            ee.data.getAsset(current)
            row["state"] = "EXISTS"
        except Exception:
            if dry_run:
                row["state"] = "DRY_RUN_CREATE"
            else:
                ee.data.createFolder(current)
                row["state"] = "CREATED"

        rows.append(row)

    return rows


def export_table_to_asset(
    collection: ee.FeatureCollection,
    description: str,
    asset_id: str,
    dry_run: bool = False,
    overwrite: bool = False,
    max_vertices: int = 1_000_000,
) -> dict:
    """Start an Earth Engine table-asset export and return manifest metadata."""
    description = description[:100]
    row = {
        "description": description,
        "asset_id": asset_id,
        "task_id": None,
        "state": "DRY_RUN" if dry_run else None,
    }

    if dry_run:
        print(f"DRY RUN table asset export: {description} -> {asset_id}")
        return row

    if not overwrite:
        try:
            ee.data.getAsset(asset_id)
            row["state"] = "EXISTS"
            print(f"Skipping existing table asset: {asset_id}")
            return row
        except Exception:
            pass

    task = ee.batch.Export.table.toAsset(
        collection=collection,
        description=description,
        assetId=asset_id,
        maxVertices=max_vertices,
        overwrite=overwrite,
    )
    task.start()
    status = task.status()
    row["task_id"] = status.get("id")
    row["state"] = status.get("state")
    print(f"Started table asset export: {description} | state={row['state']} | task_id={row['task_id']}")
    return row


def export_table_to_drive(
    collection: ee.FeatureCollection,
    description: str,
    folder: str = EXPORT_FOLDER,
    selectors: Sequence[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Start a Drive CSV export and return manifest metadata."""
    description = description[:100]
    if description == "dataset2":
        raise ValueError(
            "Refusing to start a plain 'dataset2' Drive export. That name is reserved "
            "for the old all-years task that has failed with out-of-memory errors. "
            "Use by-year descriptions like 'dataset2_2001'."
        )
    row = {
        "description": description,
        "folder": folder,
        "file_prefix": description,
        "task_id": None,
        "state": "DRY_RUN" if dry_run else None,
    }

    if dry_run:
        print(f"DRY RUN export: {description} -> Drive/{folder}")
        return row

    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        fileFormat="CSV",
        selectors=list(selectors) if selectors else None,
    )
    task.start()
    status = task.status()
    row["task_id"] = status.get("id")
    row["state"] = status.get("state")
    print(f"Started export: {description} | state={row['state']} | task_id={row['task_id']}")
    return row


def feature_collection_to_dataframe(collection: ee.FeatureCollection) -> pd.DataFrame:
    """Fetch a small FeatureCollection to a local DataFrame.

    This is convenient for toy tests. For full 10,000-point or grid exports,
    Drive export is more reliable.
    """
    info = collection.getInfo()
    rows = []
    for feature in info["features"]:
        rows.append(feature.get("properties", {}))
    return pd.DataFrame(rows)


def write_manifest(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def training_selectors(predictors: Sequence[str]) -> list[str]:
    return list(TRAINING_BASE_COLUMNS) + list(predictors)


def prediction_selectors(predictors: Sequence[str]) -> list[str]:
    return ["grid_id", "year", "latitude", "longitude", "outcome"] + list(predictors)
