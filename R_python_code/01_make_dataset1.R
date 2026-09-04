#### Step 1: Create dataset1 for the event ecological niche pipeline ####

# This script creates the base presence/pseudo-absence table used by the
# Google Earth Engine covariate extraction step.
#
# Expected output:
#   data/dataset1.csv
#
# Required columns in output:
#   id, year, latitude, longitude, outcome

#### Configuration ####
get_current_script_path <- function() {
  command_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", command_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE))
  }

  for (frame in rev(sys.frames())) {
    if (!is.null(frame$ofile)) {
      return(normalizePath(frame$ofile, winslash = "/", mustWork = TRUE))
    }
  }

  NA_character_
}

find_code_dir <- function() {
  script_path <- get_current_script_path()
  candidates <- c(
    if (!is.na(script_path)) normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = FALSE),
    file.path(getwd(), "KSPH_Code"),
    getwd()
  )

  candidates <- unique(normalizePath(candidates, winslash = "/", mustWork = FALSE))
  for (candidate in candidates) {
    if (
      file.exists(file.path(candidate, "config", "predictor_list.csv")) &&
      dir.exists(file.path(candidate, "R_python_code"))
    ) {
      return(candidate)
    }
  }

  stop(
    "Could not locate the KSPH Code directory. Run this with source('R_python_code/01_make_dataset1.R') ",
    "from the KSPH Code repo root, or source('KSPH Code/R_python_code/01_make_dataset1.R') from the parent folder.",
    call. = FALSE
  )
}

CODE_DIR <- find_code_dir()
PROJECT_DIR <- normalizePath(file.path(CODE_DIR, ".."), winslash = "/", mustWork = TRUE)
WINDOWS_USER_R_LIB <- file.path(
  Sys.getenv("LOCALAPPDATA"),
  "R",
  "win-library",
  paste0(R.version$major, ".", R.version$minor)
)
if (dir.exists(WINDOWS_USER_R_LIB) && !WINDOWS_USER_R_LIB %in% .libPaths()) {
  .libPaths(c(WINDOWS_USER_R_LIB, .libPaths()))
}

PRESENCE_CSV <- file.path(CODE_DIR, "config", "outcomes_csv.csv")
STUDY_AREA_FILE <- file.path(CODE_DIR, "config", "africacountries_nolakes.shp")
OUTPUT_CSV <- file.path(CODE_DIR, "data", "dataset1.csv")

N_PSEUDO_ABSENCE <- 5000
ABSENCE_YEARS <- 2001:2025
RANDOM_SEED <- 20260813
SAMPLING_VERSION <- "pseudo_absence_random_v1"
STUDY_AREA_NAME <- "equatorial_africa"

# STUDY_AREA_FILE is the polygon used for pseudo-absence sampling. If
# STUDY_AREA_BBOX is set, the polygon is clipped to this lon/lat box first.
# Set STUDY_AREA_BBOX <- NULL to use the full extent of the provided polygon,
# e.g. when a future analyst provides a country-specific shapefile in place of 
# the africa_nolakes shapefile i am using.
STUDY_AREA_BBOX <- c(xmin = -15.5, ymin = -10.0, xmax = 51.0, ymax = 10.0)
FILTER_PRESENCES_TO_STUDY_AREA <- TRUE
ALLOW_BBOX_FALLBACK <- FALSE

# Set this to a positive number if you want to prevent pseudo-absences from
# being sampled close to known presences. Kept at 0 for the current plan.
EXCLUDE_AROUND_PRESENCES_M <- 0

#### Helpers ####
require_package <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(
      sprintf(
        "Package '%s' is required. Install it before running this script.",
        package
      ),
      call. = FALSE
    )
  }
}

normalize_study_area_bbox <- function() {
  if (is.null(STUDY_AREA_BBOX) || length(STUDY_AREA_BBOX) == 0) {
    return(NULL)
  }

  required_names <- c("xmin", "ymin", "xmax", "ymax")
  if (is.null(names(STUDY_AREA_BBOX)) || !all(required_names %in% names(STUDY_AREA_BBOX))) {
    stop("STUDY_AREA_BBOX must be named xmin, ymin, xmax, ymax, or set to NULL.", call. = FALSE)
  }
  bbox <- as.numeric(STUDY_AREA_BBOX[required_names])
  names(bbox) <- required_names
  if (any(!is.finite(bbox))) {
    stop("STUDY_AREA_BBOX values must be finite numbers.", call. = FALSE)
  }
  if (bbox["xmin"] >= bbox["xmax"] || bbox["ymin"] >= bbox["ymax"]) {
    stop("STUDY_AREA_BBOX must have xmin < xmax and ymin < ymax.", call. = FALSE)
  }

  bbox
}

format_study_area_bbox <- function(bbox) {
  if (is.null(bbox)) {
    return("full provided polygon extent")
  }
  paste0(
    "xmin=", bbox["xmin"],
    ", ymin=", bbox["ymin"],
    ", xmax=", bbox["xmax"],
    ", ymax=", bbox["ymax"]
  )
}

load_active_study_area <- function() {
  old_s2 <- sf::sf_use_s2(FALSE)
  on.exit(sf::sf_use_s2(old_s2), add = TRUE)

  study_area <- sf::st_read(STUDY_AREA_FILE, quiet = TRUE)
  study_area <- sf::st_transform(study_area, 4326)
  study_area <- suppressWarnings(sf::st_make_valid(study_area))

  bbox <- normalize_study_area_bbox()
  if (!is.null(bbox)) {
    study_area <- suppressWarnings(sf::st_crop(study_area, sf::st_bbox(bbox, crs = sf::st_crs(4326))))
  }

  study_area <- suppressWarnings(sf::st_collection_extract(study_area, "POLYGON"))
  study_area <- study_area[!sf::st_is_empty(study_area), , drop = FALSE]
  if (nrow(study_area) == 0) {
    stop(
      "No study-area geometry remains after applying STUDY_AREA_FILE and STUDY_AREA_BBOX.",
      call. = FALSE
    )
  }

  study_area
}

filter_presence_to_study_area <- function(presence, study_area) {
  if (!FILTER_PRESENCES_TO_STUDY_AREA) {
    return(presence)
  }

  old_s2 <- sf::sf_use_s2(FALSE)
  on.exit(sf::sf_use_s2(old_s2), add = TRUE)

  presence_sf <- sf::st_as_sf(
    presence,
    coords = c("longitude", "latitude"),
    crs = 4326,
    remove = FALSE
  )
  inside <- lengths(sf::st_intersects(presence_sf, study_area)) > 0
  dropped <- sum(!inside)
  if (dropped > 0) {
    message("Dropped ", dropped, " presence rows outside the active study area.")
  }

  filtered <- presence[inside, , drop = FALSE]
  row.names(filtered) <- NULL
  if (nrow(filtered) == 0) {
    stop("No presence records remain inside the active study area.", call. = FALSE)
  }

  filtered
}

standardize_presence_columns <- function(df) {
  names(df) <- tolower(names(df))

  lon_candidates <- c("longitude", "lon", "long", "x")
  lat_candidates <- c("latitude", "lat", "y")
  year_candidates <- c("year", "observation_year", "sighting_year")
  id_candidates <- c("id", "record_id", "sighting_id")
  type_candidates <- c("type", "report_type", "origin", "source_type")
  country_candidates <- c("country", "adm0", "country_name")

  lon_col <- lon_candidates[lon_candidates %in% names(df)][1]
  lat_col <- lat_candidates[lat_candidates %in% names(df)][1]
  year_col <- year_candidates[year_candidates %in% names(df)][1]
  id_col <- id_candidates[id_candidates %in% names(df)][1]
  type_col <- type_candidates[type_candidates %in% names(df)][1]
  country_col <- country_candidates[country_candidates %in% names(df)][1]

  if (is.na(lon_col) || is.na(lat_col) || is.na(year_col)) {
    stop(
      "Presence CSV must contain longitude, latitude, and year columns.",
      call. = FALSE
    )
  }

  out <- data.frame(
    id = if (is.na(id_col)) sprintf("BF%05d", seq_len(nrow(df))) else as.character(df[[id_col]]),
    year = as.integer(df[[year_col]]),
    latitude = as.numeric(df[[lat_col]]),
    longitude = as.numeric(df[[lon_col]]),
    outcome = 1L,
    type = if (is.na(type_col)) NA_character_ else as.character(df[[type_col]]),
    country = if (is.na(country_col)) NA_character_ else as.character(df[[country_col]]),
    point_type = "presence",
    source = "outcomes_csv",
    stringsAsFactors = FALSE
  )

  out <- out[stats::complete.cases(out[, c("year", "latitude", "longitude")]), ]
  out <- out[out$year %in% ABSENCE_YEARS, ]

  if (nrow(out) == 0) {
    stop("No valid presence records remain after filtering to ABSENCE_YEARS.", call. = FALSE)
  }

  out
}

sample_absences_with_sf <- function(presence, study_area) {
  old_s2 <- sf::sf_use_s2(FALSE)
  on.exit(sf::sf_use_s2(old_s2), add = TRUE)

  absence_sampling_area <- study_area
  sampling_bbox <- sf::st_bbox(absence_sampling_area)

  if (EXCLUDE_AROUND_PRESENCES_M > 0) {
    presence_sf <- sf::st_as_sf(
      presence,
      coords = c("longitude", "latitude"),
      crs = 4326,
      remove = FALSE
    )
    presence_buffer <- sf::st_buffer(sf::st_transform(presence_sf, 6933), EXCLUDE_AROUND_PRESENCES_M)
    presence_buffer <- sf::st_transform(sf::st_union(presence_buffer), 4326)
    absence_sampling_area <- suppressWarnings(sf::st_difference(study_area, presence_buffer))
    sampling_bbox <- sf::st_bbox(absence_sampling_area)
  }

  kept_points <- list()
  kept_n <- 0
  batch_size <- max(50000, N_PSEUDO_ABSENCE * 3)
  attempt <- 0

  while (kept_n < N_PSEUDO_ABSENCE) {
    attempt <- attempt + 1
    candidates <- data.frame(
      longitude = stats::runif(batch_size, sampling_bbox["xmin"], sampling_bbox["xmax"]),
      latitude = stats::runif(batch_size, sampling_bbox["ymin"], sampling_bbox["ymax"])
    )
    candidates_sf <- sf::st_as_sf(
      candidates,
      coords = c("longitude", "latitude"),
      crs = 4326,
      remove = FALSE
    )
    inside <- lengths(sf::st_intersects(candidates_sf, absence_sampling_area)) > 0
    accepted <- candidates[inside, , drop = FALSE]

    if (nrow(accepted) > 0) {
      kept_points[[length(kept_points) + 1]] <- accepted
      kept_n <- kept_n + nrow(accepted)
    }

    if (attempt > 100) {
      stop("Could not sample enough pseudo-absence points inside the study area.", call. = FALSE)
    }
  }

  absence_coords <- do.call(rbind, kept_points)
  absence_coords <- absence_coords[seq_len(N_PSEUDO_ABSENCE), , drop = FALSE]

  data.frame(
    id = sprintf("PA%05d", seq_len(N_PSEUDO_ABSENCE)),
    year = sample(ABSENCE_YEARS, N_PSEUDO_ABSENCE, replace = TRUE),
    latitude = absence_coords$latitude,
    longitude = absence_coords$longitude,
    outcome = 0L,
    type = "pseudo_absence",
    country = NA_character_,
    point_type = "pseudo_absence",
    source = "random_background",
    stringsAsFactors = FALSE
  )
}

sample_absences_with_bbox <- function() {
  bbox <- normalize_study_area_bbox()
  if (is.null(bbox)) {
    stop(
      "ALLOW_BBOX_FALLBACK requires STUDY_AREA_BBOX. Install sf to sample from the provided polygon, ",
      "or set STUDY_AREA_BBOX to an explicit xmin/ymin/xmax/ymax test box.",
      call. = FALSE
    )
  }

  if (EXCLUDE_AROUND_PRESENCES_M > 0) {
    stop(
      "EXCLUDE_AROUND_PRESENCES_M requires the sf package. Install sf or set exclusion to 0.",
      call. = FALSE
    )
  }

  warning(
    "Package 'sf' is not installed; sampling pseudo-absences from STUDY_AREA_BBOX. ",
    "Install sf to sample from the configured study-area polygon instead of its bounding box."
  )

  data.frame(
    id = sprintf("PA%05d", seq_len(N_PSEUDO_ABSENCE)),
    year = sample(ABSENCE_YEARS, N_PSEUDO_ABSENCE, replace = TRUE),
    latitude = stats::runif(N_PSEUDO_ABSENCE, bbox["ymin"], bbox["ymax"]),
    longitude = stats::runif(N_PSEUDO_ABSENCE, bbox["xmin"], bbox["xmax"]),
    outcome = 0L,
    type = "pseudo_absence",
    country = NA_character_,
    point_type = "pseudo_absence",
    source = "random_background_bbox",
    stringsAsFactors = FALSE
  )
}

#### Main ####

dir.create(dirname(OUTPUT_CSV), showWarnings = FALSE, recursive = TRUE)

if (!file.exists(PRESENCE_CSV)) {
  stop(
    "Could not find outcomes CSV: ", PRESENCE_CSV,
    "\nPlace the outcomes file at data/outcomes_csv.csv when running from the KSPH Code repo root.",
    call. = FALSE
  )
}

set.seed(RANDOM_SEED)

presence_raw <- utils::read.csv(PRESENCE_CSV, stringsAsFactors = FALSE)
presence <- standardize_presence_columns(presence_raw)

study_area <- NULL
if (requireNamespace("sf", quietly = TRUE)) {
  study_area <- load_active_study_area()
  presence <- filter_presence_to_study_area(presence, study_area)
}

absence <- if (!is.null(study_area)) {
  sample_absences_with_sf(presence, study_area)
} else if (ALLOW_BBOX_FALLBACK) {
  sample_absences_with_bbox()
} else {
  stop(
    "Package 'sf' is required to sample pseudo-absences from ",
    STUDY_AREA_FILE,
    ".\nInstall sf or set ALLOW_BBOX_FALLBACK <- TRUE for quick testing only.",
    call. = FALSE
  )
}

dataset1 <- rbind(presence, absence)
dataset1$sampling_version <- SAMPLING_VERSION
dataset1$created_date <- as.character(Sys.Date())

dataset1 <- dataset1[, c(
  "id", "year", "latitude", "longitude", "outcome",
  "type", "country", "point_type", "source", "sampling_version", "created_date"
)]
dataset1 %>% view()

utils::write.csv(dataset1, OUTPUT_CSV, row.names = FALSE)

message("Wrote dataset1: ", OUTPUT_CSV)
message("Study area name: ", STUDY_AREA_NAME)
message("Study area file: ", STUDY_AREA_FILE)
message("Study area bbox: ", format_study_area_bbox(normalize_study_area_bbox()))
message("Rows: ", nrow(dataset1))
message("Presences: ", sum(dataset1$outcome == 1))
message("Pseudo-absences: ", sum(dataset1$outcome == 0))


