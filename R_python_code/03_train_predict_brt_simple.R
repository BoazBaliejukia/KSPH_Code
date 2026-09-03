#### 03 Train BRT Ensemble And Predict Event Probability ####

# This first modeling script intentionally keeps the workflow simple:
#   1. read the completed training dataset
#   2. create 10 sampled training datasets with 100 controls per event
#   3. train one BRT model for each sampled dataset
#   4. load the fitted model list and prediction-grid covariates
#   5. predict each model over the prediction-grid data frame
#   6. summarize model predictions by grid cell-year, then rasterize annual summaries
#   7. optionally calculate TreeSHAP values for prediction-grid rows


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
    file.path(getwd(), "KSPH Code"),
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
    "Could not locate the KSPH Code directory. Run this with source('R_python_code/03_train_predict_brt_simple.R') ",
    "from the KSPH Code repo root, or source('KSPH Code/R_python_code/03_train_predict_brt_simple.R') from the parent folder.",
    call. = FALSE
  )
}

CODE_DIR <- find_code_dir()
PROJECT_DIR <- normalizePath(file.path(CODE_DIR, ".."), winslash = "/", mustWork = TRUE)

# Help R find packages installed in the per-user Windows library, e.g. R/win-library/4.6.
LOCALAPPDATA_DIR <- normalizePath(Sys.getenv("LOCALAPPDATA"), winslash = "/", mustWork = FALSE)
WINDOWS_USER_R_LIB <- file.path(
  LOCALAPPDATA_DIR,
  "R",
  "win-library",
  paste(R.version$major, strsplit(R.version$minor, "\\.")[[1]][1], sep = ".")
)
if (dir.exists(WINDOWS_USER_R_LIB) && !WINDOWS_USER_R_LIB %in% .libPaths()) {
  .libPaths(c(WINDOWS_USER_R_LIB, .libPaths()))
}

# Leave both as "" to use the current all-types workflow and existing folders:
#   models
#   outputs
# Set TRAINING_TYPE_FILTER <- "Z" to fit the model only on type Z event
# observations; all pseudo-absence/control rows are retained.
# If TRAINING_TYPE_FILTER is set and ANALYSIS_NAME is left "", results are
# written to models/type_Z and outputs/type_Z.
ANALYSIS_NAME <- ""
TRAINING_TYPE_FILTER <- ""
RUN_PERFORMANCE_EVALUATION <- TRUE
RUN_PERFORMANCE_FIGURES <- TRUE

trim_nonempty_values <- function(values) {
  if (is.null(values) || length(values) == 0) {
    return(character(0))
  }

  values <- trimws(as.character(values))
  values[!is.na(values) & nzchar(values)]
}

sanitize_path_component <- function(value) {
  value <- trimws(as.character(value))
  value <- gsub("[^A-Za-z0-9]+", "_", value)
  value <- gsub("^_+|_+$", "", value)
  if (!nzchar(value)) {
    stop("Analysis folder name cannot be blank after sanitizing.", call. = FALSE)
  }
  value
}

ACTIVE_TRAINING_TYPE_FILTER <- trim_nonempty_values(TRAINING_TYPE_FILTER)
ACTIVE_ANALYSIS_NAME <- trim_nonempty_values(ANALYSIS_NAME)
if (length(ACTIVE_ANALYSIS_NAME) > 1) {
  stop("ANALYSIS_NAME must be a single value.", call. = FALSE)
}
if (length(ACTIVE_ANALYSIS_NAME) == 0 && length(ACTIVE_TRAINING_TYPE_FILTER) > 0) {
  ACTIVE_ANALYSIS_NAME <- paste0(
    "type_",
    paste(vapply(ACTIVE_TRAINING_TYPE_FILTER, sanitize_path_component, character(1)), collapse = "_")
  )
}
if (length(ACTIVE_ANALYSIS_NAME) == 1) {
  ACTIVE_ANALYSIS_NAME <- sanitize_path_component(ACTIVE_ANALYSIS_NAME)
} else {
  ACTIVE_ANALYSIS_NAME <- ""
}

TRAINING_CSV <- file.path(CODE_DIR, "data", "dataset2.csv")
PREDICTION_GRID_CSV <- file.path(CODE_DIR, "data", "prediction_grid_covariates_2020_2025.csv")
AFRICA_COUNTRY_BORDER_FILE <- file.path(CODE_DIR, "config", "africacountries_nolakes.shp")

MODEL_DIR <- if (nzchar(ACTIVE_ANALYSIS_NAME)) {
  file.path(CODE_DIR, "models", ACTIVE_ANALYSIS_NAME)
} else {
  file.path(CODE_DIR, "models")
}
OUTPUT_DIR <- if (nzchar(ACTIVE_ANALYSIS_NAME)) {
  file.path(CODE_DIR, "outputs", ACTIVE_ANALYSIS_NAME)
} else {
  file.path(CODE_DIR, "outputs")
}
PREDICTION_TABLE_DIR <- file.path(OUTPUT_DIR, "predictions", "tables")
ANNUAL_SUMMARY_RASTER_DIR <- file.path(OUTPUT_DIR, "predictions", "annual_summary_rasters")
ANNUAL_SUMMARY_PLOT_DIR <- file.path(OUTPUT_DIR, "predictions", "annual_summary_plots")
ROR_ESTIMATE_DIR <- file.path(OUTPUT_DIR, "predictions", "ROR estimates")
ROR_ESTIMATE_PLOT_DIR <- file.path(ROR_ESTIMATE_DIR, "plots")
ROR_CHANGE_DIR <- file.path(OUTPUT_DIR, "predictions", "ROR 1yr ratios")
ROR_CHANGE_PLOT_DIR <- file.path(ROR_CHANGE_DIR, "plots")
RAW_ODDS_CHANGE_DIR <- file.path(OUTPUT_DIR, "predictions", "raw odds 1yr ratios")
RAW_ODDS_CHANGE_PLOT_DIR <- file.path(RAW_ODDS_CHANGE_DIR, "plots")
PIXEL_RELATIVE_PREDICTION_DIR <- file.path(OUTPUT_DIR, "predictions", "pixel relative prediction ratios")
PIXEL_RELATIVE_PREDICTION_PLOT_DIR <- file.path(PIXEL_RELATIVE_PREDICTION_DIR, "plots")
MEAN_DERIVED_FIGURE_DIR <- file.path(OUTPUT_DIR, "predictions", "mean prediction plus top 1 figures")
MODEL_DIAGNOSTIC_DIR <- file.path(OUTPUT_DIR, "model_diagnostics")
MODEL_DIAGNOSTIC_PLOT_DIR <- file.path(MODEL_DIAGNOSTIC_DIR, "plots")
SHAP_DIR <- file.path(OUTPUT_DIR, "predictions", "shap")
SHAP_TABLE_DIR <- file.path(SHAP_DIR, "tables")
PERFORMANCE_EVALUATION_DIR <- file.path(OUTPUT_DIR, "performance_evaluation")

DSL_RDS <- file.path(MODEL_DIR, "dsl.rds")
PREDICTOR_NAMES_RDS <- file.path(MODEL_DIR, "predictor_names.rds")
PREDICTOR_NAMES_CSV <- file.path(MODEL_DIR, "predictor_names.csv")
MODEL_LIST_RDS <- file.path(MODEL_DIR, "model_list.rds")
MODEL_PREDICTION_TABLE_CSV <- file.path(
  PREDICTION_TABLE_DIR,
  "prediction_grid_model_predictions_2020_2025.csv"
)
ANNUAL_SUMMARY_PREDICTION_CSV <- file.path(
  PREDICTION_TABLE_DIR,
  "prediction_grid_prediction_summaries_2020_2025.csv"
)
SHAP_MEAN_TABLE_CSV <- file.path(
  SHAP_TABLE_DIR,
  "prediction_grid_mean_shap_2020_2025.csv"
)
SHAP_COMPLETE_TABLE_CSV <- file.path(
  SHAP_TABLE_DIR,
  "prediction_grid_predictions_with_mean_shap_2020_2025.csv"
)
SHAP_CHANGE_TABLE_CSV <- file.path(
  SHAP_TABLE_DIR,
  "prediction_grid_mean_shap_changes_2021_2025.csv"
)
PERFORMANCE_EVALUATION_POINT_PREDICTIONS_CSV <- file.path(
  PERFORMANCE_EVALUATION_DIR,
  "apparent_training_point_predictions_2020_2025.csv"
)
PERFORMANCE_EVALUATION_METRICS_BY_YEAR_CSV <- file.path(
  PERFORMANCE_EVALUATION_DIR,
  "apparent_performance_metrics_by_year.csv"
)
PERFORMANCE_EVALUATION_METRICS_OVERALL_CSV <- file.path(
  PERFORMANCE_EVALUATION_DIR,
  "apparent_performance_metrics_overall.csv"
)
PERFORMANCE_EVALUATION_CARET_METRICS_BY_YEAR_CSV <- file.path(
  PERFORMANCE_EVALUATION_DIR,
  "apparent_caret_metrics_by_year.csv"
)
PERFORMANCE_EVALUATION_CARET_METRICS_OVERALL_CSV <- file.path(
  PERFORMANCE_EVALUATION_DIR,
  "apparent_caret_metrics_overall.csv"
)
PERFORMANCE_EVALUATION_ROC_CURVE_CSV <- file.path(
  PERFORMANCE_EVALUATION_DIR,
  "apparent_roc_curve_overall.csv"
)
PERFORMANCE_EVALUATION_PR_CURVE_CSV <- file.path(
  PERFORMANCE_EVALUATION_DIR,
  "apparent_pr_curve_overall.csv"
)
PERFORMANCE_EVALUATION_ROC_PNG <- file.path(
  PERFORMANCE_EVALUATION_DIR,
  "apparent_roc_curve_overall.png"
)
PERFORMANCE_EVALUATION_PR_PNG <- file.path(
  PERFORMANCE_EVALUATION_DIR,
  "apparent_pr_curve_overall.png"
)

N_DATASETS <- 50
CONTROLS_PER_EVENT <- 50
CONTROL_SAMPLE_WITH_REPLACEMENT <- FALSE
RANDOM_SEED <- 20260826

OUTCOME_COLUMN <- "outcome"
EVENT_VALUE <- 1L
CONTROL_VALUE <- 0L
TRAINING_BASE_COLUMNS <- c("id", "year", "latitude", "longitude", "outcome", "type", "country")
PREDICTION_BASE_COLUMNS <- c("grid_id", "grid_batch", "x", "y", "year", "longitude", "latitude", "country")

TREE_COMPLEXITY <- 5
LEARNING_RATE <- 0.0012
BAG_FRACTION <- 0.7
N_FOLDS <- 10
BRT_FAMILY <- "bernoulli"

PREDICTION_YEARS <- 2020:2025
EVALUATION_YEARS <- PREDICTION_YEARS
EVALUATION_PREDICTION_LAYER <- "pred_mean"
EVALUATION_THRESHOLD_METHOD <- "maximize_sens_ppv_product"
EVALUATION_THRESHOLD_SCOPE <- "overall"
EVALUATION_TOP_PROPORTION <- 0.01
EVALUATION_FIXED_THRESHOLD <- 0.5
RASTER_CRS <- "EPSG:4326"
COORDINATE_ROUND_DIGITS <- 10
OVERWRITE_MODEL_PREDICTION_TABLE <- TRUE
OVERWRITE_ANNUAL_SUMMARIES <- TRUE
OVERWRITE_ANNUAL_SUMMARY_RASTERS <- TRUE
OVERWRITE_ANNUAL_SUMMARY_PLOTS <- TRUE
PREDICT_COMPLETE_CASES_ONLY <- TRUE
PREDICTION_SUMMARY_COLUMNS <- c("pred_min", "pred_max", "pred_mean", "pred_median")
DERIVED_SUMMARY_COLUMNS <- c("pred_min", "pred_median", "pred_mean", "pred_max")
ODDS_EPSILON <- 1e-6
TOP_PERCENTILE <- 0.99
ROR_BREAKPOINTS <- c(0.4, 0.7, 1.5, 5, 15)
CHANGE_RATIO_BREAKPOINTS <- c(0.6, 0.8, 1.2, 1.6, 2)
ROR_VALUE_COLORS <- c("green4", "lightgreen", "white", "red", "darkred", "purple4")
CHANGE_RATIO_VALUE_COLORS <- c("blue3", "lightblue", "white", "orange1", "red2", "red4")
TOP_PERCENTILE_COLORS <- c("lightgray", "red")
MAP_GRID_COLOR <- "gray88"
MAP_GRID_LWD <- 0.6
MAP_BORDER_COLOR <- "gray35"
MAP_BORDER_LWD <- 0.7
MAP_AXIS_CEX <- 0.9
MAP_LATITUDE_LIMITS <- c(-10, 10)
DERIVED_GGPLOT_PNG_WIDTH <- 14
DERIVED_GGPLOT_PNG_HEIGHT <- 15.5
DERIVED_GGPLOT_PNG_DPI <- 300
DERIVED_GGPLOT_BASE_SIZE <- 14
DERIVED_GGPLOT_LEGEND_KEY_HEIGHT_CM <- 1.5
DERIVED_GGPLOT_LEGEND_KEY_WIDTH_CM <- 1.0
DERIVED_COMBINED_GGPLOT_PNG_WIDTH <- 28
DERIVED_COMBINED_GGPLOT_PNG_HEIGHT <- 15.5
MEAN_DERIVED_GGPLOT_PNG_WIDTH <- 14
MEAN_DERIVED_GGPLOT_PNG_HEIGHT <- 9.5
MARGINAL_EFFECT_TOP_N <- 16
MARGINAL_EFFECT_GRID_SIZE <- 100
OVERWRITE_ROR_RASTERS <- TRUE
OVERWRITE_ROR_CHANGE_RASTERS <- TRUE
OVERWRITE_RAW_ODDS_CHANGE_RASTERS <- TRUE
OVERWRITE_PIXEL_RELATIVE_PREDICTION_RASTERS <- TRUE
OVERWRITE_PIXEL_RELATIVE_PREDICTION_PLOTS <- TRUE
OVERWRITE_MEAN_DERIVED_FIGURES <- TRUE
OVERWRITE_DERIVED_RASTER_PLOTS <- TRUE
OVERWRITE_RELATIVE_IMPORTANCE_OUTPUTS <- TRUE
OVERWRITE_MARGINAL_EFFECT_OUTPUTS <- TRUE
RUN_SHAP_CALCULATIONS <- TRUE
OVERWRITE_SHAP_OUTPUTS <- TRUE
SHAP_BATCH_SIZE <- 10000
# Use NULL for all fitted models. For a quick test run, set this to something
# like 1:3, then set it back to NULL for production outputs.
SHAP_MODEL_IDS <- NULL
SHAP_TOP_N_CHANGES <- 5
COUNTRY_NAME_COLUMN_CANDIDATES <- c(
  "country",
  "COUNTRY",
  "NAME",
  "Name",
  "name",
  "ADMIN",
  "admin",
  "SOVEREIGNT",
  "BRK_NAME",
  "CNTRY_NAME",
  "NAME_EN"
)

# The extracted point tables are already clipped to land. For Hansen-derived
# covariates, masked/null land pixels mean no mapped forest/no mapped loss, so
# they should be modeled as zero. Keep lagged loss structurally missing before
# the lagged Hansen target year exists.
HANSEN_ZERO_FILL_PREFIXES <- c(
  "forest_cover_prop_",
  "flsy_prop_",
  "frag_edge_prop_"
)
HANSEN_LAG_ZERO_FILL_RULES <- list(
  fl1yp_prop_ = 2002L,
  fl2yp_prop_ = 2003L
)

# PET is not always available for the newest prediction year. For the prediction
# grid, each location has repeated years, so we can fill the unavailable target
# year from the same grid cell's latest available year.
LATEST_AVAILABLE_COVARIATE_FILLS <- list(
  pet_mm_0_10km = c("2025" = 2024)
)

RASTER_WRITE_OPTIONS <- list(
  datatype = "FLT4S",
  gdal = c("COMPRESS=LZW")
)


#### Helpers ####

require_package <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(
      sprintf("Package '%s' is required. Install it before running this script.", package),
      call. = FALSE
    )
  }
}

make_dir <- function(path) {
  dir.create(path, showWarnings = FALSE, recursive = TRUE)
}

make_parent_dir <- function(path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
}

filter_training_dataset_by_type <- function(df) {
  if (length(ACTIVE_TRAINING_TYPE_FILTER) == 0) {
    return(df)
  }

  if (!"type" %in% names(df)) {
    stop("TRAINING_TYPE_FILTER was set, but the training dataset has no 'type' column.", call. = FALSE)
  }
  if (!OUTCOME_COLUMN %in% names(df)) {
    stop("Training dataset is missing the outcome column: ", OUTCOME_COLUMN, call. = FALSE)
  }

  event_rows <- !is.na(df[[OUTCOME_COLUMN]]) & df[[OUTCOME_COLUMN]] == EVENT_VALUE
  keep_rows <- !event_rows | df$type %in% ACTIVE_TRAINING_TYPE_FILTER

  before_rows <- nrow(df)
  before_events <- sum(event_rows, na.rm = TRUE)
  before_controls <- sum(df[[OUTCOME_COLUMN]] == CONTROL_VALUE, na.rm = TRUE)

  filtered <- df[keep_rows, , drop = FALSE]
  row.names(filtered) <- NULL

  after_events <- sum(filtered[[OUTCOME_COLUMN]] == EVENT_VALUE, na.rm = TRUE)
  after_controls <- sum(filtered[[OUTCOME_COLUMN]] == CONTROL_VALUE, na.rm = TRUE)

  if (after_events == 0) {
    stop(
      "TRAINING_TYPE_FILTER removed all event rows. Requested type value(s): ",
      paste(ACTIVE_TRAINING_TYPE_FILTER, collapse = ", "),
      call. = FALSE
    )
  }
  if (after_controls == 0) {
    stop("No control rows remain after applying TRAINING_TYPE_FILTER.", call. = FALSE)
  }

  message(
    "Training type filter retained event type(s): ",
    paste(ACTIVE_TRAINING_TYPE_FILTER, collapse = ", ")
  )
  message(
    "Rows retained after training type filter: ",
    format(nrow(filtered), big.mark = ","),
    " of ",
    format(before_rows, big.mark = ","),
    "; events retained: ",
    format(after_events, big.mark = ","),
    " of ",
    format(before_events, big.mark = ","),
    "; controls retained: ",
    format(after_controls, big.mark = ","),
    " of ",
    format(before_controls, big.mark = ",")
  )

  filtered
}

.MAP_CONTEXT_ENV <- new.env(parent = emptyenv())

load_africa_country_borders <- function() {
  if (isTRUE(.MAP_CONTEXT_ENV$borders_checked)) {
    return(.MAP_CONTEXT_ENV$borders)
  }

  if (!file.exists(AFRICA_COUNTRY_BORDER_FILE)) {
    warning("Africa country border shapefile was not found: ", AFRICA_COUNTRY_BORDER_FILE)
    .MAP_CONTEXT_ENV$borders <- NULL
    .MAP_CONTEXT_ENV$borders_checked <- TRUE
    return(NULL)
  }

  borders <- terra::vect(AFRICA_COUNTRY_BORDER_FILE)
  border_crs <- terra::crs(borders)
  if (!is.na(border_crs) && nzchar(border_crs)) {
    borders <- terra::project(borders, RASTER_CRS)
  } else {
    terra::crs(borders) <- RASTER_CRS
  }

  .MAP_CONTEXT_ENV$borders <- borders
  .MAP_CONTEXT_ENV$borders_checked <- TRUE
  borders
}

detect_country_name_column <- function(sf_object) {
  candidates <- COUNTRY_NAME_COLUMN_CANDIDATES[COUNTRY_NAME_COLUMN_CANDIDATES %in% names(sf_object)]
  if (length(candidates) > 0) {
    return(candidates[1])
  }

  non_geometry_columns <- setdiff(names(sf_object), attr(sf_object, "sf_column"))
  character_columns <- non_geometry_columns[vapply(sf_object[non_geometry_columns], is.character, logical(1))]
  if (length(character_columns) > 0) {
    return(character_columns[1])
  }

  stop("Could not identify a country-name column in ", AFRICA_COUNTRY_BORDER_FILE, call. = FALSE)
}

add_country_to_prediction_grid <- function(df) {
  if ("country" %in% names(df) && any(nzchar(as.character(df$country)))) {
    return(df)
  }
  if (!file.exists(AFRICA_COUNTRY_BORDER_FILE)) {
    warning("Africa country border shapefile was not found; prediction-grid country will be NA.")
    df$country <- NA_character_
    return(df)
  }
  if (!all(c("longitude", "latitude") %in% names(df))) {
    stop("Prediction grid needs longitude and latitude columns before joining country names.", call. = FALSE)
  }

  key_columns <- if ("grid_id" %in% names(df)) "grid_id" else c("longitude", "latitude")
  location_table <- df[!duplicated(df[key_columns]), c(key_columns, "longitude", "latitude"), drop = FALSE]
  location_table$.row_id <- seq_len(nrow(location_table))

  old_s2 <- sf::sf_use_s2(FALSE)
  on.exit(sf::sf_use_s2(old_s2), add = TRUE)

  countries <- sf::st_read(AFRICA_COUNTRY_BORDER_FILE, quiet = TRUE)
  country_column <- detect_country_name_column(countries)
  countries <- sf::st_transform(countries[, country_column, drop = FALSE], crs = RASTER_CRS)
  countries <- tryCatch(
    sf::st_make_valid(countries),
    error = function(exc) {
      warning("Could not repair country border geometry with st_make_valid(); using original geometry. Error: ", conditionMessage(exc))
      countries
    }
  )
  countries <- countries[!sf::st_is_empty(countries), , drop = FALSE]

  points <- sf::st_as_sf(
    location_table,
    coords = c("longitude", "latitude"),
    crs = RASTER_CRS,
    remove = FALSE
  )

  joined <- sf::st_join(points, countries, join = sf::st_within, left = TRUE)
  joined_table <- sf::st_drop_geometry(joined)
  joined_table <- joined_table[!duplicated(joined_table$.row_id), , drop = FALSE]
  country_values <- as.character(joined_table[[country_column]])

  if (all(is.na(country_values))) {
    joined <- sf::st_join(points, countries, join = sf::st_intersects, left = TRUE)
    joined_table <- sf::st_drop_geometry(joined)
    joined_table <- joined_table[!duplicated(joined_table$.row_id), , drop = FALSE]
    country_values <- as.character(joined_table[[country_column]])
  }

  country_lookup <- data.frame(
    joined_table[key_columns],
    country = country_values,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  country_lookup <- country_lookup[!duplicated(country_lookup[key_columns]), , drop = FALSE]

  if ("country" %in% names(df)) {
    df$country <- NULL
  }

  df$.original_row_order <- seq_len(nrow(df))
  df <- merge(df, country_lookup, by = key_columns, all.x = TRUE, sort = FALSE)
  df <- df[order(df$.original_row_order), , drop = FALSE]
  df$.original_row_order <- NULL
  row.names(df) <- NULL
  df
}

raster_extent_limits <- function(raster_layer) {
  extent <- terra::ext(raster_layer)
  extent_limits(extent)
}

extent_limits <- function(extent) {
  if (inherits(extent, "SpatExtent")) {
    extent_values <- c(
      terra::xmin(extent),
      terra::xmax(extent),
      terra::ymin(extent),
      terra::ymax(extent)
    )
  } else {
    extent_values <- suppressWarnings(unname(as.numeric(extent)))
    if (length(extent_values) < 4 || any(!is.finite(extent_values[1:4]))) {
      extent_values <- c(
        extent[1],
        extent[2],
        extent[3],
        extent[4]
      )
    }
  }

  extent_values <- unname(as.numeric(extent_values[1:4]))
  if (length(extent_values) < 4 || any(!is.finite(extent_values))) {
    stop("Could not read xmin/xmax/ymin/ymax from the plotting extent.", call. = FALSE)
  }

  stats::setNames(extent_values, c("xmin", "xmax", "ymin", "ymax"))
}

map_plot_extent <- function(raster_layer) {
  limits <- raster_extent_limits(raster_layer)
  ymin <- MAP_LATITUDE_LIMITS[1]
  ymax <- MAP_LATITUDE_LIMITS[2]

  if (
    !is.finite(ymin) ||
    !is.finite(ymax) ||
    ymin >= ymax ||
    ymax < limits[["ymin"]] ||
    ymin > limits[["ymax"]]
  ) {
    ymin <- limits[["ymin"]]
    ymax <- limits[["ymax"]]
  }

  terra::ext(limits[["xmin"]], limits[["xmax"]], ymin, ymax)
}

axis_ticks_within <- function(limits, n = 6) {
  ticks <- pretty(limits, n = n)
  ticks[ticks >= min(limits) & ticks <= max(limits)]
}

format_degree_labels <- function(values, positive_suffix, negative_suffix) {
  suffix <- ifelse(values > 0, positive_suffix, ifelse(values < 0, negative_suffix, ""))
  abs_values <- abs(values)
  labels <- ifelse(
    abs(abs_values - round(abs_values)) < 1e-8,
    as.character(round(abs_values)),
    trimws(format(round(abs_values, 2), scientific = FALSE, trim = TRUE))
  )
  paste0(labels, intToUtf8(176), suffix)
}

add_latlong_grid <- function(plot_extent) {
  limits <- extent_limits(plot_extent)
  x_ticks <- axis_ticks_within(c(limits[["xmin"]], limits[["xmax"]]))
  y_ticks <- axis_ticks_within(c(limits[["ymin"]], limits[["ymax"]]))

  graphics::abline(v = x_ticks, col = MAP_GRID_COLOR, lwd = MAP_GRID_LWD)
  graphics::abline(h = y_ticks, col = MAP_GRID_COLOR, lwd = MAP_GRID_LWD)
  graphics::axis(
    side = 1,
    at = x_ticks,
    labels = format_degree_labels(x_ticks, "E", "W"),
    cex.axis = MAP_AXIS_CEX,
    las = 1,
    tck = -0.015
  )
  graphics::axis(
    side = 2,
    at = y_ticks,
    labels = format_degree_labels(y_ticks, "N", "S"),
    cex.axis = MAP_AXIS_CEX,
    las = 1,
    tck = -0.015
  )
}

add_country_borders <- function(plot_extent) {
  borders <- load_africa_country_borders()
  if (is.null(borders)) {
    return(invisible(NULL))
  }

  limits <- extent_limits(plot_extent)
  borders_to_plot <- tryCatch(
    terra::crop(borders, plot_extent),
    error = function(e) borders
  )
  graphics::clip(limits[["xmin"]], limits[["xmax"]], limits[["ymin"]], limits[["ymax"]])
  terra::lines(borders_to_plot, col = MAP_BORDER_COLOR, lwd = MAP_BORDER_LWD)
  invisible(NULL)
}

add_static_map_context <- function(plot_extent) {
  add_latlong_grid(plot_extent)
  add_country_borders(plot_extent)
  graphics::box(col = MAP_BORDER_COLOR)
  invisible(NULL)
}

test_plot_extent_helpers <- function(year = PREDICTION_YEARS[1]) {
  raster_stack <- read_annual_summary_raster(year)
  plot_extent <- map_plot_extent(raster_stack[[DERIVED_SUMMARY_COLUMNS[1]]])
  extent_limits(plot_extent)
}

as_numeric_predictors <- function(df, predictor_names) {
  for (predictor in predictor_names) {
    df[[predictor]] <- suppressWarnings(as.numeric(df[[predictor]]))
    df[[predictor]][!is.finite(df[[predictor]])] <- NA_real_
  }
  df
}

fill_hansen_land_na_with_zero <- function(df) {
  if (!"year" %in% names(df)) {
    stop("Data needs a year column before applying Hansen zero-fill rules.", call. = FALSE)
  }

  fill_column <- function(data, column, rows) {
    fill_rows <- rows & is.na(data[[column]])
    data[[column]][fill_rows] <- 0
    if (any(fill_rows)) {
      message("Filled ", format(sum(fill_rows), big.mark = ","), " missing ", column, " values with 0.")
    }
    data
  }

  for (prefix in HANSEN_ZERO_FILL_PREFIXES) {
    columns <- grep(paste0("^", prefix), names(df), value = TRUE)
    for (column in columns) {
      df <- fill_column(df, column, rep(TRUE, nrow(df)))
    }
  }

  for (prefix in names(HANSEN_LAG_ZERO_FILL_RULES)) {
    min_valid_year <- HANSEN_LAG_ZERO_FILL_RULES[[prefix]]
    columns <- grep(paste0("^", prefix), names(df), value = TRUE)
    for (column in columns) {
      df <- fill_column(df, column, df$year >= min_valid_year)
    }
  }

  df
}

identify_predictors <- function(training_df, prediction_df) {
  training_predictors <- setdiff(names(training_df), TRAINING_BASE_COLUMNS)
  prediction_predictors <- setdiff(names(prediction_df), PREDICTION_BASE_COLUMNS)

  if (!identical(training_predictors, prediction_predictors)) {
    stop(
      "Training and prediction-grid covariate columns do not match.\n",
      "Missing from training: ",
      paste(setdiff(prediction_predictors, training_predictors), collapse = ", "),
      "\nExtra in training: ",
      paste(setdiff(training_predictors, prediction_predictors), collapse = ", "),
      call. = FALSE
    )
  }

  training_predictors
}

check_training_dataset <- function(df, predictor_names) {
  required <- c(TRAINING_BASE_COLUMNS, predictor_names)
  missing_required <- setdiff(required, names(df))
  if (length(missing_required) > 0) {
    stop("Training dataset is missing required columns: ", paste(missing_required, collapse = ", "), call. = FALSE)
  }

  outcomes <- sort(unique(df[[OUTCOME_COLUMN]]))
  if (!all(outcomes %in% c(CONTROL_VALUE, EVENT_VALUE))) {
    stop("Outcome column must contain only 0/1 values.", call. = FALSE)
  }

  blank_predictors <- predictor_names[vapply(df[predictor_names], function(x) all(is.na(x)), logical(1))]
  if (length(blank_predictors) > 0) {
    stop("Training dataset has all-blank predictors: ", paste(blank_predictors, collapse = ", "), call. = FALSE)
  }

  invisible(TRUE)
}

make_sampled_dataset <- function(df, controls_per_event, seed, replace_controls = FALSE) {
  set.seed(seed)

  event_rows <- which(df[[OUTCOME_COLUMN]] == EVENT_VALUE)
  control_rows <- which(df[[OUTCOME_COLUMN]] == CONTROL_VALUE)
  n_controls <- length(event_rows) * controls_per_event

  if (length(event_rows) == 0) {
    stop("No event rows are available for modeling.", call. = FALSE)
  }
  if (length(control_rows) == 0) {
    stop("No control rows are available for modeling.", call. = FALSE)
  }
  if (length(control_rows) < n_controls && !replace_controls) {
    warning("Not enough controls to sample without replacement; switching to replacement for this dataset.")
    replace_controls <- TRUE
  }

  sampled_controls <- sample(control_rows, n_controls, replace = replace_controls)
  sampled_rows <- c(event_rows, sampled_controls)
  sampled_df <- df[sampled_rows, , drop = FALSE]
  sampled_df <- sampled_df[sample(seq_len(nrow(sampled_df))), , drop = FALSE]
  row.names(sampled_df) <- NULL
  sampled_df
}

fit_brt_model <- function(df, predictor_names, model_id) {
  gbm_x <- match(predictor_names, names(df))
  gbm_y <- match(OUTCOME_COLUMN, names(df))

  args <- list(
    data = df,
    gbm.x = gbm_x,
    gbm.y = gbm_y,
    family = BRT_FAMILY,
    tree.complexity = TREE_COMPLEXITY,
    learning.rate = LEARNING_RATE,
    bag.fraction = BAG_FRACTION,
    n.folds = N_FOLDS
  )

  # Keep model output quiet while leaving modeling defaults otherwise unchanged.
  quiet_args <- list(verbose = FALSE, silent = TRUE, plot.main = FALSE, plot.folds = FALSE)
  valid_args <- names(formals(dismo::gbm.step))
  args <- c(args[names(args) %in% valid_args], quiet_args[names(quiet_args) %in% valid_args])

  set.seed(RANDOM_SEED + model_id)
  model <- do.call(dismo::gbm.step, args)
  if (is.null(model)) {
    stop(sprintf("BRT model %s returned NULL. Try lowering the learning rate less, or inspect sampled data.", model_id), call. = FALSE)
  }

  model
}

get_gbm_object <- function(model) {
  if (!is.null(model$model) && inherits(model$model, "gbm")) {
    return(model$model)
  }
  model
}

best_trees <- function(model) {
  if (!is.null(model$gbm.call$best.trees)) {
    return(model$gbm.call$best.trees)
  }
  if (!is.null(model$n.trees)) {
    return(model$n.trees)
  }
  stop("Could not determine the number of trees to use for prediction.", call. = FALSE)
}

predict_brt_probability <- function(model, data) {
  gbm_model <- get_gbm_object(model)
  stats::predict(
    gbm_model,
    newdata = as.data.frame(data),
    n.trees = best_trees(model),
    type = "response"
  )
}

model_prediction_column <- function(model_id) {
  sprintf("pred_model_%03d", model_id)
}

prediction_metadata_columns <- function(df) {
  required_first <- c("longitude", "latitude")
  optional_after <- c("x", "y", "grid_id", "grid_batch", "year", "country")
  metadata_columns <- c(required_first, optional_after[optional_after %in% names(df)])
  missing_required <- setdiff(required_first, names(df))
  if (length(missing_required) > 0) {
    stop("Prediction grid is missing required coordinate columns: ", paste(missing_required, collapse = ", "), call. = FALSE)
  }
  unique(metadata_columns)
}

make_model_prediction_table <- function(prediction_grid, predictor_names, model_list) {
  missing_prediction_predictors <- setdiff(predictor_names, names(prediction_grid))
  if (length(missing_prediction_predictors) > 0) {
    stop("Prediction grid is missing predictors: ", paste(missing_prediction_predictors, collapse = ", "), call. = FALSE)
  }

  metadata_columns <- prediction_metadata_columns(prediction_grid)
  prediction_grid <- as_numeric_predictors(prediction_grid, predictor_names)
  prediction_table <- prediction_grid[, metadata_columns, drop = FALSE]

  complete_rows <- stats::complete.cases(prediction_grid[, predictor_names, drop = FALSE])
  if (PREDICT_COMPLETE_CASES_ONLY) {
    message("Prediction rows with complete covariates: ", format(sum(complete_rows), big.mark = ","), " of ", format(nrow(prediction_grid), big.mark = ","))
  } else {
    complete_rows <- rep(TRUE, nrow(prediction_grid))
  }

  for (j in seq_along(model_list)) {
    prediction_column <- model_prediction_column(j)
    prediction_values <- rep(NA_real_, nrow(prediction_grid))
    message("  predicting ", prediction_column, "...")
    prediction_values[complete_rows] <- predict_brt_probability(
      model_list[[j]],
      prediction_grid[complete_rows, predictor_names, drop = FALSE]
    )
    prediction_table[[prediction_column]] <- prediction_values
  }

  prediction_table
}

gbm_object_for_best_trees <- function(model) {
  gbm_model <- get_gbm_object(model)
  n_best <- best_trees(model)

  if (!is.null(gbm_model$trees) && length(gbm_model$trees) > n_best) {
    gbm_model$trees <- gbm_model$trees[seq_len(n_best)]
    gbm_model$n.trees <- n_best
  }

  gbm_model
}

shap_mean_column <- function(predictor) {
  paste0("shap_mean_", predictor)
}

shap_delta_column <- function(predictor) {
  paste0("delta_shap_mean_", predictor)
}

selected_shap_model_ids <- function(model_list) {
  if (is.null(SHAP_MODEL_IDS)) {
    return(seq_along(model_list))
  }

  model_ids <- as.integer(SHAP_MODEL_IDS)
  invalid_ids <- model_ids[is.na(model_ids) | model_ids < 1 | model_ids > length(model_list)]
  if (length(invalid_ids) > 0) {
    stop("SHAP_MODEL_IDS contains invalid model ids: ", paste(invalid_ids, collapse = ", "), call. = FALSE)
  }

  unique(model_ids)
}

extract_treeshap_matrix <- function(shap_object, predictor_names) {
  shap_matrix <- as.data.frame(shap_object$shaps)
  missing_predictors <- setdiff(predictor_names, names(shap_matrix))
  if (length(missing_predictors) > 0) {
    stop("TreeSHAP output is missing predictors: ", paste(missing_predictors, collapse = ", "), call. = FALSE)
  }

  as.matrix(shap_matrix[, predictor_names, drop = FALSE])
}

display_covariate_name <- function(predictor) {
  scale_match <- regexpr("_[0-9]+_[0-9]+km$", predictor)
  scale <- if (scale_match[1] == -1) "" else regmatches(predictor, scale_match)
  scale_label <- ifelse(
    nzchar(scale),
    paste0(" (", gsub("_", "-", sub("^_", "", scale)), ")"),
    ""
  )
  base_name <- sub("_[0-9]+_[0-9]+km$", "", predictor)

  base_labels <- c(
    forest_cover_prop = "Forest cover",
    frag_edge_prop = "Forest edge",
    flsy_prop = "Forest loss, same yr",
    fl1yp_prop = "Forest loss, 1-yr lag",
    fl2yp_prop = "Forest loss, 2-yr lag",
    pop_density = "Population density",
    precip_mm = "Precipitation",
    precip_anom_mm = "Precip. anomaly",
    precip_z = "Precip. z-score",
    temp_c = "Temperature",
    temp_anom_c = "Temp. anomaly",
    temp_z = "Temp. z-score",
    pet_mm = "PET",
    ndvi = "NDVI",
    ndvi_anom = "NDVI anomaly",
    ndvi_z = "NDVI z-score",
    elevation_m = "Elevation"
  )

  label <- unname(base_labels[base_name])
  if (is.na(label)) {
    label <- tools::toTitleCase(gsub("_", " ", base_name))
  }

  paste0(label, scale_label)
}

calculate_mean_shap_table <- function(prediction_grid, predictor_names, model_list, dsl) {
  require_package("treeshap")

  missing_prediction_predictors <- setdiff(predictor_names, names(prediction_grid))
  if (length(missing_prediction_predictors) > 0) {
    stop("Prediction grid is missing predictors for SHAP: ", paste(missing_prediction_predictors, collapse = ", "), call. = FALSE)
  }

  prediction_grid <- as_numeric_predictors(prediction_grid, predictor_names)
  complete_rows <- stats::complete.cases(prediction_grid[, predictor_names, drop = FALSE])
  complete_row_ids <- which(complete_rows)
  if (length(complete_row_ids) == 0) {
    stop("No prediction-grid rows have complete covariates for SHAP.", call. = FALSE)
  }

  model_ids <- selected_shap_model_ids(model_list)
  if (length(model_ids) == 0) {
    stop("No SHAP model ids were selected.", call. = FALSE)
  }

  if (length(dsl) < max(model_ids)) {
    stop("The sampled dataset list does not contain all selected SHAP model ids.", call. = FALSE)
  }

  message(
    "Calculating TreeSHAP values for ",
    format(length(complete_row_ids), big.mark = ","),
    " complete prediction-grid rows across ",
    length(model_ids),
    " model fit(s)."
  )

  explanation_x <- as.data.frame(prediction_grid[complete_rows, predictor_names, drop = FALSE])
  shap_sum <- matrix(
    0,
    nrow = nrow(explanation_x),
    ncol = length(predictor_names),
    dimnames = list(NULL, predictor_names)
  )

  batch_starts <- seq(1, nrow(explanation_x), by = SHAP_BATCH_SIZE)
  for (model_index in seq_along(model_ids)) {
    model_id <- model_ids[model_index]
    message("  TreeSHAP model ", model_id, " of ", length(model_list), " (", model_index, "/", length(model_ids), ")...")

    reference_x <- as.data.frame(dsl[[model_id]][, predictor_names, drop = FALSE])
    reference_x <- reference_x[stats::complete.cases(reference_x), , drop = FALSE]
    if (nrow(reference_x) == 0) {
      stop("Model ", model_id, " has no complete reference rows for TreeSHAP.", call. = FALSE)
    }

    unified_model <- treeshap::unify(gbm_object_for_best_trees(model_list[[model_id]]), reference_x)

    for (batch_index in seq_along(batch_starts)) {
      batch_start <- batch_starts[batch_index]
      batch_end <- min(batch_start + SHAP_BATCH_SIZE - 1, nrow(explanation_x))
      batch_rows <- batch_start:batch_end

      shap_object <- treeshap::treeshap(
        unified_model,
        explanation_x[batch_rows, , drop = FALSE],
        interactions = FALSE,
        verbose = FALSE
      )
      shap_sum[batch_rows, ] <- shap_sum[batch_rows, ] + extract_treeshap_matrix(shap_object, predictor_names)

      if (batch_index %% 10 == 0 || batch_index == length(batch_starts)) {
        message("    batch ", batch_index, "/", length(batch_starts), " complete")
      }
    }

    rm(unified_model)
    gc(verbose = FALSE)
  }

  shap_mean <- shap_sum / length(model_ids)
  metadata_columns <- prediction_metadata_columns(prediction_grid)
  shap_table <- prediction_grid[, metadata_columns, drop = FALSE]
  shap_table$shap_model_count <- NA_integer_

  shap_columns <- shap_mean_column(predictor_names)
  for (column in shap_columns) {
    shap_table[[column]] <- NA_real_
  }

  shap_table$shap_model_count[complete_row_ids] <- length(model_ids)
  for (j in seq_along(predictor_names)) {
    shap_table[[shap_columns[j]]][complete_row_ids] <- shap_mean[, j]
  }

  shap_table
}

prediction_location_year_keys <- function(df) {
  if (all(c("grid_id", "year") %in% names(df))) {
    return(c("grid_id", "year"))
  }
  if (all(c("x", "y", "year") %in% names(df))) {
    return(c("x", "y", "year"))
  }
  stop("Could not identify prediction location-year keys.", call. = FALSE)
}

join_predictions_with_mean_shap <- function(prediction_summary_table, shap_table, predictor_names) {
  key_columns <- prediction_location_year_keys(prediction_summary_table)
  shap_columns <- c("shap_model_count", shap_mean_column(predictor_names))
  country_column <- if (!"country" %in% names(prediction_summary_table) && "country" %in% names(shap_table)) "country" else character(0)
  merge_columns <- c(key_columns, country_column, shap_columns[shap_columns %in% names(shap_table)])
  missing_columns <- setdiff(shap_columns, names(shap_table))
  if (length(missing_columns) > 0) {
    stop("SHAP table is missing expected columns: ", paste(missing_columns, collapse = ", "), call. = FALSE)
  }

  prediction_summary_table$.original_row_order <- seq_len(nrow(prediction_summary_table))
  out <- merge(
    prediction_summary_table,
    shap_table[, merge_columns, drop = FALSE],
    by = key_columns,
    all.x = TRUE,
    sort = FALSE
  )
  out <- out[order(out$.original_row_order), , drop = FALSE]
  out$.original_row_order <- NULL
  row.names(out) <- NULL
  out
}

top_shap_change_table <- function(delta_matrix, predictor_names, top_n) {
  top_n <- min(top_n, length(predictor_names))
  predictor_labels <- vapply(predictor_names, display_covariate_name, character(1))

  out <- data.frame(matrix(nrow = nrow(delta_matrix), ncol = 0))
  for (rank_id in seq_len(top_n)) {
    out[[sprintf("top_shap_change_%s_predictor", rank_id)]] <- NA_character_
    out[[sprintf("top_shap_change_%s_label", rank_id)]] <- NA_character_
    out[[sprintf("top_shap_change_%s_value", rank_id)]] <- NA_real_
  }

  for (i in seq_len(nrow(delta_matrix))) {
    values <- delta_matrix[i, ]
    finite <- is.finite(values)
    if (!any(finite)) {
      next
    }

    ordered <- order(abs(values[finite]), decreasing = TRUE)
    finite_predictors <- predictor_names[finite]
    finite_labels <- predictor_labels[finite]
    finite_values <- values[finite]
    selected <- ordered[seq_len(min(top_n, length(ordered)))]

    for (rank_id in seq_along(selected)) {
      out[[sprintf("top_shap_change_%s_predictor", rank_id)]][i] <- finite_predictors[selected[rank_id]]
      out[[sprintf("top_shap_change_%s_label", rank_id)]][i] <- finite_labels[selected[rank_id]]
      out[[sprintf("top_shap_change_%s_value", rank_id)]][i] <- finite_values[selected[rank_id]]
    }
  }

  out
}

build_shap_change_table <- function(prediction_with_shap_table, predictor_names) {
  key_columns <- if ("grid_id" %in% names(prediction_with_shap_table)) "grid_id" else c("x", "y")
  missing_key_columns <- setdiff(c(key_columns, "year"), names(prediction_with_shap_table))
  if (length(missing_key_columns) > 0) {
    stop("Prediction SHAP table is missing columns needed for year-to-year changes: ", paste(missing_key_columns, collapse = ", "), call. = FALSE)
  }

  shap_columns <- shap_mean_column(predictor_names)
  delta_columns <- shap_delta_column(predictor_names)
  missing_shap_columns <- setdiff(shap_columns, names(prediction_with_shap_table))
  if (length(missing_shap_columns) > 0) {
    stop("Prediction SHAP table is missing expected SHAP columns: ", paste(missing_shap_columns, collapse = ", "), call. = FALSE)
  }

  change_tables <- list()
  for (year in PREDICTION_YEARS[-1]) {
    previous_year <- year - 1L
    current <- prediction_with_shap_table[prediction_with_shap_table$year == year, , drop = FALSE]
    previous <- prediction_with_shap_table[prediction_with_shap_table$year == previous_year, , drop = FALSE]
    if (nrow(current) == 0 || nrow(previous) == 0) {
      warning("Skipping SHAP changes for ", year, ": current or previous year rows are absent.")
      next
    }

    previous_keep <- c(key_columns, "pred_mean", shap_columns)
    previous <- previous[, previous_keep, drop = FALSE]
    names(previous)[names(previous) == "pred_mean"] <- "pred_mean_previous"
    names(previous)[match(shap_columns, names(previous))] <- paste0(shap_columns, "_previous")

    current$.original_row_order <- seq_len(nrow(current))
    joined <- merge(current, previous, by = key_columns, all.x = TRUE, sort = FALSE)
    joined <- joined[order(joined$.original_row_order), , drop = FALSE]
    joined$.original_row_order <- NULL

    out_columns <- unique(c(
      "longitude",
      "latitude",
      "x",
      "y",
      "grid_id",
      "grid_batch",
      "country",
      "year",
      "pred_mean",
      "pred_mean_previous"
    ))
    out_columns <- out_columns[out_columns %in% names(joined)]
    out <- joined[, out_columns, drop = FALSE]
    out$previous_year <- previous_year
    out$pred_mean_change <- out$pred_mean - out$pred_mean_previous
    out$pred_mean_ratio <- out$pred_mean / out$pred_mean_previous
    out$pred_mean_ratio[!is.finite(out$pred_mean_ratio)] <- NA_real_

    delta_matrix <- matrix(NA_real_, nrow = nrow(joined), ncol = length(predictor_names))
    colnames(delta_matrix) <- predictor_names
    for (j in seq_along(predictor_names)) {
      delta_values <- joined[[shap_columns[j]]] - joined[[paste0(shap_columns[j], "_previous")]]
      delta_values[!is.finite(delta_values)] <- NA_real_
      delta_matrix[, j] <- delta_values
      out[[delta_columns[j]]] <- delta_values
    }

    out <- cbind(out, top_shap_change_table(delta_matrix, predictor_names, SHAP_TOP_N_CHANGES))
    change_tables[[as.character(year)]] <- out
  }

  if (length(change_tables) == 0) {
    stop("No SHAP change tables could be created.", call. = FALSE)
  }

  do.call(rbind, change_tables)
}

write_prediction_shap_outputs <- function(prediction_grid, model_prediction_table, predictor_names, model_list, dsl) {
  if (!RUN_SHAP_CALCULATIONS) {
    message("RUN_SHAP_CALCULATIONS is FALSE; skipping SHAP outputs.")
    return(invisible(NULL))
  }

  if (file.exists(SHAP_MEAN_TABLE_CSV) && !OVERWRITE_SHAP_OUTPUTS) {
    message("Mean SHAP table exists; loading cached file: ", SHAP_MEAN_TABLE_CSV)
    shap_table <- utils::read.csv(SHAP_MEAN_TABLE_CSV, stringsAsFactors = FALSE)
  } else {
    shap_table <- calculate_mean_shap_table(prediction_grid, predictor_names, model_list, dsl)
    make_parent_dir(SHAP_MEAN_TABLE_CSV)
    utils::write.csv(shap_table, SHAP_MEAN_TABLE_CSV, row.names = FALSE)
    message("Saved mean SHAP table: ", SHAP_MEAN_TABLE_CSV)
  }

  if (file.exists(SHAP_COMPLETE_TABLE_CSV) && !OVERWRITE_SHAP_OUTPUTS) {
    message("Prediction + SHAP table exists; loading cached file: ", SHAP_COMPLETE_TABLE_CSV)
    prediction_with_shap <- utils::read.csv(SHAP_COMPLETE_TABLE_CSV, stringsAsFactors = FALSE)
  } else {
    prediction_summary_table <- summarize_model_predictions(model_prediction_table)
    prediction_with_shap <- join_predictions_with_mean_shap(prediction_summary_table, shap_table, predictor_names)
    make_parent_dir(SHAP_COMPLETE_TABLE_CSV)
    utils::write.csv(prediction_with_shap, SHAP_COMPLETE_TABLE_CSV, row.names = FALSE)
    message("Saved prediction + mean SHAP table: ", SHAP_COMPLETE_TABLE_CSV)
  }

  if (file.exists(SHAP_CHANGE_TABLE_CSV) && !OVERWRITE_SHAP_OUTPUTS) {
    message("SHAP change table exists; keeping cached file: ", SHAP_CHANGE_TABLE_CSV)
  } else {
    shap_change_table <- build_shap_change_table(prediction_with_shap, predictor_names)
    make_parent_dir(SHAP_CHANGE_TABLE_CSV)
    utils::write.csv(shap_change_table, SHAP_CHANGE_TABLE_CSV, row.names = FALSE)
    message("Saved year-to-year SHAP change table: ", SHAP_CHANGE_TABLE_CSV)
  }

  invisible(TRUE)
}

check_performance_evaluation_settings <- function() {
  valid_methods <- c("maximize_sens_ppv_product", "top_percent", "fixed_probability")
  if (!EVALUATION_THRESHOLD_METHOD %in% valid_methods) {
    stop(
      "EVALUATION_THRESHOLD_METHOD must be one of: ",
      paste(valid_methods, collapse = ", "),
      call. = FALSE
    )
  }
  valid_scopes <- c("overall", "by_year")
  if (!EVALUATION_THRESHOLD_SCOPE %in% valid_scopes) {
    stop(
      "EVALUATION_THRESHOLD_SCOPE must be one of: ",
      paste(valid_scopes, collapse = ", "),
      call. = FALSE
    )
  }
  if (!is.numeric(EVALUATION_TOP_PROPORTION) || length(EVALUATION_TOP_PROPORTION) != 1) {
    stop("EVALUATION_TOP_PROPORTION must be a single numeric value.", call. = FALSE)
  }
  if (EVALUATION_TOP_PROPORTION <= 0 || EVALUATION_TOP_PROPORTION > 1) {
    stop("EVALUATION_TOP_PROPORTION must be > 0 and <= 1.", call. = FALSE)
  }
  if (!is.numeric(EVALUATION_FIXED_THRESHOLD) || length(EVALUATION_FIXED_THRESHOLD) != 1) {
    stop("EVALUATION_FIXED_THRESHOLD must be a single numeric value.", call. = FALSE)
  }
  if (EVALUATION_FIXED_THRESHOLD < 0 || EVALUATION_FIXED_THRESHOLD > 1) {
    stop("EVALUATION_FIXED_THRESHOLD must be between 0 and 1.", call. = FALSE)
  }

  invisible(TRUE)
}

threshold_metric_grid <- function(outcome, scores) {
  keep <- !is.na(outcome) & is.finite(scores)
  outcome <- as.integer(outcome[keep])
  scores <- as.numeric(scores[keep])
  n_pos <- sum(outcome == EVENT_VALUE)
  n_neg <- sum(outcome == CONTROL_VALUE)

  if (n_pos == 0 || n_neg == 0 || length(scores) == 0) {
    return(data.frame())
  }

  order_id <- order(scores, decreasing = TRUE)
  ordered_scores <- scores[order_id]
  ordered_outcome <- outcome[order_id]
  score_runs <- rle(ordered_scores)
  threshold_index <- cumsum(score_runs$lengths)

  tp <- cumsum(ordered_outcome == EVENT_VALUE)[threshold_index]
  fp <- cumsum(ordered_outcome == CONTROL_VALUE)[threshold_index]
  fn <- n_pos - tp
  tn <- n_neg - fp
  sensitivity <- tp / n_pos
  ppv <- tp / (tp + fp)
  specificity <- tn / n_neg
  npv <- tn / (tn + fn)
  f1 <- ifelse(sensitivity + ppv == 0, NA_real_, 2 * sensitivity * ppv / (sensitivity + ppv))

  data.frame(
    threshold = score_runs$values,
    predicted_positive_count = tp + fp,
    tp = tp,
    fp = fp,
    tn = tn,
    fn = fn,
    sensitivity = sensitivity,
    ppv = ppv,
    specificity = specificity,
    npv = npv,
    f1 = f1,
    sens_ppv_product = sensitivity * ppv,
    stringsAsFactors = FALSE
  )
}

select_max_sens_ppv_threshold <- function(outcome, scores) {
  grid <- threshold_metric_grid(outcome, scores)
  if (nrow(grid) == 0) {
    return(list(threshold = NA_real_, objective_value = NA_real_))
  }

  grid <- grid[is.finite(grid$sens_ppv_product), , drop = FALSE]
  if (nrow(grid) == 0) {
    return(list(threshold = NA_real_, objective_value = NA_real_))
  }

  grid <- grid[order(-grid$sens_ppv_product, -grid$f1, -grid$threshold), , drop = FALSE]
  best <- grid[1, , drop = FALSE]
  list(
    threshold = best$threshold,
    objective_value = best$sens_ppv_product
  )
}

classify_evaluation_scores <- function(scores, outcome = NULL) {
  scores <- as.numeric(scores)
  finite_scores <- is.finite(scores)
  predicted_positive <- rep(NA, length(scores))
  threshold <- NA_real_
  objective_value <- NA_real_

  if (!any(finite_scores)) {
    return(list(
      predicted_positive = predicted_positive,
      threshold = threshold,
      objective_value = objective_value
    ))
  }

  if (EVALUATION_THRESHOLD_METHOD == "fixed_probability") {
    threshold <- EVALUATION_FIXED_THRESHOLD
    predicted_positive[finite_scores] <- scores[finite_scores] >= threshold
  } else if (EVALUATION_THRESHOLD_METHOD == "top_percent") {
    n_positive <- max(1L, ceiling(sum(finite_scores) * EVALUATION_TOP_PROPORTION))
    ordered_scores <- sort(scores[finite_scores], decreasing = TRUE)
    threshold <- ordered_scores[n_positive]
    predicted_positive[finite_scores] <- scores[finite_scores] >= threshold
  } else if (EVALUATION_THRESHOLD_METHOD == "maximize_sens_ppv_product") {
    if (is.null(outcome)) {
      stop("Outcome values are required when optimizing the cutoff.", call. = FALSE)
    }
    selected <- select_max_sens_ppv_threshold(outcome, scores)
    threshold <- selected$threshold
    objective_value <- selected$objective_value
    predicted_positive[finite_scores] <- scores[finite_scores] >= threshold
  }

  list(
    predicted_positive = as.integer(predicted_positive),
    threshold = threshold,
    objective_value = objective_value
  )
}

classify_performance_evaluation_predictions <- function(point_predictions) {
  point_predictions$predicted_positive <- NA_integer_
  point_predictions$classification_threshold <- NA_real_
  point_predictions$threshold_objective <- if (EVALUATION_THRESHOLD_METHOD == "maximize_sens_ppv_product") {
    "sensitivity_x_ppv"
  } else {
    NA_character_
  }
  point_predictions$threshold_objective_value <- NA_real_
  point_predictions$threshold_method <- EVALUATION_THRESHOLD_METHOD
  point_predictions$threshold_scope <- EVALUATION_THRESHOLD_SCOPE
  point_predictions$threshold_top_proportion <- if (EVALUATION_THRESHOLD_METHOD == "top_percent") {
    EVALUATION_TOP_PROPORTION
  } else {
    NA_real_
  }
  point_predictions$threshold_fixed_probability <- if (EVALUATION_THRESHOLD_METHOD == "fixed_probability") {
    EVALUATION_FIXED_THRESHOLD
  } else {
    NA_real_
  }

  threshold_groups <- if (EVALUATION_THRESHOLD_SCOPE == "overall") {
    list(overall = seq_len(nrow(point_predictions)))
  } else {
    stats::setNames(
      lapply(sort(unique(point_predictions$year)), function(year) which(point_predictions$year == year)),
      sort(unique(point_predictions$year))
    )
  }

  for (group_name in names(threshold_groups)) {
    group_rows <- threshold_groups[[group_name]]
    classified <- classify_evaluation_scores(
      scores = point_predictions$predicted_probability[group_rows],
      outcome = point_predictions[[OUTCOME_COLUMN]][group_rows]
    )
    point_predictions$predicted_positive[group_rows] <- classified$predicted_positive
    point_predictions$classification_threshold[group_rows] <- classified$threshold
    point_predictions$threshold_objective_value[group_rows] <- classified$objective_value
    message(
      "Apparent-performance cutoff for ",
      EVALUATION_THRESHOLD_SCOPE,
      " threshold group ",
      group_name,
      ": ",
      signif(classified$threshold, 5)
    )
  }

  point_predictions
}

roc_auc_value <- function(outcome, score) {
  require_package("pROC")
  keep <- !is.na(outcome) & is.finite(score)
  outcome <- as.integer(outcome[keep])
  score <- as.numeric(score[keep])
  n_pos <- sum(outcome == EVENT_VALUE)
  n_neg <- sum(outcome == CONTROL_VALUE)

  if (n_pos == 0 || n_neg == 0) {
    return(NA_real_)
  }

  roc_object <- pROC::roc(
    response = outcome,
    predictor = score,
    levels = c(CONTROL_VALUE, EVENT_VALUE),
    direction = "<",
    quiet = TRUE
  )
  as.numeric(pROC::auc(roc_object))
}

pr_auc_value <- function(outcome, score) {
  require_package("PRROC")
  keep <- !is.na(outcome) & is.finite(score)
  outcome <- as.integer(outcome[keep])
  score <- as.numeric(score[keep])
  n_pos <- sum(outcome == EVENT_VALUE)
  n_neg <- sum(outcome == CONTROL_VALUE)

  if (n_pos == 0 || n_neg == 0 || length(score) == 0) {
    return(NA_real_)
  }

  pr_object <- PRROC::pr.curve(
    scores.class0 = score[outcome == EVENT_VALUE],
    scores.class1 = score[outcome == CONTROL_VALUE],
    curve = FALSE
  )
  as.numeric(pr_object$auc.integral)
}

roc_curve_table <- function(point_predictions) {
  require_package("pROC")
  keep <- !is.na(point_predictions[[OUTCOME_COLUMN]]) &
    is.finite(point_predictions$predicted_probability)
  outcome <- as.integer(point_predictions[[OUTCOME_COLUMN]][keep])
  score <- as.numeric(point_predictions$predicted_probability[keep])
  n_pos <- sum(outcome == EVENT_VALUE)
  n_neg <- sum(outcome == CONTROL_VALUE)

  if (n_pos == 0 || n_neg == 0) {
    return(list(curve = data.frame(), auc = NA_real_))
  }

  roc_object <- pROC::roc(
    response = outcome,
    predictor = score,
    levels = c(CONTROL_VALUE, EVENT_VALUE),
    direction = "<",
    quiet = TRUE
  )
  curve <- as.data.frame(pROC::coords(
    roc_object,
    x = "all",
    ret = c("threshold", "specificity", "sensitivity"),
    transpose = FALSE
  ))
  curve$fpr <- 1 - curve$specificity
  curve$tpr <- curve$sensitivity
  curve <- curve[is.finite(curve$fpr) & is.finite(curve$tpr), , drop = FALSE]
  curve <- curve[order(curve$fpr, curve$tpr), , drop = FALSE]
  row.names(curve) <- NULL

  list(curve = curve, auc = as.numeric(pROC::auc(roc_object)))
}

pr_curve_table <- function(point_predictions) {
  require_package("PRROC")
  keep <- !is.na(point_predictions[[OUTCOME_COLUMN]]) &
    is.finite(point_predictions$predicted_probability)
  outcome <- as.integer(point_predictions[[OUTCOME_COLUMN]][keep])
  score <- as.numeric(point_predictions$predicted_probability[keep])
  n_pos <- sum(outcome == EVENT_VALUE)
  n_neg <- sum(outcome == CONTROL_VALUE)

  if (n_pos == 0 || n_neg == 0 || length(score) == 0) {
    return(list(curve = data.frame(), auc = NA_real_))
  }

  pr_object <- PRROC::pr.curve(
    scores.class0 = score[outcome == EVENT_VALUE],
    scores.class1 = score[outcome == CONTROL_VALUE],
    curve = TRUE
  )
  curve <- as.data.frame(pr_object$curve)
  if (ncol(curve) >= 3) {
    names(curve)[1:3] <- c("recall", "precision", "threshold")
  } else {
    names(curve) <- paste0("curve_col_", seq_len(ncol(curve)))
  }
  curve <- curve[is.finite(curve$recall) & is.finite(curve$precision), , drop = FALSE]
  curve <- curve[order(curve$recall, curve$precision), , drop = FALSE]
  row.names(curve) <- NULL

  list(curve = curve, auc = as.numeric(pr_object$auc.integral))
}

plot_roc_curve <- function(curve_table, auc, output_png) {
  if (!RUN_PERFORMANCE_FIGURES) {
    return(invisible(output_png))
  }
  if (nrow(curve_table) == 0 || !is.finite(auc)) {
    warning("Skipping ROC curve plot: no finite ROC curve values were available.")
    return(invisible(output_png))
  }

  make_parent_dir(output_png)
  p <- ggplot2::ggplot(curve_table, ggplot2::aes(x = fpr, y = tpr)) +
    ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray55") +
    ggplot2::geom_path(color = "steelblue4", linewidth = 1.2) +
    ggplot2::coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    ggplot2::labs(
      title = sprintf("Apparent ROC Curve (AUC = %.3f)", auc),
      x = "False positive rate",
      y = "True positive rate"
    ) +
    ggplot2::theme_bw(base_size = 14) +
    ggplot2::theme(plot.title = ggplot2::element_text(face = "bold", hjust = 0.5))

  ggplot2::ggsave(
    filename = output_png,
    plot = p,
    width = 7,
    height = 6,
    dpi = 300,
    units = "in"
  )
  invisible(output_png)
}

plot_pr_curve <- function(curve_table, auc, output_png) {
  if (!RUN_PERFORMANCE_FIGURES) {
    return(invisible(output_png))
  }
  if (nrow(curve_table) == 0 || !is.finite(auc)) {
    warning("Skipping precision-recall curve plot: no finite PR curve values were available.")
    return(invisible(output_png))
  }

  make_parent_dir(output_png)
  p <- ggplot2::ggplot(curve_table, ggplot2::aes(x = recall, y = precision)) +
    ggplot2::geom_path(color = "firebrick3", linewidth = 1.2) +
    ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    ggplot2::labs(
      title = sprintf("Apparent Precision-Recall Curve (PR-AUC = %.3f)", auc),
      x = "Recall",
      y = "Precision"
    ) +
    ggplot2::theme_bw(base_size = 14) +
    ggplot2::theme(plot.title = ggplot2::element_text(face = "bold", hjust = 0.5))

  ggplot2::ggsave(
    filename = output_png,
    plot = p,
    width = 7,
    height = 6,
    dpi = 300,
    units = "in"
  )
  invisible(output_png)
}

write_performance_curve_outputs <- function(point_predictions) {
  if (!RUN_PERFORMANCE_FIGURES) {
    message("RUN_PERFORMANCE_FIGURES is FALSE; skipping ROC/PR curve outputs.")
    return(invisible(NULL))
  }

  roc_result <- roc_curve_table(point_predictions)
  pr_result <- pr_curve_table(point_predictions)

  make_parent_dir(PERFORMANCE_EVALUATION_ROC_CURVE_CSV)
  utils::write.csv(roc_result$curve, PERFORMANCE_EVALUATION_ROC_CURVE_CSV, row.names = FALSE)
  utils::write.csv(pr_result$curve, PERFORMANCE_EVALUATION_PR_CURVE_CSV, row.names = FALSE)
  plot_roc_curve(roc_result$curve, roc_result$auc, PERFORMANCE_EVALUATION_ROC_PNG)
  plot_pr_curve(pr_result$curve, pr_result$auc, PERFORMANCE_EVALUATION_PR_PNG)

  message("Saved ROC curve table: ", PERFORMANCE_EVALUATION_ROC_CURVE_CSV)
  message("Saved ROC curve figure: ", PERFORMANCE_EVALUATION_ROC_PNG)
  message("Saved PR curve table: ", PERFORMANCE_EVALUATION_PR_CURVE_CSV)
  message("Saved PR curve figure: ", PERFORMANCE_EVALUATION_PR_PNG)

  invisible(list(roc = roc_result, pr = pr_result))
}

safe_metric_ratio <- function(numerator, denominator) {
  if (is.na(denominator) || denominator == 0) {
    return(NA_real_)
  }
  numerator / denominator
}

caret_confusion_matrix_object <- function(df) {
  require_package("caret")
  keep <- !is.na(df[[OUTCOME_COLUMN]]) &
    !is.na(df$predicted_positive) &
    is.finite(df$predicted_probability)
  eval_df <- df[keep, , drop = FALSE]
  if (nrow(eval_df) == 0) {
    return(NULL)
  }

  observed <- factor(
    ifelse(as.integer(eval_df[[OUTCOME_COLUMN]]) == EVENT_VALUE, "Event", "No Event"),
    levels = c("No Event", "Event")
  )
  predicted <- factor(
    ifelse(as.integer(eval_df$predicted_positive) == 1L, "Event", "No Event"),
    levels = c("No Event", "Event")
  )

  caret::confusionMatrix(
    data = predicted,
    reference = observed,
    positive = "Event",
    mode = "everything"
  )
}

extract_caret_metric <- function(values, metric_name) {
  if (is.null(values) || !metric_name %in% names(values)) {
    return(NA_real_)
  }
  as.numeric(values[[metric_name]])
}

caret_metrics_table <- function(df, label, year = NA_integer_) {
  cm <- caret_confusion_matrix_object(df)
  if (is.null(cm)) {
    return(data.frame(
      evaluation = label,
      year = year,
      n_evaluated = 0L,
      accuracy = NA_real_,
      kappa = NA_real_,
      sensitivity = NA_real_,
      specificity = NA_real_,
      ppv = NA_real_,
      npv = NA_real_,
      precision = NA_real_,
      recall = NA_real_,
      f1 = NA_real_,
      balanced_accuracy = NA_real_,
      detection_rate = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  by_class <- cm$byClass
  if (is.matrix(by_class) || is.data.frame(by_class)) {
    by_class <- by_class[1, ]
  }

  data.frame(
    evaluation = label,
    year = year,
    n_evaluated = sum(!is.na(df[[OUTCOME_COLUMN]]) & !is.na(df$predicted_positive) & is.finite(df$predicted_probability)),
    accuracy = extract_caret_metric(cm$overall, "Accuracy"),
    kappa = extract_caret_metric(cm$overall, "Kappa"),
    sensitivity = extract_caret_metric(by_class, "Sensitivity"),
    specificity = extract_caret_metric(by_class, "Specificity"),
    ppv = extract_caret_metric(by_class, "Pos Pred Value"),
    npv = extract_caret_metric(by_class, "Neg Pred Value"),
    precision = extract_caret_metric(by_class, "Precision"),
    recall = extract_caret_metric(by_class, "Recall"),
    f1 = extract_caret_metric(by_class, "F1"),
    balanced_accuracy = extract_caret_metric(by_class, "Balanced Accuracy"),
    detection_rate = extract_caret_metric(by_class, "Detection Rate"),
    stringsAsFactors = FALSE
  )
}

print_caret_confusion_matrix <- function(df, label) {
  message("")
  message("Caret confusion matrix: ", label)
  cm <- caret_confusion_matrix_object(df)
  if (is.null(cm)) {
    message("No complete evaluation rows were available.")
  } else {
    print(cm)
  }
  invisible(cm)
}

write_caret_performance_outputs <- function(point_predictions) {
  by_year <- do.call(
    rbind,
    lapply(sort(unique(point_predictions$year)), function(year) {
      caret_metrics_table(
        point_predictions[point_predictions$year == year, , drop = FALSE],
        label = "by_year",
        year = as.integer(year)
      )
    })
  )
  row.names(by_year) <- NULL
  overall <- caret_metrics_table(point_predictions, label = "overall", year = NA_integer_)

  print_caret_confusion_matrix(point_predictions, "overall apparent 2020-2025 training-point performance")
  for (year in sort(unique(point_predictions$year))) {
    print_caret_confusion_matrix(
      point_predictions[point_predictions$year == year, , drop = FALSE],
      paste0("apparent training-point performance ", year)
    )
  }

  utils::write.csv(by_year, PERFORMANCE_EVALUATION_CARET_METRICS_BY_YEAR_CSV, row.names = FALSE)
  utils::write.csv(overall, PERFORMANCE_EVALUATION_CARET_METRICS_OVERALL_CSV, row.names = FALSE)
  message("Saved caret metrics by year: ", PERFORMANCE_EVALUATION_CARET_METRICS_BY_YEAR_CSV)
  message("Saved caret metrics overall: ", PERFORMANCE_EVALUATION_CARET_METRICS_OVERALL_CSV)

  invisible(list(by_year = by_year, overall = overall))
}

confusion_performance_metrics <- function(df, label, year = NA_integer_) {
  keep <- !is.na(df[[OUTCOME_COLUMN]]) &
    !is.na(df$predicted_positive) &
    is.finite(df$predicted_probability)
  eval_df <- df[keep, , drop = FALSE]
  outcome <- as.integer(eval_df[[OUTCOME_COLUMN]])
  predicted <- as.integer(eval_df$predicted_positive)

  tp <- sum(outcome == EVENT_VALUE & predicted == 1L)
  fp <- sum(outcome == CONTROL_VALUE & predicted == 1L)
  tn <- sum(outcome == CONTROL_VALUE & predicted == 0L)
  fn <- sum(outcome == EVENT_VALUE & predicted == 0L)
  sensitivity <- safe_metric_ratio(tp, tp + fn)
  ppv <- safe_metric_ratio(tp, tp + fp)
  sens_ppv_product <- if (is.na(sensitivity) || is.na(ppv)) NA_real_ else sensitivity * ppv

  data.frame(
    evaluation = label,
    year = year,
    n_rows = nrow(df),
    n_evaluated = nrow(eval_df),
    n_events = sum(outcome == EVENT_VALUE),
    n_controls = sum(outcome == CONTROL_VALUE),
    tp = tp,
    fp = fp,
    tn = tn,
    fn = fn,
    sensitivity = sensitivity,
    ppv = ppv,
    sens_ppv_product = sens_ppv_product,
    specificity = safe_metric_ratio(tn, tn + fp),
    npv = safe_metric_ratio(tn, tn + fn),
    f1 = if (is.na(sensitivity) || is.na(ppv) || sensitivity + ppv == 0) {
      NA_real_
    } else {
      2 * sensitivity * ppv / (sensitivity + ppv)
    },
    roc_auc = roc_auc_value(df[[OUTCOME_COLUMN]], df$predicted_probability),
    pr_auc = pr_auc_value(df[[OUTCOME_COLUMN]], df$predicted_probability),
    threshold_method = EVALUATION_THRESHOLD_METHOD,
    threshold_scope = EVALUATION_THRESHOLD_SCOPE,
    classification_threshold = if (length(unique(stats::na.omit(df$classification_threshold))) == 1) {
      unique(stats::na.omit(df$classification_threshold))
    } else {
      NA_real_
    },
    threshold_objective = if (EVALUATION_THRESHOLD_METHOD == "maximize_sens_ppv_product") "sensitivity_x_ppv" else NA_character_,
    threshold_objective_value = if (length(unique(stats::na.omit(df$threshold_objective_value))) == 1) {
      unique(stats::na.omit(df$threshold_objective_value))
    } else {
      NA_real_
    },
    top_proportion = if (EVALUATION_THRESHOLD_METHOD == "top_percent") EVALUATION_TOP_PROPORTION else NA_real_,
    fixed_threshold = if (EVALUATION_THRESHOLD_METHOD == "fixed_probability") EVALUATION_FIXED_THRESHOLD else NA_real_,
    stringsAsFactors = FALSE
  )
}

extract_performance_point_predictions <- function(training_df, evaluation_years) {
  check_performance_evaluation_settings()

  point_predictions <- list()
  for (year in evaluation_years) {
    year <- as.integer(year)
    raster_path <- annual_summary_raster_path(year)
    if (!file.exists(raster_path)) {
      stop(
        "Could not find prediction raster for evaluation year ",
        year,
        ": ",
        raster_path,
        "\nRun Section 8 first, or remove this year from EVALUATION_YEARS.",
        call. = FALSE
      )
    }

    eval_rows <- training_df[training_df$year == year, , drop = FALSE]
    if (nrow(eval_rows) == 0) {
      warning("No training dataset rows were found for evaluation year ", year, "; skipping.")
      next
    }

    if (!all(c("longitude", "latitude") %in% names(eval_rows))) {
      stop("Training dataset needs longitude and latitude columns for performance extraction.", call. = FALSE)
    }

    prediction_raster <- terra::rast(raster_path)
    if (!EVALUATION_PREDICTION_LAYER %in% names(prediction_raster)) {
      stop(
        "Prediction raster for ",
        year,
        " does not contain layer '",
        EVALUATION_PREDICTION_LAYER,
        "'. Available layers: ",
        paste(names(prediction_raster), collapse = ", "),
        call. = FALSE
      )
    }

    points <- terra::vect(eval_rows, geom = c("longitude", "latitude"), crs = RASTER_CRS)
    extracted <- terra::extract(prediction_raster[[EVALUATION_PREDICTION_LAYER]], points, ID = FALSE)

    keep_columns <- intersect(
      c("id", "year", "latitude", "longitude", OUTCOME_COLUMN, "type", "country"),
      names(eval_rows)
    )
    out <- eval_rows[, keep_columns, drop = FALSE]
    out$predicted_probability <- as.numeric(extracted[[1]])
    out$prediction_layer <- EVALUATION_PREDICTION_LAYER
    out$model_scope <- "all_years_model"
    out$prediction_raster <- raster_path

    point_predictions[[as.character(year)]] <- out
    message(
      "Extracted ",
      format(nrow(out), big.mark = ","),
      " apparent training-point predictions for ",
      year,
      " from ",
      basename(raster_path),
      "."
    )
  }

  if (length(point_predictions) == 0) {
    stop("No performance point predictions were created.", call. = FALSE)
  }

  point_predictions <- do.call(rbind, point_predictions)
  row.names(point_predictions) <- NULL
  classify_performance_evaluation_predictions(point_predictions)
}

summarize_performance <- function(point_predictions) {
  by_year <- do.call(
    rbind,
    lapply(sort(unique(point_predictions$year)), function(year) {
      confusion_performance_metrics(
        point_predictions[point_predictions$year == year, , drop = FALSE],
        label = "by_year",
        year = as.integer(year)
      )
    })
  )
  row.names(by_year) <- NULL

  overall <- confusion_performance_metrics(point_predictions, label = "overall", year = NA_integer_)
  list(by_year = by_year, overall = overall)
}

write_performance_outputs <- function(training_df, evaluation_years) {
  if (!RUN_PERFORMANCE_EVALUATION) {
    message("RUN_PERFORMANCE_EVALUATION is FALSE; skipping apparent performance evaluation.")
    return(invisible(NULL))
  }

  make_dir(PERFORMANCE_EVALUATION_DIR)
  point_predictions <- extract_performance_point_predictions(training_df, evaluation_years)
  metrics <- summarize_performance(point_predictions)
  caret_metrics <- write_caret_performance_outputs(point_predictions)
  performance_curves <- write_performance_curve_outputs(point_predictions)

  utils::write.csv(point_predictions, PERFORMANCE_EVALUATION_POINT_PREDICTIONS_CSV, row.names = FALSE)
  utils::write.csv(metrics$by_year, PERFORMANCE_EVALUATION_METRICS_BY_YEAR_CSV, row.names = FALSE)
  utils::write.csv(metrics$overall, PERFORMANCE_EVALUATION_METRICS_OVERALL_CSV, row.names = FALSE)

  message("Saved apparent training-point predictions: ", PERFORMANCE_EVALUATION_POINT_PREDICTIONS_CSV)
  message("Saved apparent performance metrics by year: ", PERFORMANCE_EVALUATION_METRICS_BY_YEAR_CSV)
  message("Saved apparent performance metrics overall: ", PERFORMANCE_EVALUATION_METRICS_OVERALL_CSV)

  invisible(list(
    point_predictions = point_predictions,
    metrics = metrics,
    caret_metrics = caret_metrics,
    performance_curves = performance_curves
  ))
}

fill_prediction_covariates_from_reference_years <- function(df, fill_rules) {
  if (length(fill_rules) == 0) {
    return(df)
  }
  if (!"year" %in% names(df)) {
    stop("Prediction grid needs a year column before applying covariate fill rules.", call. = FALSE)
  }

  key_columns <- if ("grid_id" %in% names(df)) {
    "grid_id"
  } else {
    c("x", "y")
  }
  missing_keys <- setdiff(key_columns, names(df))
  if (length(missing_keys) > 0) {
    stop("Prediction grid is missing key columns for covariate fills: ", paste(missing_keys, collapse = ", "), call. = FALSE)
  }

  make_key <- function(data) {
    if (length(key_columns) == 1) {
      return(as.character(data[[key_columns]]))
    }
    do.call(paste, c(data[key_columns], sep = "||"))
  }

  for (covariate in names(fill_rules)) {
    if (!covariate %in% names(df)) {
      warning("Skipping fill rule for missing covariate: ", covariate)
      next
    }

    for (target_year_name in names(fill_rules[[covariate]])) {
      target_year <- as.integer(target_year_name)
      reference_year <- as.integer(fill_rules[[covariate]][[target_year_name]])
      target_rows <- which(df$year == target_year)
      reference_rows <- which(df$year == reference_year)

      if (length(target_rows) == 0 || length(reference_rows) == 0) {
        warning(
          "Skipping fill rule for ", covariate, ": target year ", target_year,
          " or reference year ", reference_year, " is absent."
        )
        next
      }

      target_finite <- sum(is.finite(df[[covariate]][target_rows]))
      if (target_finite > 0) {
        message(
          "Fill rule not needed for ", covariate, " in ", target_year,
          ": already has ", format(target_finite, big.mark = ","), " finite values."
        )
        next
      }

      reference_lookup <- data.frame(
        key = make_key(df[reference_rows, , drop = FALSE]),
        value = df[[covariate]][reference_rows],
        stringsAsFactors = FALSE
      )
      reference_lookup <- reference_lookup[is.finite(reference_lookup$value), , drop = FALSE]
      reference_lookup <- reference_lookup[!duplicated(reference_lookup$key), , drop = FALSE]

      target_key <- make_key(df[target_rows, , drop = FALSE])
      matched_reference <- match(target_key, reference_lookup$key)
      fill_values <- reference_lookup$value[matched_reference]
      fillable <- is.na(df[[covariate]][target_rows]) & is.finite(fill_values)
      df[[covariate]][target_rows[fillable]] <- fill_values[fillable]

      message(
        "Filled ", format(sum(fillable), big.mark = ","), " ", covariate,
        " values for ", target_year, " from ", reference_year, "."
      )
    }
  }

  df
}

summarize_model_predictions <- function(prediction_table) {
  model_columns <- grep("^pred_model_[0-9]+$", names(prediction_table), value = TRUE)
  if (length(model_columns) == 0) {
    stop("No model prediction columns were found.", call. = FALSE)
  }

  prediction_matrix <- as.matrix(prediction_table[, model_columns, drop = FALSE])
  all_missing <- rowSums(!is.na(prediction_matrix)) == 0

  prediction_table$pred_min <- apply(prediction_matrix, 1, function(x) if (all(is.na(x))) NA_real_ else min(x, na.rm = TRUE))
  prediction_table$pred_max <- apply(prediction_matrix, 1, function(x) if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE))
  prediction_table$pred_mean <- rowMeans(prediction_matrix, na.rm = TRUE)
  prediction_table$pred_mean[all_missing] <- NA_real_
  prediction_table$pred_median <- apply(prediction_matrix, 1, function(x) if (all(is.na(x))) NA_real_ else stats::median(x, na.rm = TRUE))

  metadata_columns <- prediction_metadata_columns(prediction_table)
  prediction_table[, c(metadata_columns, model_columns, PREDICTION_SUMMARY_COLUMNS), drop = FALSE]
}

infer_grid_step <- function(values) {
  unique_values <- sort(unique(round(values, COORDINATE_ROUND_DIGITS)))
  diffs <- diff(unique_values)
  diffs <- diffs[diffs > 0]
  if (length(diffs) == 0) {
    stop("Could not infer raster grid resolution from prediction coordinates.", call. = FALSE)
  }
  stats::median(diffs)
}

make_prediction_template <- function(df_year) {
  x_values <- round(df_year$x, COORDINATE_ROUND_DIGITS)
  y_values <- round(df_year$y, COORDINATE_ROUND_DIGITS)
  x_res <- infer_grid_step(x_values)
  y_res <- infer_grid_step(y_values)
  x_unique <- sort(unique(x_values))
  y_unique <- sort(unique(y_values))

  terra::rast(
    ncols = length(x_unique),
    nrows = length(y_unique),
    xmin = min(x_unique) - x_res / 2,
    xmax = max(x_unique) + x_res / 2,
    ymin = min(y_unique) - y_res / 2,
    ymax = max(y_unique) + y_res / 2,
    crs = RASTER_CRS
  )
}

prediction_summary_to_raster_stack <- function(df_year) {
  df_year$x <- round(df_year$x, COORDINATE_ROUND_DIGITS)
  df_year$y <- round(df_year$y, COORDINATE_ROUND_DIGITS)

  template <- make_prediction_template(df_year)
  point_values <- df_year[, c("x", "y", PREDICTION_SUMMARY_COLUMNS), drop = FALSE]
  points <- terra::vect(point_values, geom = c("x", "y"), crs = RASTER_CRS)
  raster_stack <- terra::rasterize(points, template, field = PREDICTION_SUMMARY_COLUMNS, fun = "mean")
  names(raster_stack) <- PREDICTION_SUMMARY_COLUMNS
  raster_stack
}

annual_summary_raster_path <- function(year) {
  file.path(ANNUAL_SUMMARY_RASTER_DIR, sprintf("event_probability_summary_%s.tif", year))
}

annual_summary_plot_path <- function(year, summary_column) {
  file.path(ANNUAL_SUMMARY_PLOT_DIR, sprintf("event_probability_%s_%s.png", summary_column, year))
}

plot_annual_summary_rasters <- function(raster_stack, year) {
  for (summary_column in PREDICTION_SUMMARY_COLUMNS) {
    output_png <- annual_summary_plot_path(year, summary_column)
    if (file.exists(output_png) && !OVERWRITE_ANNUAL_SUMMARY_PLOTS) {
      next
    }

    raster_layer <- raster_stack[[summary_column]]
    raster_values <- terra::values(raster_layer, mat = FALSE)
    if (!any(is.finite(raster_values))) {
      warning("Skipping plot for ", summary_column, " in ", year, ": raster layer has no finite values.")
      next
    }

    plot_extent <- map_plot_extent(raster_layer)
    make_parent_dir(output_png)
    grDevices::png(output_png, width = 1600, height = 1000, res = 150)
    terra::plot(
      raster_layer,
      main = sprintf("Event Probability %s %s", sub("^pred_", "", summary_column), year),
      col = grDevices::hcl.colors(100, "Viridis"),
      axes = FALSE,
      ext = plot_extent
    )
    add_static_map_context(plot_extent)
    grDevices::dev.off()
  }
}

summary_measure_name <- function(summary_column) {
  sub("^pred_", "", summary_column)
}

stack_rasters <- function(raster_list) {
  if (length(raster_list) == 0) {
    stop("No rasters were supplied for stacking.", call. = FALSE)
  }

  raster_stack <- raster_list[[1]]
  if (length(raster_list) > 1) {
    for (i in 2:length(raster_list)) {
      raster_stack <- c(raster_stack, raster_list[[i]])
    }
  }
  raster_stack
}

probability_scalar_to_odds <- function(probability) {
  if (!is.finite(probability)) {
    return(NA_real_)
  }

  bounded_probability <- max(min(probability, 1 - ODDS_EPSILON), ODDS_EPSILON)
  bounded_probability / (1 - bounded_probability)
}

probability_to_odds_raster <- function(probability_raster) {
  bounded_probability <- terra::ifel(
    is.na(probability_raster),
    NA_real_,
    terra::ifel(
      probability_raster < ODDS_EPSILON,
      ODDS_EPSILON,
      terra::ifel(probability_raster > 1 - ODDS_EPSILON, 1 - ODDS_EPSILON, probability_raster)
    )
  )
  bounded_probability / (1 - bounded_probability)
}

raster_quantile <- function(raster_layer, probability = TOP_PERCENTILE) {
  quantile_value <- terra::global(
    raster_layer,
    fun = function(values, ...) {
      values <- values[is.finite(values)]
      if (length(values) == 0) {
        return(NA_real_)
      }
      as.numeric(stats::quantile(values, probs = probability, na.rm = TRUE, names = FALSE))
    }
  )[1, 1]

  as.numeric(quantile_value)
}

top_percentile_binary_raster <- function(raster_layer, layer_name) {
  threshold <- raster_quantile(raster_layer, TOP_PERCENTILE)
  if (!is.finite(threshold)) {
    warning("Could not calculate the top-percentile threshold for ", layer_name, ". Returning an all-NA layer.")
    out <- raster_layer * NA_real_
  } else {
    out <- terra::ifel(
      is.na(raster_layer),
      NA_real_,
      terra::ifel(raster_layer >= threshold, 1, 0)
    )
  }
  names(out) <- layer_name
  out
}

read_annual_summary_raster <- function(year) {
  raster_path <- annual_summary_raster_path(year)
  if (!file.exists(raster_path)) {
    stop("Annual summary raster does not exist: ", raster_path, call. = FALSE)
  }

  raster_stack <- terra::rast(raster_path)
  missing_layers <- setdiff(DERIVED_SUMMARY_COLUMNS, names(raster_stack))
  if (length(missing_layers) > 0) {
    stop(
      "Annual summary raster for ", year, " is missing layers: ",
      paste(missing_layers, collapse = ", "),
      call. = FALSE
    )
  }

  raster_stack
}

make_annual_odds_raster_stack <- function(probability_raster) {
  odds_layers <- lapply(DERIVED_SUMMARY_COLUMNS, function(summary_column) {
    odds_layer <- probability_to_odds_raster(probability_raster[[summary_column]])
    names(odds_layer) <- paste0("odds_", summary_measure_name(summary_column))
    odds_layer
  })
  stack_rasters(odds_layers)
}

make_ror_raster_stack <- function(probability_raster, year) {
  ror_layers <- list()

  for (summary_column in DERIVED_SUMMARY_COLUMNS) {
    measure <- summary_measure_name(summary_column)
    probability_layer <- probability_raster[[summary_column]]
    area_probability <- as.numeric(terra::global(probability_layer, "mean", na.rm = TRUE)[1, 1])
    area_odds <- probability_scalar_to_odds(area_probability)
    if (!is.finite(area_odds) || area_odds <= 0) {
      stop("Could not calculate study-area odds for ", measure, " in ", year, ".", call. = FALSE)
    }

    ror_name <- paste0("ROR_", measure)
    ror_layer <- probability_to_odds_raster(probability_layer) / area_odds
    names(ror_layer) <- ror_name
    ror_layers[[ror_name]] <- ror_layer

    top_name <- paste0(ror_name, "_top1pct")
    ror_layers[[top_name]] <- top_percentile_binary_raster(ror_layer, top_name)
  }

  stack_rasters(ror_layers)
}

make_ratio_raster_stack <- function(numerator_stack, denominator_stack, input_prefix, output_prefix) {
  ratio_layers <- list()

  for (summary_column in DERIVED_SUMMARY_COLUMNS) {
    measure <- summary_measure_name(summary_column)
    input_name <- paste0(input_prefix, measure)
    if (!all(input_name %in% names(numerator_stack)) || !all(input_name %in% names(denominator_stack))) {
      stop("Missing ratio input layer: ", input_name, call. = FALSE)
    }

    ratio_name <- paste0(output_prefix, measure)
    ratio_layer <- numerator_stack[[input_name]] / denominator_stack[[input_name]]
    ratio_layer <- terra::ifel(is.na(ratio_layer), NA_real_, ratio_layer)
    names(ratio_layer) <- ratio_name
    ratio_layers[[ratio_name]] <- ratio_layer

    top_name <- paste0(ratio_name, "_top1pct")
    ratio_layers[[top_name]] <- top_percentile_binary_raster(ratio_layer, top_name)
  }

  stack_rasters(ratio_layers)
}

make_pixel_relative_prediction_mean_layers <- function(annual_rasters) {
  mean_layers <- list()

  for (summary_column in DERIVED_SUMMARY_COLUMNS) {
    measure <- summary_measure_name(summary_column)
    year_stack <- stack_rasters(lapply(annual_rasters, function(raster_stack) raster_stack[[summary_column]]))
    mean_layer <- terra::app(year_stack, fun = "mean", na.rm = TRUE)
    mean_layer <- terra::ifel(is.na(mean_layer), NA_real_, mean_layer)
    names(mean_layer) <- paste0("mean_", measure)
    mean_layers[[summary_column]] <- mean_layer
  }

  mean_layers
}

make_pixel_relative_prediction_raster_stack <- function(probability_raster, pixel_mean_layers) {
  ratio_layers <- list()

  for (summary_column in DERIVED_SUMMARY_COLUMNS) {
    measure <- summary_measure_name(summary_column)
    numerator <- probability_raster[[summary_column]]
    denominator <- pixel_mean_layers[[summary_column]]
    ratio_name <- paste0("relPred_", measure)
    ratio_layer <- terra::ifel(
      is.na(numerator),
      NA_real_,
      terra::ifel(
        is.na(denominator),
        NA_real_,
        terra::ifel(denominator <= 0, NA_real_, numerator / denominator)
      )
    )
    names(ratio_layer) <- ratio_name
    ratio_layers[[ratio_name]] <- ratio_layer
  }

  stack_rasters(ratio_layers)
}

write_raster_stack <- function(raster_stack, output_path, overwrite) {
  if (file.exists(output_path) && !overwrite) {
    message("Raster exists; loading cached file: ", output_path)
    return(terra::rast(output_path))
  }

  make_parent_dir(output_path)
  terra::writeRaster(
    raster_stack,
    output_path,
    overwrite = overwrite,
    wopt = RASTER_WRITE_OPTIONS
  )
  message("  Wrote: ", output_path)
  terra::rast(output_path)
}

value_plot_kind <- function(layer_name) {
  if (grepl("^ROR_", layer_name)) {
    return("ror")
  }
  if (grepl("^chgROR_1yr_|^chgOdds_1yr_", layer_name)) {
    return("change")
  }
  if (grepl("^relPred_", layer_name)) {
    return("change")
  }
  "generic"
}

value_plot_palette <- function(plot_kind, n_colors) {
  if (plot_kind == "ror") {
    return(grDevices::colorRampPalette(ROR_VALUE_COLORS)(n_colors))
  }
  if (plot_kind == "change") {
    return(grDevices::colorRampPalette(CHANGE_RATIO_VALUE_COLORS)(n_colors))
  }
  grDevices::hcl.colors(n_colors, "Viridis")
}

value_plot_breaks_from_values <- function(values, plot_kind) {
  finite_values <- values[is.finite(values)]
  if (length(finite_values) == 0) {
    return(c(0, 1))
  }

  min_value <- min(finite_values, na.rm = TRUE)
  max_value <- max(finite_values, na.rm = TRUE)
  if (!is.finite(min_value) || !is.finite(max_value)) {
    return(c(0, 1))
  }
  if (min_value == max_value) {
    pad <- max(abs(min_value) * 0.05, 1e-6)
    return(c(min_value - pad, max_value + pad))
  }

  internal_breaks <- switch(
    plot_kind,
    ror = ROR_BREAKPOINTS,
    change = CHANGE_RATIO_BREAKPOINTS,
    pretty(finite_values, n = 6)
  )
  breaks <- sort(unique(c(
    min_value,
    internal_breaks[internal_breaks > min_value & internal_breaks < max_value],
    max_value
  )))

  if (length(breaks) < 2) {
    pad <- max(abs(min_value) * 0.05, 1e-6)
    breaks <- c(min_value - pad, max_value + pad)
  }
  breaks
}

value_plot_breaks <- function(raster_layer, plot_kind) {
  value_plot_breaks_from_values(terra::values(raster_layer, mat = FALSE), plot_kind)
}

value_plot_breaks_for_layers <- function(raster_stack, layer_names, plot_kind) {
  values <- unlist(
    lapply(layer_names, function(layer_name) terra::values(raster_stack[[layer_name]], mat = FALSE)),
    use.names = FALSE
  )
  value_plot_breaks_from_values(values, plot_kind)
}

pretty_derived_layer_label <- function(layer_name) {
  label <- layer_name
  label <- sub("_top1pct$", "", label)
  label <- sub("^ROR_", "ROR ", label)
  label <- sub("^chgROR_1yr_", "1-year ROR ratio ", label)
  label <- sub("^chgOdds_1yr_", "1-year odds ratio ", label)
  label <- sub("^relPred_", "pixel-relative prediction ratio ", label)
  label <- gsub("_", " ", label)
  label
}

pretty_legend_value <- function(value) {
  formatted <- format(signif(value, 3), scientific = FALSE, trim = TRUE)
  sub("\\.?0+$", "", formatted)
}

discrete_break_labels <- function(breaks) {
  if (length(breaks) < 2) {
    return(character(0))
  }

  labels <- character(length(breaks) - 1)
  for (i in seq_along(labels)) {
    lower <- pretty_legend_value(breaks[i])
    upper <- pretty_legend_value(breaks[i + 1])
    if (i == 1) {
      labels[i] <- paste0("<", upper)
    } else if (i == length(labels)) {
      labels[i] <- paste0(lower, "+")
    } else {
      labels[i] <- paste0(lower, "-", upper)
    }
  }
  labels
}

value_legend_title <- function(plot_kind) {
  switch(
    plot_kind,
    ror = "ROR",
    change = "Ratio",
    "Value"
  )
}

raster_layers_to_long_df <- function(raster_stack, layer_names, plot_extent) {
  raster_subset <- raster_stack[[layer_names]]
  raster_subset <- tryCatch(
    terra::crop(raster_subset, plot_extent, snap = "out"),
    error = function(e) raster_subset
  )

  raster_df <- terra::as.data.frame(raster_subset, xy = TRUE, na.rm = FALSE)
  names(raster_df)[1:2] <- c("x", "y")

  layer_labels <- pretty_derived_layer_label(layer_names)
  out <- do.call(
    rbind,
    lapply(seq_along(layer_names), function(i) {
      layer_name <- layer_names[i]
      data.frame(
        x = raster_df$x,
        y = raster_df$y,
        layer = layer_labels[i],
        value = raster_df[[layer_name]],
        stringsAsFactors = FALSE
      )
    })
  )

  out <- out[is.finite(out$x) & is.finite(out$y) & is.finite(out$value), , drop = FALSE]
  out$layer <- factor(out$layer, levels = layer_labels)
  out
}

ggplot_country_borders <- function(plot_extent) {
  borders <- load_africa_country_borders()
  if (is.null(borders)) {
    return(NULL)
  }

  borders_to_plot <- tryCatch(
    terra::crop(borders, plot_extent),
    error = function(e) borders
  )
  if (terra::nrow(borders_to_plot) == 0) {
    return(NULL)
  }

  sf::st_as_sf(borders_to_plot)
}

ggplot_axis_breaks <- function(plot_extent) {
  limits <- extent_limits(plot_extent)
  list(
    x = axis_ticks_within(c(limits[["xmin"]], limits[["xmax"]])),
    y = axis_ticks_within(c(limits[["ymin"]], limits[["ymax"]])),
    limits = limits
  )
}

ggplot_map_grid_layers <- function(axis_breaks) {
  list(
    ggplot2::geom_vline(
      xintercept = axis_breaks$x,
      color = MAP_GRID_COLOR,
      linewidth = 0.2
    ),
    ggplot2::geom_hline(
      yintercept = axis_breaks$y,
      color = MAP_GRID_COLOR,
      linewidth = 0.2
    )
  )
}

ggplot_map_context_layers <- function(plot_extent) {
  axis_breaks <- ggplot_axis_breaks(plot_extent)
  borders <- ggplot_country_borders(plot_extent)
  layers <- ggplot_map_grid_layers(axis_breaks)

  if (!is.null(borders)) {
    layers <- c(
      layers,
      list(
        ggplot2::geom_sf(
          data = borders,
          inherit.aes = FALSE,
          fill = NA,
          color = MAP_BORDER_COLOR,
          linewidth = 0.25
        )
      )
    )
  }

  layers
}

derived_map_theme <- function() {
  ggplot2::theme_bw(base_size = DERIVED_GGPLOT_BASE_SIZE) +
    ggplot2::theme(
      axis.title = ggplot2::element_blank(),
      legend.position = "right",
      legend.title = ggplot2::element_text(size = DERIVED_GGPLOT_BASE_SIZE + 1),
      legend.text = ggplot2::element_text(size = DERIVED_GGPLOT_BASE_SIZE),
      legend.key.height = grid::unit(DERIVED_GGPLOT_LEGEND_KEY_HEIGHT_CM, "cm"),
      legend.key.width = grid::unit(DERIVED_GGPLOT_LEGEND_KEY_WIDTH_CM, "cm"),
      panel.grid = ggplot2::element_blank(),
      panel.background = ggplot2::element_rect(fill = "white", color = NA),
      strip.background = ggplot2::element_rect(fill = "white", color = "gray70"),
      strip.text = ggplot2::element_text(size = DERIVED_GGPLOT_BASE_SIZE, face = "bold"),
      plot.title = ggplot2::element_text(
        size = DERIVED_GGPLOT_BASE_SIZE + 2,
        face = "bold",
        hjust = 0.5
      ),
      plot.margin = ggplot2::margin(6, 10, 6, 6)
    )
}

make_ggplot_map_base <- function(plot_df, plot_extent, title) {
  axis_breaks <- ggplot_axis_breaks(plot_extent)
  limits <- axis_breaks$limits

  ggplot2::ggplot(plot_df, ggplot2::aes(x = x, y = y)) +
    ggplot2::scale_x_continuous(
      breaks = axis_breaks$x,
      labels = format_degree_labels(axis_breaks$x, "E", "W"),
      expand = c(0, 0)
    ) +
    ggplot2::scale_y_continuous(
      breaks = axis_breaks$y,
      labels = format_degree_labels(axis_breaks$y, "N", "S"),
      expand = c(0, 0)
    ) +
    ggplot2::coord_sf(
      xlim = c(limits[["xmin"]], limits[["xmax"]]),
      ylim = c(limits[["ymin"]], limits[["ymax"]]),
      expand = FALSE,
      crs = sf::st_crs(4326)
    ) +
    ggplot2::labs(title = title) +
    derived_map_theme()
}

top_percentile_plot_path <- function(output_png) {
  if (grepl("\\.png$", output_png, ignore.case = TRUE)) {
    return(sub("\\.png$", "_top1pct.png", output_png, ignore.case = TRUE))
  }
  paste0(output_png, "_top1pct.png")
}

combined_derived_plot_path <- function(output_png) {
  if (grepl("\\.png$", output_png, ignore.case = TRUE)) {
    return(sub("\\.png$", "_combined.png", output_png, ignore.case = TRUE))
  }
  paste0(output_png, "_combined.png")
}

mean_derived_figure_path <- function(output_png) {
  output_name <- basename(output_png)
  if (grepl("\\.png$", output_name, ignore.case = TRUE)) {
    output_name <- sub("\\.png$", "_mean_plus_top1pct.png", output_name, ignore.case = TRUE)
  } else {
    output_name <- paste0(output_name, "_mean_plus_top1pct.png")
  }

  file.path(MEAN_DERIVED_FIGURE_DIR, output_name)
}

mean_value_layer_name <- function(raster_stack) {
  value_layers <- names(raster_stack)[!grepl("_top1pct$", names(raster_stack))]
  mean_layers <- value_layers[grepl("(^|_)mean$", value_layers)]
  if (length(mean_layers) == 0) {
    return(NA_character_)
  }
  if (length(mean_layers) > 1) {
    warning("Multiple mean layers found. Using the first: ", mean_layers[1])
  }
  mean_layers[1]
}

make_single_value_raster_plot <- function(raster_stack, value_layer, title) {
  plot_kind <- value_plot_kind(value_layer)
  breaks <- value_plot_breaks(raster_stack[[value_layer]], plot_kind)
  labels <- discrete_break_labels(breaks)
  colors <- value_plot_palette(plot_kind, length(labels))
  plot_extent <- map_plot_extent(raster_stack[[value_layer]])
  plot_df <- raster_layers_to_long_df(raster_stack, value_layer, plot_extent)

  if (nrow(plot_df) == 0 || length(labels) == 0) {
    warning("Skipping mean actual-value plot for ", title, ": no finite values were found.")
    return(NULL)
  }

  plot_df$value_class <- cut(
    plot_df$value,
    breaks = breaks,
    labels = labels,
    include.lowest = TRUE
  )
  plot_df <- plot_df[!is.na(plot_df$value_class), , drop = FALSE]

  make_ggplot_map_base(plot_df, plot_extent, title) +
    ggplot2::geom_raster(ggplot2::aes(fill = value_class)) +
    ggplot_map_context_layers(plot_extent) +
    ggplot2::scale_fill_manual(
      values = stats::setNames(colors, labels),
      breaks = labels,
      drop = FALSE,
      name = value_legend_title(plot_kind),
      guide = ggplot2::guide_legend(
        reverse = TRUE,
        keyheight = grid::unit(DERIVED_GGPLOT_LEGEND_KEY_HEIGHT_CM, "cm"),
        keywidth = grid::unit(DERIVED_GGPLOT_LEGEND_KEY_WIDTH_CM, "cm")
      )
    )
}

make_single_top_percentile_raster_plot <- function(raster_stack, value_layer, title) {
  top_layer <- paste0(value_layer, "_top1pct")
  if (top_layer %in% names(raster_stack)) {
    top_raster <- raster_stack[[top_layer]]
  } else {
    top_raster <- top_percentile_binary_raster(raster_stack[[value_layer]], top_layer)
  }

  plot_extent <- map_plot_extent(top_raster)
  plot_df <- raster_layers_to_long_df(top_raster, names(top_raster), plot_extent)
  if (nrow(plot_df) == 0) {
    warning("Skipping mean top-percentile plot for ", title, ": no finite values were found.")
    return(NULL)
  }

  plot_df$top1pct <- factor(ifelse(plot_df$value >= 1, "Yes", "No"), levels = c("No", "Yes"))

  make_ggplot_map_base(plot_df, plot_extent, title) +
    ggplot2::geom_raster(ggplot2::aes(fill = top1pct)) +
    ggplot_map_context_layers(plot_extent) +
    ggplot2::scale_fill_manual(
      values = c(No = TOP_PERCENTILE_COLORS[1], Yes = TOP_PERCENTILE_COLORS[2]),
      breaks = c("Yes", "No"),
      drop = FALSE,
      name = "Top 1%",
      guide = ggplot2::guide_legend(
        keyheight = grid::unit(DERIVED_GGPLOT_LEGEND_KEY_HEIGHT_CM, "cm"),
        keywidth = grid::unit(DERIVED_GGPLOT_LEGEND_KEY_WIDTH_CM, "cm")
      )
    )
}

plot_mean_value_and_top_percentile_figure <- function(raster_stack, output_png, overwrite, title) {
  mean_layer <- mean_value_layer_name(raster_stack)
  mean_output_png <- mean_derived_figure_path(output_png)
  if (file.exists(mean_output_png) && !overwrite) {
    return(invisible(mean_output_png))
  }
  if (is.na(mean_layer)) {
    warning("Skipping mean-only figure for ", title, ": no mean layer was found.")
    return(invisible(mean_output_png))
  }

  value_plot <- make_single_value_raster_plot(
    raster_stack = raster_stack,
    value_layer = mean_layer,
    title = pretty_derived_layer_label(mean_layer)
  )
  top_plot <- make_single_top_percentile_raster_plot(
    raster_stack = raster_stack,
    value_layer = mean_layer,
    title = paste(pretty_derived_layer_label(mean_layer), "top 1%")
  )
  if (is.null(value_plot) || is.null(top_plot)) {
    return(invisible(mean_output_png))
  }

  make_parent_dir(mean_output_png)
  grDevices::png(
    filename = mean_output_png,
    width = MEAN_DERIVED_GGPLOT_PNG_WIDTH,
    height = MEAN_DERIVED_GGPLOT_PNG_HEIGHT,
    units = "in",
    res = DERIVED_GGPLOT_PNG_DPI
  )
  on.exit(grDevices::dev.off(), add = TRUE)

  grid::grid.newpage()
  layout <- grid::grid.layout(
    nrow = 3,
    ncol = 1,
    heights = grid::unit.c(
      grid::unit(0.45, "in"),
      grid::unit(1, "null"),
      grid::unit(1, "null")
    )
  )
  grid::pushViewport(grid::viewport(layout = layout))
  grid::grid.text(
    title,
    vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 1),
    gp = grid::gpar(fontsize = DERIVED_GGPLOT_BASE_SIZE + 4, fontface = "bold")
  )
  print(
    value_plot + ggplot2::labs(title = "Mean value"),
    vp = grid::viewport(layout.pos.row = 2, layout.pos.col = 1)
  )
  print(
    top_plot + ggplot2::labs(title = "Top 1% of mean value"),
    vp = grid::viewport(layout.pos.row = 3, layout.pos.col = 1)
  )
  grid::popViewport()

  invisible(mean_output_png)
}

make_value_raster_facet_plot <- function(raster_stack, value_layers, title) {
  plot_kind <- value_plot_kind(value_layers[1])
  breaks <- value_plot_breaks_for_layers(raster_stack, value_layers, plot_kind)
  labels <- discrete_break_labels(breaks)
  colors <- value_plot_palette(plot_kind, length(labels))
  plot_extent <- map_plot_extent(raster_stack[[value_layers[1]]])
  plot_df <- raster_layers_to_long_df(raster_stack, value_layers, plot_extent)

  if (nrow(plot_df) == 0 || length(labels) == 0) {
    warning("Skipping actual-value plot for ", title, ": no finite values were found.")
    return(NULL)
  }

  plot_df$value_class <- cut(
    plot_df$value,
    breaks = breaks,
    labels = labels,
    include.lowest = TRUE
  )
  plot_df <- plot_df[!is.na(plot_df$value_class), , drop = FALSE]

  make_ggplot_map_base(plot_df, plot_extent, title) +
    ggplot2::geom_raster(ggplot2::aes(fill = value_class)) +
    ggplot_map_context_layers(plot_extent) +
    ggplot2::facet_wrap(~layer, ncol = 1) +
    ggplot2::scale_fill_manual(
      values = stats::setNames(colors, labels),
      breaks = labels,
      drop = FALSE,
      name = value_legend_title(plot_kind),
      guide = ggplot2::guide_legend(
        reverse = TRUE,
        keyheight = grid::unit(DERIVED_GGPLOT_LEGEND_KEY_HEIGHT_CM, "cm"),
        keywidth = grid::unit(DERIVED_GGPLOT_LEGEND_KEY_WIDTH_CM, "cm")
      )
    )
}

plot_value_raster_facets <- function(raster_stack, value_layers, output_png, overwrite, title) {
  if (file.exists(output_png) && !overwrite) {
    return(invisible(list(path = output_png, plot = NULL)))
  }

  p <- make_value_raster_facet_plot(raster_stack, value_layers, title)
  if (is.null(p)) {
    return(invisible(list(path = output_png, plot = NULL)))
  }

  make_parent_dir(output_png)
  ggplot2::ggsave(
    filename = output_png,
    plot = p,
    width = DERIVED_GGPLOT_PNG_WIDTH,
    height = DERIVED_GGPLOT_PNG_HEIGHT,
    dpi = DERIVED_GGPLOT_PNG_DPI,
    units = "in",
    limitsize = FALSE
  )
  invisible(list(path = output_png, plot = p))
}

make_top_percentile_raster_facet_plot <- function(raster_stack, value_layers, title) {
  top_layers <- paste0(value_layers, "_top1pct")
  top_layers <- top_layers[top_layers %in% names(raster_stack)]
  if (length(top_layers) == 0) {
    warning("Skipping top-percentile plot for ", title, ": no top-percentile layers were found.")
    return(NULL)
  }

  plot_extent <- map_plot_extent(raster_stack[[top_layers[1]]])
  plot_df <- raster_layers_to_long_df(raster_stack, top_layers, plot_extent)
  if (nrow(plot_df) == 0) {
    warning("Skipping top-percentile plot for ", title, ": no finite values were found.")
    return(NULL)
  }

  plot_df$top1pct <- factor(ifelse(plot_df$value >= 1, "Yes", "No"), levels = c("No", "Yes"))

  make_ggplot_map_base(plot_df, plot_extent, paste(title, "Top 1%")) +
    ggplot2::geom_raster(ggplot2::aes(fill = top1pct)) +
    ggplot_map_context_layers(plot_extent) +
    ggplot2::facet_wrap(~layer, ncol = 1) +
    ggplot2::scale_fill_manual(
      values = c(No = TOP_PERCENTILE_COLORS[1], Yes = TOP_PERCENTILE_COLORS[2]),
      breaks = c("Yes", "No"),
      drop = FALSE,
      name = "Top 1%",
      guide = ggplot2::guide_legend(
        keyheight = grid::unit(DERIVED_GGPLOT_LEGEND_KEY_HEIGHT_CM, "cm"),
        keywidth = grid::unit(DERIVED_GGPLOT_LEGEND_KEY_WIDTH_CM, "cm")
      )
    )
}

plot_top_percentile_raster_facets <- function(raster_stack, value_layers, output_png, overwrite, title) {
  top_output_png <- top_percentile_plot_path(output_png)
  if (file.exists(top_output_png) && !overwrite) {
    return(invisible(list(path = top_output_png, plot = NULL)))
  }

  p <- make_top_percentile_raster_facet_plot(raster_stack, value_layers, title)
  if (is.null(p)) {
    return(invisible(list(path = top_output_png, plot = NULL)))
  }

  make_parent_dir(top_output_png)
  ggplot2::ggsave(
    filename = top_output_png,
    plot = p,
    width = DERIVED_GGPLOT_PNG_WIDTH,
    height = DERIVED_GGPLOT_PNG_HEIGHT,
    dpi = DERIVED_GGPLOT_PNG_DPI,
    units = "in",
    limitsize = FALSE
  )
  invisible(list(path = top_output_png, plot = p))
}

plot_combined_value_and_top_percentile_facets <- function(value_plot, top_plot, output_png, overwrite, title) {
  combined_output_png <- combined_derived_plot_path(output_png)
  if (file.exists(combined_output_png) && !overwrite) {
    return(invisible(combined_output_png))
  }
  if (is.null(value_plot) || is.null(top_plot)) {
    return(invisible(combined_output_png))
  }

  make_parent_dir(combined_output_png)
  grDevices::png(
    filename = combined_output_png,
    width = DERIVED_COMBINED_GGPLOT_PNG_WIDTH,
    height = DERIVED_COMBINED_GGPLOT_PNG_HEIGHT,
    units = "in",
    res = DERIVED_GGPLOT_PNG_DPI
  )
  on.exit(grDevices::dev.off(), add = TRUE)

  grid::grid.newpage()
  layout <- grid::grid.layout(
    nrow = 2,
    ncol = 2,
    heights = grid::unit.c(grid::unit(0.45, "in"), grid::unit(1, "null")),
    widths = grid::unit(c(1, 1), "null")
  )
  grid::pushViewport(grid::viewport(layout = layout))
  grid::grid.text(
    title,
    vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 1:2),
    gp = grid::gpar(fontsize = DERIVED_GGPLOT_BASE_SIZE + 4, fontface = "bold")
  )
  print(
    value_plot + ggplot2::labs(title = "Actual values"),
    vp = grid::viewport(layout.pos.row = 2, layout.pos.col = 1)
  )
  print(
    top_plot + ggplot2::labs(title = "Top 1%"),
    vp = grid::viewport(layout.pos.row = 2, layout.pos.col = 2)
  )
  grid::popViewport()

  invisible(combined_output_png)
}

plot_multilayer_raster_stack <- function(raster_stack, output_png, overwrite, title) {
  finite_layer <- vapply(
    seq_len(terra::nlyr(raster_stack)),
    function(i) any(is.finite(terra::values(raster_stack[[i]], mat = FALSE))),
    logical(1)
  )
  if (!any(finite_layer)) {
    warning("Skipping plots for ", title, ": raster stack has no finite values.")
    return(invisible(output_png))
  }

  value_layers <- names(raster_stack)[!grepl("_top1pct$", names(raster_stack))]
  if (length(value_layers) == 0) {
    warning("Skipping plots for ", title, ": no actual-value layers were found.")
    return(invisible(output_png))
  }

  value_plot <- plot_value_raster_facets(
    raster_stack = raster_stack,
    value_layers = value_layers,
    output_png = output_png,
    overwrite = overwrite,
    title = title
  )
  top_plot <- plot_top_percentile_raster_facets(
    raster_stack = raster_stack,
    value_layers = value_layers,
    output_png = output_png,
    overwrite = overwrite,
    title = title
  )
  combined_plot <- plot_combined_value_and_top_percentile_facets(
    value_plot = value_plot$plot,
    top_plot = top_plot$plot,
    output_png = output_png,
    overwrite = overwrite,
    title = title
  )
  mean_plot <- plot_mean_value_and_top_percentile_figure(
    raster_stack = raster_stack,
    output_png = output_png,
    overwrite = OVERWRITE_MEAN_DERIVED_FIGURES,
    title = title
  )

  invisible(c(
    value_plot = value_plot$path,
    top_percentile_plot = top_plot$path,
    combined_plot = combined_plot,
    mean_plot = mean_plot
  ))
}

ror_raster_path <- function(year) {
  file.path(ROR_ESTIMATE_DIR, sprintf("event_probability_ROR_%s.tif", year))
}

ror_plot_path <- function(year) {
  file.path(ROR_ESTIMATE_PLOT_DIR, sprintf("event_probability_ROR_%s.png", year))
}

ror_change_raster_path <- function(year) {
  file.path(ROR_CHANGE_DIR, sprintf("chgROR_1yr_%s_over_%s.tif", year, year - 1))
}

ror_change_plot_path <- function(year) {
  file.path(ROR_CHANGE_PLOT_DIR, sprintf("chgROR_1yr_%s_over_%s.png", year, year - 1))
}

raw_odds_change_raster_path <- function(year) {
  file.path(RAW_ODDS_CHANGE_DIR, sprintf("chgOdds_1yr_%s_over_%s.tif", year, year - 1))
}

raw_odds_change_plot_path <- function(year) {
  file.path(RAW_ODDS_CHANGE_PLOT_DIR, sprintf("chgOdds_1yr_%s_over_%s.png", year, year - 1))
}

pixel_relative_prediction_raster_path <- function(year) {
  file.path(PIXEL_RELATIVE_PREDICTION_DIR, sprintf("event_probability_pixel_relative_ratio_%s.tif", year))
}

pixel_relative_prediction_plot_path <- function(year) {
  file.path(PIXEL_RELATIVE_PREDICTION_PLOT_DIR, sprintf("event_probability_pixel_relative_ratio_%s.png", year))
}

build_ror_outputs <- function(years) {
  ror_rasters <- list()
  for (year in years) {
    message("Calculating ROR raster for ", year, "...")
    ror_stack <- make_ror_raster_stack(read_annual_summary_raster(year), year)
    ror_rasters[[as.character(year)]] <- write_raster_stack(
      ror_stack,
      ror_raster_path(year),
      overwrite = OVERWRITE_ROR_RASTERS
    )
    plot_multilayer_raster_stack(
      ror_rasters[[as.character(year)]],
      ror_plot_path(year),
      overwrite = OVERWRITE_DERIVED_RASTER_PLOTS,
      title = sprintf("Relative Odds Ratio %s", year)
    )
  }

  invisible(ror_rasters)
}

build_ror_change_outputs <- function(years) {
  for (year in years[-1]) {
    previous_year <- year - 1
    message("Calculating 1-year ROR ratio for ", year, "/", previous_year, "...")
    ror_stack <- make_ratio_raster_stack(
      numerator_stack = terra::rast(ror_raster_path(year)),
      denominator_stack = terra::rast(ror_raster_path(previous_year)),
      input_prefix = "ROR_",
      output_prefix = "chgROR_1yr_"
    )
    ror_stack <- write_raster_stack(
      ror_stack,
      ror_change_raster_path(year),
      overwrite = OVERWRITE_ROR_CHANGE_RASTERS
    )
    plot_multilayer_raster_stack(
      ror_stack,
      ror_change_plot_path(year),
      overwrite = OVERWRITE_DERIVED_RASTER_PLOTS,
      title = sprintf("1-Year ROR Ratio %s/%s", year, previous_year)
    )
  }

  invisible(TRUE)
}

build_raw_odds_change_outputs <- function(years) {
  for (year in years[-1]) {
    previous_year <- year - 1
    message("Calculating 1-year raw odds ratio for ", year, "/", previous_year, "...")
    odds_stack <- make_ratio_raster_stack(
      numerator_stack = make_annual_odds_raster_stack(read_annual_summary_raster(year)),
      denominator_stack = make_annual_odds_raster_stack(read_annual_summary_raster(previous_year)),
      input_prefix = "odds_",
      output_prefix = "chgOdds_1yr_"
    )
    odds_stack <- write_raster_stack(
      odds_stack,
      raw_odds_change_raster_path(year),
      overwrite = OVERWRITE_RAW_ODDS_CHANGE_RASTERS
    )
    plot_multilayer_raster_stack(
      odds_stack,
      raw_odds_change_plot_path(year),
      overwrite = OVERWRITE_DERIVED_RASTER_PLOTS,
      title = sprintf("1-Year Raw Odds Ratio %s/%s", year, previous_year)
    )
  }

  invisible(TRUE)
}

build_pixel_relative_prediction_outputs <- function(years) {
  years <- sort(unique(years))
  annual_rasters <- stats::setNames(
    lapply(years, read_annual_summary_raster),
    as.character(years)
  )
  pixel_mean_layers <- make_pixel_relative_prediction_mean_layers(annual_rasters)

  for (year in years) {
    message("Calculating pixel-relative prediction ratio for ", year, "...")
    relative_stack <- make_pixel_relative_prediction_raster_stack(
      annual_rasters[[as.character(year)]],
      pixel_mean_layers
    )
    relative_stack <- write_raster_stack(
      relative_stack,
      pixel_relative_prediction_raster_path(year),
      overwrite = OVERWRITE_PIXEL_RELATIVE_PREDICTION_RASTERS
    )
    plot_value_raster_facets(
      raster_stack = relative_stack,
      value_layers = names(relative_stack),
      output_png = pixel_relative_prediction_plot_path(year),
      overwrite = OVERWRITE_PIXEL_RELATIVE_PREDICTION_PLOTS,
      title = sprintf("Pixel-Relative Event Probability Ratio %s", year)
    )
    plot_mean_value_and_top_percentile_figure(
      raster_stack = relative_stack,
      output_png = pixel_relative_prediction_plot_path(year),
      overwrite = OVERWRITE_MEAN_DERIVED_FIGURES,
      title = sprintf("Pixel-Relative Event Probability Ratio %s", year)
    )
  }

  invisible(TRUE)
}

relative_importance_csv_path <- function() {
  file.path(MODEL_DIAGNOSTIC_DIR, "brt_relative_importance_by_model.csv")
}

relative_importance_summary_csv_path <- function() {
  file.path(MODEL_DIAGNOSTIC_DIR, "brt_relative_importance_summary.csv")
}

relative_importance_plot_path <- function() {
  file.path(MODEL_DIAGNOSTIC_PLOT_DIR, "brt_relative_importance_boxplot.png")
}

extract_relative_importance <- function(model_list, predictor_names) {
  importance_rows <- lapply(seq_along(model_list), function(model_id) {
    importance <- as.data.frame(summary(get_gbm_object(model_list[[model_id]]), plotit = FALSE))
    if (!all(c("var", "rel.inf") %in% names(importance))) {
      stop("Relative-importance output from model ", model_id, " did not contain var and rel.inf columns.", call. = FALSE)
    }

    rel_inf <- stats::setNames(rep(0, length(predictor_names)), predictor_names)
    matched_predictors <- match(as.character(importance$var), predictor_names)
    keep <- !is.na(matched_predictors)
    rel_inf[matched_predictors[keep]] <- as.numeric(importance$rel.inf[keep])

    data.frame(
      model_id = model_id,
      predictor = predictor_names,
      rel_inf = as.numeric(rel_inf),
      stringsAsFactors = FALSE
    )
  })

  do.call(rbind, importance_rows)
}

summarize_relative_importance <- function(importance_long) {
  predictors <- unique(importance_long$predictor)
  importance_summary <- data.frame(
    predictor = predictors,
    median = vapply(predictors, function(predictor) stats::median(importance_long$rel_inf[importance_long$predictor == predictor], na.rm = TRUE), numeric(1)),
    mean = vapply(predictors, function(predictor) mean(importance_long$rel_inf[importance_long$predictor == predictor], na.rm = TRUE), numeric(1)),
    min = vapply(predictors, function(predictor) min(importance_long$rel_inf[importance_long$predictor == predictor], na.rm = TRUE), numeric(1)),
    max = vapply(predictors, function(predictor) max(importance_long$rel_inf[importance_long$predictor == predictor], na.rm = TRUE), numeric(1)),
    stringsAsFactors = FALSE
  )

  importance_summary <- importance_summary[order(-importance_summary$median, importance_summary$predictor), , drop = FALSE]
  row.names(importance_summary) <- NULL
  importance_summary
}

plot_relative_importance_boxplot <- function(importance_long, importance_summary) {
  output_png <- relative_importance_plot_path()
  if (file.exists(output_png) && !OVERWRITE_RELATIVE_IMPORTANCE_OUTPUTS) {
    return(invisible(output_png))
  }

  make_parent_dir(output_png)
  importance_long$predictor <- factor(importance_long$predictor, levels = importance_summary$predictor)
  grDevices::png(output_png, width = 2600, height = 1400, res = 150)
  on.exit(grDevices::dev.off(), add = TRUE)
  graphics::par(mar = c(12, 5, 4, 1))
  graphics::boxplot(
    rel_inf ~ predictor,
    data = importance_long,
    las = 2,
    cex.axis = 0.7,
    outline = FALSE,
    col = "gray85",
    border = "gray30",
    ylab = "Relative Importance (%)",
    main = sprintf("BRT Relative Importance Across %s Model Fits", length(unique(importance_long$model_id)))
  )
  invisible(output_png)
}

write_relative_importance_outputs <- function(model_list, predictor_names) {
  if (
    file.exists(relative_importance_csv_path()) &&
    file.exists(relative_importance_summary_csv_path()) &&
    !OVERWRITE_RELATIVE_IMPORTANCE_OUTPUTS
  ) {
    importance_long <- utils::read.csv(relative_importance_csv_path(), stringsAsFactors = FALSE)
    importance_summary <- utils::read.csv(relative_importance_summary_csv_path(), stringsAsFactors = FALSE)
  } else {
    message("Calculating BRT relative importance across ", length(model_list), " model fits...")
    importance_long <- extract_relative_importance(model_list, predictor_names)
    importance_summary <- summarize_relative_importance(importance_long)
    make_parent_dir(relative_importance_csv_path())
    make_parent_dir(relative_importance_summary_csv_path())
    utils::write.csv(importance_long, relative_importance_csv_path(), row.names = FALSE)
    utils::write.csv(importance_summary, relative_importance_summary_csv_path(), row.names = FALSE)
    message("Saved relative-importance tables.")
  }

  plot_relative_importance_boxplot(importance_long, importance_summary)
  importance_summary
}

marginal_effect_curves_csv_path <- function() {
  file.path(MODEL_DIAGNOSTIC_DIR, "brt_marginal_effect_curves_top16.csv")
}

marginal_effect_plot_path <- function() {
  file.path(MODEL_DIAGNOSTIC_PLOT_DIR, "brt_marginal_effect_curves_top16.png")
}

partial_curve_from_model <- function(model, predictor) {
  gbm_model <- get_gbm_object(model)
  i_var <- if (!is.null(gbm_model$var.names) && predictor %in% gbm_model$var.names) {
    predictor
  } else {
    match(predictor, gbm_model$var.names)
  }

  if (length(i_var) != 1 || is.na(i_var)) {
    stop("Predictor is not present in the fitted GBM object: ", predictor, call. = FALSE)
  }

  curve <- as.data.frame(plot(
    gbm_model,
    i.var = i_var,
    n.trees = best_trees(model),
    return.grid = TRUE
  ))

  y_column <- if ("y" %in% names(curve)) "y" else names(curve)[ncol(curve)]
  x_candidates <- setdiff(names(curve), y_column)
  if (length(x_candidates) == 0) {
    stop("Could not identify the x column for marginal effect predictor ", predictor, ".", call. = FALSE)
  }

  curve <- data.frame(
    x = suppressWarnings(as.numeric(curve[[x_candidates[1]]])),
    y = suppressWarnings(as.numeric(curve[[y_column]]))
  )
  curve <- curve[is.finite(curve$x) & is.finite(curve$y), , drop = FALSE]
  curve[order(curve$x), , drop = FALSE]
}

interpolate_partial_curves <- function(curve_list, predictor) {
  x_min <- max(vapply(curve_list, function(curve) min(curve$x, na.rm = TRUE), numeric(1)), na.rm = TRUE)
  x_max <- min(vapply(curve_list, function(curve) max(curve$x, na.rm = TRUE), numeric(1)), na.rm = TRUE)

  if (!is.finite(x_min) || !is.finite(x_max) || x_min >= x_max) {
    all_x <- unlist(lapply(curve_list, function(curve) curve$x))
    x_min <- min(all_x, na.rm = TRUE)
    x_max <- max(all_x, na.rm = TRUE)
  }

  if (!is.finite(x_min) || !is.finite(x_max) || x_min >= x_max) {
    stop("Could not create a common x grid for marginal effect predictor ", predictor, ".", call. = FALSE)
  }

  x_grid <- seq(x_min, x_max, length.out = MARGINAL_EFFECT_GRID_SIZE)
  curve_rows <- lapply(seq_along(curve_list), function(i) {
    curve <- curve_list[[i]]
    y_grid <- stats::approx(curve$x, curve$y, xout = x_grid, ties = mean, rule = 2)$y
    data.frame(
      predictor = predictor,
      model_id = as.integer(names(curve_list)[i]),
      x = x_grid,
      y = y_grid,
      stringsAsFactors = FALSE
    )
  })

  do.call(rbind, curve_rows)
}

extract_marginal_effect_curves <- function(model_list, top_predictors) {
  curve_tables <- list()

  for (predictor in top_predictors) {
    message("Calculating marginal effect curves for ", predictor, "...")
    raw_curves <- list()
    for (model_id in seq_along(model_list)) {
      curve <- tryCatch(
        partial_curve_from_model(model_list[[model_id]], predictor),
        error = function(e) {
          warning("Skipping marginal effect curve for ", predictor, " in model ", model_id, ": ", conditionMessage(e))
          NULL
        }
      )
      if (!is.null(curve) && nrow(curve) >= 2) {
        raw_curves[[as.character(model_id)]] <- curve
      }
    }

    if (length(raw_curves) == 0) {
      warning("No marginal effect curves were available for ", predictor, ".")
      next
    }

    curve_tables[[predictor]] <- interpolate_partial_curves(raw_curves, predictor)
  }

  if (length(curve_tables) == 0) {
    stop("No marginal effect curves could be calculated.", call. = FALSE)
  }

  do.call(rbind, curve_tables)
}

expand_plot_range <- function(value_range) {
  if (!all(is.finite(value_range))) {
    return(c(0, 1))
  }
  if (diff(value_range) == 0) {
    pad <- max(abs(value_range[1]) * 0.05, 0.5)
    return(value_range + c(-pad, pad))
  }
  value_range
}

plot_marginal_effect_curves <- function(curve_table, top_predictors) {
  output_png <- marginal_effect_plot_path()
  if (file.exists(output_png) && !OVERWRITE_MARGINAL_EFFECT_OUTPUTS) {
    return(invisible(output_png))
  }

  make_parent_dir(output_png)
  grDevices::png(output_png, width = 2400, height = 2200, res = 150)
  on.exit(grDevices::dev.off(), add = TRUE)
  graphics::par(mfrow = c(4, 4), mar = c(4, 4, 3, 1), mgp = c(2.5, 0.8, 0))

  for (predictor in top_predictors) {
    predictor_curves <- curve_table[curve_table$predictor == predictor, , drop = FALSE]
    if (nrow(predictor_curves) == 0) {
      plot.new()
      graphics::title(main = predictor)
      next
    }

    x_range <- expand_plot_range(range(predictor_curves$x, na.rm = TRUE))
    y_range <- expand_plot_range(range(predictor_curves$y, na.rm = TRUE))
    graphics::plot(
      NA,
      xlim = x_range,
      ylim = y_range,
      xlab = predictor,
      ylab = "Partial effect",
      main = predictor,
      las = 1
    )

    for (model_id in sort(unique(predictor_curves$model_id))) {
      model_curve <- predictor_curves[predictor_curves$model_id == model_id, , drop = FALSE]
      graphics::lines(
        model_curve$x,
        model_curve$y,
        col = grDevices::adjustcolor("steelblue4", alpha.f = 0.18),
        lwd = 1
      )
    }

    average_curve <- stats::aggregate(y ~ x, data = predictor_curves, FUN = mean)
    graphics::lines(average_curve$x, average_curve$y, col = "black", lwd = 2.5)
  }

  invisible(output_png)
}

write_marginal_effect_outputs <- function(model_list, importance_summary) {
  top_predictors <- head(importance_summary$predictor, MARGINAL_EFFECT_TOP_N)

  if (file.exists(marginal_effect_curves_csv_path()) && !OVERWRITE_MARGINAL_EFFECT_OUTPUTS) {
    curve_table <- utils::read.csv(marginal_effect_curves_csv_path(), stringsAsFactors = FALSE)
  } else {
    curve_table <- extract_marginal_effect_curves(model_list, top_predictors)
    make_parent_dir(marginal_effect_curves_csv_path())
    utils::write.csv(curve_table, marginal_effect_curves_csv_path(), row.names = FALSE)
    message("Saved marginal effect curve table: ", marginal_effect_curves_csv_path())
  }

  plot_marginal_effect_curves(curve_table, top_predictors)
  invisible(curve_table)
}


#### 1. Read Training Dataset ####

require_package("gbm")
require_package("dismo")
require_package("terra")
require_package("sf")
require_package("ggplot2")
if (RUN_PERFORMANCE_EVALUATION) {
  require_package("pROC")
  require_package("PRROC")
  require_package("caret")
}

make_dir(MODEL_DIR)
make_dir(OUTPUT_DIR)
make_dir(PREDICTION_TABLE_DIR)
make_dir(ANNUAL_SUMMARY_RASTER_DIR)
make_dir(ANNUAL_SUMMARY_PLOT_DIR)
make_dir(ROR_ESTIMATE_DIR)
make_dir(ROR_ESTIMATE_PLOT_DIR)
make_dir(ROR_CHANGE_DIR)
make_dir(ROR_CHANGE_PLOT_DIR)
make_dir(RAW_ODDS_CHANGE_DIR)
make_dir(RAW_ODDS_CHANGE_PLOT_DIR)
make_dir(PIXEL_RELATIVE_PREDICTION_DIR)
make_dir(PIXEL_RELATIVE_PREDICTION_PLOT_DIR)
make_dir(MEAN_DERIVED_FIGURE_DIR)
make_dir(MODEL_DIAGNOSTIC_DIR)
make_dir(MODEL_DIAGNOSTIC_PLOT_DIR)
make_dir(SHAP_DIR)
make_dir(SHAP_TABLE_DIR)
make_dir(PERFORMANCE_EVALUATION_DIR)

message("Analysis folder: ", if (nzchar(ACTIVE_ANALYSIS_NAME)) ACTIVE_ANALYSIS_NAME else "default all-types folders")
message(
  "Training type filter: ",
  if (length(ACTIVE_TRAINING_TYPE_FILTER) > 0) paste(ACTIVE_TRAINING_TYPE_FILTER, collapse = ", ") else "all event types"
)
message("Model directory: ", MODEL_DIR)
message("Output directory: ", OUTPUT_DIR)

dataset2 <- utils::read.csv(TRAINING_CSV, stringsAsFactors = FALSE)
prediction_grid <- utils::read.csv(PREDICTION_GRID_CSV, stringsAsFactors = FALSE)
dataset2 <- filter_training_dataset_by_type(dataset2)

predictor_names <- identify_predictors(dataset2, prediction_grid)
dataset2 <- as_numeric_predictors(dataset2, predictor_names)
prediction_grid <- as_numeric_predictors(prediction_grid, predictor_names)
dataset2 <- fill_hansen_land_na_with_zero(dataset2)
prediction_grid <- fill_hansen_land_na_with_zero(prediction_grid)
prediction_grid <- add_country_to_prediction_grid(prediction_grid)
dataset2[[OUTCOME_COLUMN]] <- as.integer(dataset2[[OUTCOME_COLUMN]])

check_training_dataset(dataset2, predictor_names)
utils::write.csv(data.frame(predictor = predictor_names), PREDICTOR_NAMES_CSV, row.names = FALSE)
saveRDS(predictor_names, PREDICTOR_NAMES_RDS)

message("Training rows: ", format(nrow(dataset2), big.mark = ","))
message("Prediction-grid rows: ", format(nrow(prediction_grid), big.mark = ","))
message("Predictors: ", length(predictor_names))
message("Training year range: ", min(dataset2$year), "-", max(dataset2$year))
message("Prediction years: ", paste(PREDICTION_YEARS, collapse = ", "))
message("BRT settings: tree.complexity = ", TREE_COMPLEXITY, ", learning.rate = ", LEARNING_RATE, ", bag.fraction = ", BAG_FRACTION, ", n.folds = ", N_FOLDS)


#### 2. Create Sampled Dataset List ####

set.seed(RANDOM_SEED)
event_count <- sum(dataset2[[OUTCOME_COLUMN]] == EVENT_VALUE)
control_count <- sum(dataset2[[OUTCOME_COLUMN]] == CONTROL_VALUE)

message("Events: ", event_count)
message("Controls: ", control_count)
message("Sampling ", CONTROLS_PER_EVENT, " controls per event for each of ", N_DATASETS, " datasets.")

dsl <- vector("list", N_DATASETS)
for (j in seq_len(N_DATASETS)) {
  dsl[[j]] <- make_sampled_dataset(
    dataset2,
    controls_per_event = CONTROLS_PER_EVENT,
    seed = RANDOM_SEED + j,
    replace_controls = CONTROL_SAMPLE_WITH_REPLACEMENT
  )
  message("  dsl[[", j, "]] rows: ", nrow(dsl[[j]]))
}

saveRDS(dsl, DSL_RDS)
message("Saved sampled dataset list: ", DSL_RDS)


#### 3. Train BRT Model List ####

dsl <- readRDS(DSL_RDS)
predictor_names <- readRDS(PREDICTOR_NAMES_RDS)

model_list <- vector("list", length(dsl))
for (j in seq_along(dsl)) {
  message("Training BRT model ", j, " of ", length(dsl), "...")
  model_list[[j]] <- fit_brt_model(dsl[[j]], predictor_names, model_id = j)
}

saveRDS(model_list, MODEL_LIST_RDS)
message("Saved model_list: ", MODEL_LIST_RDS)


#### 4. Load Models And Prediction Grid Dataset ####

model_list <- readRDS(MODEL_LIST_RDS)
predictor_names <- readRDS(PREDICTOR_NAMES_RDS)
prediction_grid <- utils::read.csv(PREDICTION_GRID_CSV, stringsAsFactors = FALSE)

missing_prediction_predictors <- setdiff(predictor_names, names(prediction_grid))
if (length(missing_prediction_predictors) > 0) {
  stop("Prediction grid is missing predictors: ", paste(missing_prediction_predictors, collapse = ", "), call. = FALSE)
}
prediction_grid <- as_numeric_predictors(prediction_grid, predictor_names)
prediction_grid <- fill_hansen_land_na_with_zero(prediction_grid)
prediction_grid <- fill_prediction_covariates_from_reference_years(prediction_grid, LATEST_AVAILABLE_COVARIATE_FILLS)
prediction_grid <- add_country_to_prediction_grid(prediction_grid)

missing_years <- setdiff(PREDICTION_YEARS, sort(unique(prediction_grid$year)))
if (length(missing_years) > 0) {
  stop("Prediction grid is missing requested years: ", paste(missing_years, collapse = ", "), call. = FALSE)
}

prediction_grid <- prediction_grid[prediction_grid$year %in% PREDICTION_YEARS, , drop = FALSE]
message("Loaded ", length(model_list), " fitted BRT models.")
message("Prediction rows retained: ", format(nrow(prediction_grid), big.mark = ","))


#### 5. Predict Each Model On Prediction Grid Data Frame ####

model_list <- readRDS(MODEL_LIST_RDS)
predictor_names <- readRDS(PREDICTOR_NAMES_RDS)
prediction_grid <- utils::read.csv(PREDICTION_GRID_CSV, stringsAsFactors = FALSE)
prediction_grid <- as_numeric_predictors(prediction_grid, predictor_names)
prediction_grid <- fill_hansen_land_na_with_zero(prediction_grid)
prediction_grid <- fill_prediction_covariates_from_reference_years(prediction_grid, LATEST_AVAILABLE_COVARIATE_FILLS)
prediction_grid <- add_country_to_prediction_grid(prediction_grid)
prediction_grid <- prediction_grid[prediction_grid$year %in% PREDICTION_YEARS, , drop = FALSE]

if (!file.exists(MODEL_PREDICTION_TABLE_CSV) || OVERWRITE_MODEL_PREDICTION_TABLE) {
  message("Predicting event probability for each model over the prediction-grid table...")
  model_prediction_table <- make_model_prediction_table(prediction_grid, predictor_names, model_list)
  utils::write.csv(model_prediction_table, MODEL_PREDICTION_TABLE_CSV, row.names = FALSE)
  message("Saved model prediction table: ", MODEL_PREDICTION_TABLE_CSV)
} else {
  message("Model prediction table exists; loading cached file: ", MODEL_PREDICTION_TABLE_CSV)
  model_prediction_table <- utils::read.csv(MODEL_PREDICTION_TABLE_CSV, stringsAsFactors = FALSE)
}




#### 6. Plot Relative Importance Across Model Fits ####

model_list <- readRDS(MODEL_LIST_RDS)
predictor_names <- readRDS(PREDICTOR_NAMES_RDS)
importance_summary <- write_relative_importance_outputs(model_list, predictor_names)


#### 7. Plot Marginal Effect Curves ####

write_marginal_effect_outputs(model_list, importance_summary)

message("Model diagnostics complete.")




#### 8. Summarize, Rasterize, And Plot Annual Predictions ####

model_prediction_table <- utils::read.csv(MODEL_PREDICTION_TABLE_CSV, stringsAsFactors = FALSE)

if (!file.exists(ANNUAL_SUMMARY_PREDICTION_CSV) || OVERWRITE_ANNUAL_SUMMARIES) {
  message("Calculating min, max, mean, and median prediction columns...")
  annual_summary_predictions <- summarize_model_predictions(model_prediction_table)
  utils::write.csv(annual_summary_predictions, ANNUAL_SUMMARY_PREDICTION_CSV, row.names = FALSE)
  message("Saved prediction summary table: ", ANNUAL_SUMMARY_PREDICTION_CSV)
} else {
  message("Prediction summary table exists; loading cached file: ", ANNUAL_SUMMARY_PREDICTION_CSV)
  annual_summary_predictions <- utils::read.csv(ANNUAL_SUMMARY_PREDICTION_CSV, stringsAsFactors = FALSE)
}

for (year in PREDICTION_YEARS) {
  message("Rasterizing and plotting annual prediction summaries for ", year, "...")
  df_year <- annual_summary_predictions[annual_summary_predictions$year == year, , drop = FALSE]
  if (nrow(df_year) == 0) {
    stop("Prediction summary table has no rows for year ", year, ".", call. = FALSE)
  }

  summary_raster <- prediction_summary_to_raster_stack(df_year)
  output_tif <- annual_summary_raster_path(year)
  terra::writeRaster(
    summary_raster,
    output_tif,
    overwrite = OVERWRITE_ANNUAL_SUMMARY_RASTERS,
    wopt = RASTER_WRITE_OPTIONS
  )
  message("  Wrote: ", output_tif)
  plot_annual_summary_rasters(summary_raster, year)
}

#### 9. Assess Apparent Training-Point Prediction Performance ####

# This extracts annual predictions back to the rows of dataset2 from the same
# year, then evaluates ranking and thresholded classification performance.
# Because this 03_ model is trained on the full dataset, these are apparent
# in-sample metrics. Use 03b_ for held-forward temporal validation.
write_performance_outputs(dataset2, EVALUATION_YEARS)


#### 10. Calculate SHAP Values For Prediction Grid ####

# TreeSHAP values are calculated on the GBM link scale. For Bernoulli BRTs,
# this is the additive log-odds scale, which is the preferred scale for asking
# which covariates pushed a pixel-year prediction up or down.
model_list <- readRDS(MODEL_LIST_RDS)
dsl <- readRDS(DSL_RDS)
predictor_names <- readRDS(PREDICTOR_NAMES_RDS)
prediction_grid <- utils::read.csv(PREDICTION_GRID_CSV, stringsAsFactors = FALSE)
prediction_grid <- as_numeric_predictors(prediction_grid, predictor_names)
prediction_grid <- fill_hansen_land_na_with_zero(prediction_grid)
prediction_grid <- fill_prediction_covariates_from_reference_years(prediction_grid, LATEST_AVAILABLE_COVARIATE_FILLS)
prediction_grid <- add_country_to_prediction_grid(prediction_grid)
prediction_grid <- prediction_grid[prediction_grid$year %in% PREDICTION_YEARS, , drop = FALSE]
model_prediction_table <- utils::read.csv(MODEL_PREDICTION_TABLE_CSV, stringsAsFactors = FALSE)

write_prediction_shap_outputs(
  prediction_grid = prediction_grid,
  model_prediction_table = model_prediction_table,
  predictor_names = predictor_names,
  model_list = model_list,
  dsl = dsl
)


#### 11. Create Relative Odds Ratio Annual Rasters ####

# ROR is calculated as each cell's predicted odds divided by the study-area
# baseline odds for that same year and summary measure.
build_ror_outputs(PREDICTION_YEARS)


#### 12. Create 1-Year Ratios In RORs ####

# chgROR_1yr compares each year's ROR with the previous year's ROR.
build_ror_change_outputs(PREDICTION_YEARS)


#### 13. Create 1-Year Ratios In Raw Predicted Odds ####

# chgOdds_1yr compares raw predicted odds between adjacent years, without
# rescaling by each year's study-area baseline odds.
build_raw_odds_change_outputs(PREDICTION_YEARS)


#### 14. Create Pixel-Relative Annual Prediction Ratios ####

# relPred compares each pixel-year's predicted probability with that same
# pixel's average predicted probability across all requested prediction years.
build_pixel_relative_prediction_outputs(PREDICTION_YEARS)
