#### 04 Train SuperLearner With CV And Predict Event Probability ####

# This script mirrors the table-first prediction workflow in
# 03_train_predict_brt_simple.R, but fits a SuperLearner model instead of a BRT.
# It keeps the first pass deliberately light: three quick learners, 10-fold
# outer cross-validation, F1-based candidate/threshold selection, and annual
# prediction maps for 2020-2025.


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
    "Could not locate the KSPH Code directory. Run this with source('R_python_code/05_train_predict_SuperLearner_CV.R') ",
    "from the KSPH Code repo root, or source('KSPH Code/R_python_code/05_train_predict_SuperLearner_CV.R') from the parent folder.",
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

TRAINING_CSV <- file.path(CODE_DIR, "data", "dataset2.csv")
PREDICTION_GRID_CSV <- file.path(CODE_DIR, "data", "prediction_grid_covariates_2020_2025.csv")

MODEL_DIR <- file.path(CODE_DIR, "models", "superlearner_cv")
OUTPUT_DIR <- file.path(CODE_DIR, "outputs", "predictions", "superlearner_cv")
PREDICTION_TABLE_DIR <- file.path(OUTPUT_DIR, "tables")
ANNUAL_SUMMARY_RASTER_DIR <- file.path(OUTPUT_DIR, "annual_summary_rasters")
ANNUAL_SUMMARY_PLOT_DIR <- file.path(OUTPUT_DIR, "annual_summary_plots")

PREDICTOR_NAMES_RDS <- file.path(MODEL_DIR, "predictor_names.rds")
PREDICTOR_NAMES_CSV <- file.path(MODEL_DIR, "predictor_names.csv")
CLEAN_TRAINING_CSV <- file.path(MODEL_DIR, "training_model_matrix.csv")
IMPUTATION_VALUES_RDS <- file.path(MODEL_DIR, "predictor_imputation_values.rds")
IMPUTATION_VALUES_CSV <- file.path(MODEL_DIR, "predictor_imputation_values.csv")
FOLD_ASSIGNMENT_CSV <- file.path(MODEL_DIR, "cv_fold_assignments.csv")
SL_TUNING_RESULTS_RDS <- file.path(MODEL_DIR, "superlearner_tuning_results.rds")
SL_TUNING_RESULTS_CSV <- file.path(MODEL_DIR, "superlearner_tuning_results.csv")
SL_CV_PREDICTIONS_CSV <- file.path(MODEL_DIR, "superlearner_cv_predictions.csv")
SL_BEST_SETTINGS_RDS <- file.path(MODEL_DIR, "superlearner_best_settings.rds")
SL_BEST_SETTINGS_CSV <- file.path(MODEL_DIR, "superlearner_best_settings.csv")
SL_FIT_RDS <- file.path(MODEL_DIR, "superlearner_fit.rds")

MODEL_PREDICTION_TABLE_CSV <- file.path(
  PREDICTION_TABLE_DIR,
  "prediction_grid_superlearner_predictions_2020_2025.csv"
)
ANNUAL_SUMMARY_PREDICTION_CSV <- file.path(
  PREDICTION_TABLE_DIR,
  "prediction_grid_superlearner_summaries_2020_2025.csv"
)

RANDOM_SEED <- 20260827
OUTCOME_COLUMN <- "outcome"
EVENT_VALUE <- 1L
CONTROL_VALUE <- 0L
TRAINING_BASE_COLUMNS <- c("id", "year", "latitude", "longitude", "outcome", "type", "country")
PREDICTION_BASE_COLUMNS <- c("grid_id", "grid_batch", "x", "y", "year", "longitude", "latitude")

N_FOLDS <- 10
SL_INTERNAL_FOLDS <- 5
SL_LIBRARY <- c("SL.glm", "SL.rpart_tuned", "SL.ranger_tuned")
SL_METHOD <- "method.NNloglik"
RANGER_THREADS <- 1
USE_CLASS_WEIGHTS <- FALSE

# Keep this grid small while the workflow is being tested. Each row is one
# SuperLearner candidate evaluated by the same 10 outer folds and F1 threshold
# grid.
SL_TUNING_GRID <- data.frame(
  candidate_id = c("quick_conservative", "quick_balanced", "quick_flexible"),
  rpart_cp = c(0.010, 0.005, 0.002),
  rpart_maxdepth = c(4L, 5L, 6L),
  rpart_minbucket = c(20L, 12L, 8L),
  ranger_num_trees = c(150L, 200L, 250L),
  ranger_mtry_fraction = c(0.35, 0.50, 0.65),
  ranger_min_node_size = c(20L, 12L, 8L),
  stringsAsFactors = FALSE
)

THRESHOLD_GRID <- sort(unique(c(
  seq(0.001, 0.050, by = 0.001),
  seq(0.055, 0.200, by = 0.005),
  seq(0.250, 0.500, by = 0.050)
)))

PREDICTION_YEARS <- 2020:2025
RASTER_CRS <- "EPSG:4326"
COORDINATE_ROUND_DIGITS <- 10
OVERWRITE_CV_RESULTS <- TRUE
OVERWRITE_FINAL_MODEL <- TRUE
OVERWRITE_MODEL_PREDICTION_TABLE <- TRUE
OVERWRITE_ANNUAL_SUMMARIES <- TRUE
OVERWRITE_ANNUAL_SUMMARY_RASTERS <- TRUE
OVERWRITE_ANNUAL_SUMMARY_PLOTS <- TRUE
PREDICTION_COLUMNS <- "pred_superlearner"
PREDICTION_SUMMARY_COLUMNS <- c("pred_min", "pred_max", "pred_mean", "pred_median")

# SuperLearner does not tolerate missing predictors. For this first version,
# Hansen-derived missing values are treated as 0, including early lagged forest
# loss years. Remaining missing values are filled with training-set medians.
HANSEN_ZERO_FILL_PREFIXES <- c(
  "forest_cover_prop_",
  "flsy_prop_",
  "fl1yp_prop_",
  "fl2yp_prop_",
  "frag_edge_prop_"
)

# This fill is only needed for already-exported tables that used TerraClimate
# PET. New extraction runs use ERA5-Land Daily Aggregated PET and should have
# 2025 values.
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

as_numeric_predictors <- function(df, predictor_names) {
  for (predictor in predictor_names) {
    df[[predictor]] <- suppressWarnings(as.numeric(df[[predictor]]))
    df[[predictor]][!is.finite(df[[predictor]])] <- NA_real_
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

fill_hansen_na_with_zero <- function(df) {
  for (prefix in HANSEN_ZERO_FILL_PREFIXES) {
    columns <- grep(paste0("^", prefix), names(df), value = TRUE)
    for (column in columns) {
      fill_rows <- is.na(df[[column]])
      if (any(fill_rows)) {
        df[[column]][fill_rows] <- 0
        message("Filled ", format(sum(fill_rows), big.mark = ","), " missing ", column, " values with 0.")
      }
    }
  }
  df
}

fit_imputation_values <- function(df, predictor_names) {
  imputation_values <- vapply(
    predictor_names,
    function(predictor) {
      values <- df[[predictor]]
      finite_values <- values[is.finite(values)]
      if (length(finite_values) == 0) {
        stop("Predictor has no finite training values after Hansen zero-fill: ", predictor, call. = FALSE)
      }
      stats::median(finite_values, na.rm = TRUE)
    },
    numeric(1)
  )
  imputation_values
}

apply_imputation_values <- function(df, predictor_names, imputation_values, label) {
  for (predictor in predictor_names) {
    fill_rows <- is.na(df[[predictor]])
    if (any(fill_rows)) {
      df[[predictor]][fill_rows] <- imputation_values[[predictor]]
      message(
        label, ": filled ", format(sum(fill_rows), big.mark = ","),
        " remaining missing ", predictor, " values with training median ",
        signif(imputation_values[[predictor]], 5), "."
      )
    }
  }
  df
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

  if (sum(df[[OUTCOME_COLUMN]] == EVENT_VALUE) < N_FOLDS) {
    stop("There are fewer events than CV folds; reduce N_FOLDS or add more event rows.", call. = FALSE)
  }

  missing_counts <- vapply(df[predictor_names], function(x) sum(is.na(x)), integer(1))
  if (any(missing_counts > 0)) {
    stop("Training predictors still contain missing values after imputation.", call. = FALSE)
  }

  invisible(TRUE)
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

prepare_training_data <- function(training_df, prediction_df) {
  predictor_names <- identify_predictors(training_df, prediction_df)
  training_df <- as_numeric_predictors(training_df, predictor_names)
  training_df <- fill_hansen_na_with_zero(training_df)
  imputation_values <- fit_imputation_values(training_df, predictor_names)
  training_df <- apply_imputation_values(training_df, predictor_names, imputation_values, "training")
  training_df[[OUTCOME_COLUMN]] <- as.integer(training_df[[OUTCOME_COLUMN]])
  check_training_dataset(training_df, predictor_names)

  list(
    training_df = training_df,
    predictor_names = predictor_names,
    imputation_values = imputation_values
  )
}

prepare_prediction_grid <- function(prediction_grid, predictor_names, imputation_values) {
  missing_prediction_predictors <- setdiff(predictor_names, names(prediction_grid))
  if (length(missing_prediction_predictors) > 0) {
    stop("Prediction grid is missing predictors: ", paste(missing_prediction_predictors, collapse = ", "), call. = FALSE)
  }

  prediction_grid <- prediction_grid[prediction_grid$year %in% PREDICTION_YEARS, , drop = FALSE]
  prediction_grid <- as_numeric_predictors(prediction_grid, predictor_names)
  prediction_grid <- fill_hansen_na_with_zero(prediction_grid)
  prediction_grid <- fill_prediction_covariates_from_reference_years(prediction_grid, LATEST_AVAILABLE_COVARIATE_FILLS)
  prediction_grid <- apply_imputation_values(prediction_grid, predictor_names, imputation_values, "prediction grid")
  prediction_grid
}

make_stratified_fold_ids <- function(y, k, seed) {
  set.seed(seed)
  fold_id <- rep(NA_integer_, length(y))
  event_rows <- sample(which(y == EVENT_VALUE))
  control_rows <- sample(which(y == CONTROL_VALUE))

  if (length(event_rows) < k) {
    stop("Each fold needs at least one event. Reduce k or add event rows.", call. = FALSE)
  }

  fold_id[event_rows] <- rep(seq_len(k), length.out = length(event_rows))
  fold_id[control_rows] <- rep(seq_len(k), length.out = length(control_rows))
  fold_id
}

make_observation_weights <- function(y) {
  if (!USE_CLASS_WEIGHTS) {
    return(rep(1, length(y)))
  }

  n_event <- sum(y == EVENT_VALUE)
  n_control <- sum(y == CONTROL_VALUE)
  weights <- rep(NA_real_, length(y))
  weights[y == EVENT_VALUE] <- length(y) / (2 * n_event)
  weights[y == CONTROL_VALUE] <- length(y) / (2 * n_control)
  weights / mean(weights)
}

f1_metrics <- function(truth, probability, threshold) {
  predicted <- ifelse(probability >= threshold, EVENT_VALUE, CONTROL_VALUE)
  tp <- sum(predicted == EVENT_VALUE & truth == EVENT_VALUE, na.rm = TRUE)
  fp <- sum(predicted == EVENT_VALUE & truth == CONTROL_VALUE, na.rm = TRUE)
  fn <- sum(predicted == CONTROL_VALUE & truth == EVENT_VALUE, na.rm = TRUE)
  tn <- sum(predicted == CONTROL_VALUE & truth == CONTROL_VALUE, na.rm = TRUE)

  precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
  f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)

  data.frame(
    threshold = threshold,
    f1 = f1,
    precision = precision,
    recall = recall,
    tp = tp,
    fp = fp,
    fn = fn,
    tn = tn
  )
}

find_best_f1_threshold <- function(truth, probability, threshold_grid) {
  keep <- is.finite(probability) & !is.na(truth)
  if (!any(keep)) {
    stop("No finite probabilities are available for F1 threshold tuning.", call. = FALSE)
  }

  threshold_results <- do.call(
    rbind,
    lapply(threshold_grid, function(threshold) f1_metrics(truth[keep], probability[keep], threshold))
  )
  threshold_results <- threshold_results[order(
    -threshold_results$f1,
    -threshold_results$precision,
    -threshold_results$recall,
    threshold_results$threshold
  ), , drop = FALSE]
  row.names(threshold_results) <- NULL
  threshold_results[1, , drop = FALSE]
}

.SL_TUNING_ENV <- new.env(parent = emptyenv())

set_sl_tuning_params <- function(candidate) {
  .SL_TUNING_ENV$rpart_cp <- as.numeric(candidate$rpart_cp)
  .SL_TUNING_ENV$rpart_maxdepth <- as.integer(candidate$rpart_maxdepth)
  .SL_TUNING_ENV$rpart_minbucket <- as.integer(candidate$rpart_minbucket)
  .SL_TUNING_ENV$ranger_num_trees <- as.integer(candidate$ranger_num_trees)
  .SL_TUNING_ENV$ranger_mtry_fraction <- as.numeric(candidate$ranger_mtry_fraction)
  .SL_TUNING_ENV$ranger_min_node_size <- as.integer(candidate$ranger_min_node_size)
  invisible(TRUE)
}

event_factor <- function(y) {
  factor(ifelse(y == EVENT_VALUE, "event", "control"), levels = c("control", "event"))
}

probability_from_two_class_matrix <- function(pred_matrix, event_level = "event") {
  if (is.null(dim(pred_matrix))) {
    return(as.numeric(pred_matrix))
  }
  if (event_level %in% colnames(pred_matrix)) {
    return(as.numeric(pred_matrix[, event_level]))
  }
  rep(NA_real_, nrow(pred_matrix))
}

SL.rpart_tuned <- function(Y, X, newX, family, obsWeights, id, ...) {
  data <- data.frame(Y = event_factor(Y), as.data.frame(X), check.names = FALSE)
  fit <- rpart::rpart(
    Y ~ .,
    data = data,
    method = "class",
    weights = obsWeights,
    control = rpart::rpart.control(
      cp = .SL_TUNING_ENV$rpart_cp,
      maxdepth = .SL_TUNING_ENV$rpart_maxdepth,
      minbucket = .SL_TUNING_ENV$rpart_minbucket
    )
  )
  pred <- probability_from_two_class_matrix(predict(fit, newdata = as.data.frame(newX), type = "prob"))
  fit <- list(object = fit)
  class(fit) <- "SL.rpart_tuned"
  list(pred = pred, fit = fit)
}

predict.SL.rpart_tuned <- function(object, newdata, ...) {
  probability_from_two_class_matrix(predict(object$object, newdata = as.data.frame(newdata), type = "prob"))
}

SL.ranger_tuned <- function(Y, X, newX, family, obsWeights, id, ...) {
  X <- as.data.frame(X)
  newX <- as.data.frame(newX)
  mtry <- max(1L, min(ncol(X), round(.SL_TUNING_ENV$ranger_mtry_fraction * ncol(X))))
  data <- data.frame(Y = event_factor(Y), X, check.names = FALSE)

  fit <- ranger::ranger(
    dependent.variable.name = "Y",
    data = data,
    probability = TRUE,
    num.trees = .SL_TUNING_ENV$ranger_num_trees,
    mtry = mtry,
    min.node.size = .SL_TUNING_ENV$ranger_min_node_size,
    case.weights = obsWeights,
    num.threads = RANGER_THREADS,
    seed = RANDOM_SEED
  )
  pred <- probability_from_two_class_matrix(predict(fit, data = newX)$predictions)
  fit <- list(object = fit)
  class(fit) <- "SL.ranger_tuned"
  list(pred = pred, fit = fit)
}

predict.SL.ranger_tuned <- function(object, newdata, ...) {
  probability_from_two_class_matrix(predict(object$object, data = as.data.frame(newdata))$predictions)
}

fit_superlearner_model <- function(X, y, candidate, obs_weights) {
  set_sl_tuning_params(candidate)
  SuperLearner::SuperLearner(
    Y = y,
    X = as.data.frame(X),
    family = stats::binomial(),
    SL.library = SL_LIBRARY,
    method = SL_METHOD,
    obsWeights = obs_weights,
    cvControl = list(V = SL_INTERNAL_FOLDS),
    verbose = FALSE
  )
}

predict_superlearner_probability <- function(model, X) {
  as.numeric(predict(model, newdata = as.data.frame(X), onlySL = TRUE)$pred)
}

candidate_from_grid <- function(row_index) {
  as.list(SL_TUNING_GRID[row_index, , drop = FALSE])
}

cross_validate_superlearner_candidate <- function(training_df, predictor_names, fold_id, candidate) {
  X <- training_df[, predictor_names, drop = FALSE]
  y <- training_df[[OUTCOME_COLUMN]]
  cv_pred <- rep(NA_real_, nrow(training_df))

  for (fold in sort(unique(fold_id))) {
    message("    fold ", fold, " of ", length(unique(fold_id)))
    train_rows <- which(fold_id != fold)
    valid_rows <- which(fold_id == fold)
    obs_weights <- make_observation_weights(y[train_rows])

    fold_fit <- fit_superlearner_model(
      X = X[train_rows, , drop = FALSE],
      y = y[train_rows],
      candidate = candidate,
      obs_weights = obs_weights
    )
    cv_pred[valid_rows] <- predict_superlearner_probability(
      fold_fit,
      X[valid_rows, , drop = FALSE]
    )
    rm(fold_fit)
    gc()
  }

  best_threshold <- find_best_f1_threshold(y, cv_pred, THRESHOLD_GRID)
  list(cv_pred = cv_pred, best_threshold = best_threshold)
}

select_best_tuning_result <- function(tuning_results) {
  tuning_results <- tuning_results[order(
    -tuning_results$f1,
    -tuning_results$precision,
    -tuning_results$recall,
    tuning_results$threshold
  ), , drop = FALSE]
  row.names(tuning_results) <- NULL
  tuning_results[1, , drop = FALSE]
}

prediction_metadata_columns <- function(df) {
  required_first <- c("longitude", "latitude")
  optional_after <- c("x", "y", "grid_id", "grid_batch", "year")
  metadata_columns <- c(required_first, optional_after[optional_after %in% names(df)])
  missing_required <- setdiff(required_first, names(df))
  if (length(missing_required) > 0) {
    stop("Prediction grid is missing required coordinate columns: ", paste(missing_required, collapse = ", "), call. = FALSE)
  }
  unique(metadata_columns)
}

make_superlearner_prediction_table <- function(prediction_grid, predictor_names, model_object) {
  metadata_columns <- prediction_metadata_columns(prediction_grid)
  prediction_table <- prediction_grid[, metadata_columns, drop = FALSE]
  prediction_table$pred_superlearner <- predict_superlearner_probability(
    model_object$model,
    prediction_grid[, predictor_names, drop = FALSE]
  )
  prediction_table
}

summarize_model_predictions <- function(prediction_table) {
  missing_prediction_columns <- setdiff(PREDICTION_COLUMNS, names(prediction_table))
  if (length(missing_prediction_columns) > 0) {
    stop("Prediction table is missing columns: ", paste(missing_prediction_columns, collapse = ", "), call. = FALSE)
  }

  prediction_matrix <- as.matrix(prediction_table[, PREDICTION_COLUMNS, drop = FALSE])
  all_missing <- rowSums(!is.na(prediction_matrix)) == 0

  prediction_table$pred_min <- apply(prediction_matrix, 1, function(x) if (all(is.na(x))) NA_real_ else min(x, na.rm = TRUE))
  prediction_table$pred_max <- apply(prediction_matrix, 1, function(x) if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE))
  prediction_table$pred_mean <- rowMeans(prediction_matrix, na.rm = TRUE)
  prediction_table$pred_mean[all_missing] <- NA_real_
  prediction_table$pred_median <- apply(prediction_matrix, 1, function(x) if (all(is.na(x))) NA_real_ else stats::median(x, na.rm = TRUE))

  metadata_columns <- prediction_metadata_columns(prediction_table)
  prediction_table[, c(metadata_columns, PREDICTION_COLUMNS, PREDICTION_SUMMARY_COLUMNS), drop = FALSE]
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
  file.path(ANNUAL_SUMMARY_RASTER_DIR, sprintf("event_probability_superlearner_summary_%s.tif", year))
}

annual_summary_plot_path <- function(year, summary_column) {
  file.path(ANNUAL_SUMMARY_PLOT_DIR, sprintf("event_probability_superlearner_%s_%s.png", summary_column, year))
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

    grDevices::png(output_png, width = 1600, height = 1000, res = 150)
    terra::plot(
      raster_layer,
      main = sprintf("SuperLearner Event Probability %s %s", sub("^pred_", "", summary_column), year),
      col = grDevices::hcl.colors(100, "Viridis")
    )
    grDevices::dev.off()
  }
}


#### 1. Read Training Dataset ####

require_package("SuperLearner")
require_package("rpart")
require_package("ranger")
require_package("terra")

make_dir(MODEL_DIR)
make_dir(OUTPUT_DIR)
make_dir(PREDICTION_TABLE_DIR)
make_dir(ANNUAL_SUMMARY_RASTER_DIR)
make_dir(ANNUAL_SUMMARY_PLOT_DIR)

dataset2_raw <- utils::read.csv(TRAINING_CSV, stringsAsFactors = FALSE)
prediction_grid_raw <- utils::read.csv(PREDICTION_GRID_CSV, stringsAsFactors = FALSE)

prepared <- prepare_training_data(dataset2_raw, prediction_grid_raw)
dataset2 <- prepared$training_df
predictor_names <- prepared$predictor_names
imputation_values <- prepared$imputation_values

utils::write.csv(data.frame(predictor = predictor_names), PREDICTOR_NAMES_CSV, row.names = FALSE)
saveRDS(predictor_names, PREDICTOR_NAMES_RDS)
utils::write.csv(
  data.frame(predictor = names(imputation_values), imputation_value = as.numeric(imputation_values)),
  IMPUTATION_VALUES_CSV,
  row.names = FALSE
)
saveRDS(imputation_values, IMPUTATION_VALUES_RDS)
utils::write.csv(dataset2, CLEAN_TRAINING_CSV, row.names = FALSE)

message("Training rows: ", format(nrow(dataset2), big.mark = ","))
message("Events: ", sum(dataset2[[OUTCOME_COLUMN]] == EVENT_VALUE))
message("Controls: ", sum(dataset2[[OUTCOME_COLUMN]] == CONTROL_VALUE))
message("Prediction-grid rows available: ", format(nrow(prediction_grid_raw), big.mark = ","))
message("Predictors: ", length(predictor_names))
message("SuperLearner library: ", paste(SL_LIBRARY, collapse = ", "))
message("Outer CV folds: ", N_FOLDS)
message("Class weights enabled: ", USE_CLASS_WEIGHTS)


#### 2. Create Stratified 10-Fold CV Splits ####

fold_id <- make_stratified_fold_ids(dataset2[[OUTCOME_COLUMN]], N_FOLDS, RANDOM_SEED)
fold_assignments <- data.frame(
  row_id = seq_len(nrow(dataset2)),
  id = dataset2$id,
  outcome = dataset2[[OUTCOME_COLUMN]],
  fold = fold_id
)
utils::write.csv(fold_assignments, FOLD_ASSIGNMENT_CSV, row.names = FALSE)

fold_table <- table(fold_assignments$fold, fold_assignments$outcome)
print(fold_table)
message("Saved fold assignments: ", FOLD_ASSIGNMENT_CSV)


#### 3. Tune SuperLearner With 10-Fold CV And F1 ####

if (!file.exists(SL_TUNING_RESULTS_RDS) || OVERWRITE_CV_RESULTS) {
  tuning_results <- data.frame()
  cv_prediction_list <- list()

  for (candidate_index in seq_len(nrow(SL_TUNING_GRID))) {
    candidate <- candidate_from_grid(candidate_index)
    message("Evaluating SuperLearner candidate ", candidate_index, " of ", nrow(SL_TUNING_GRID), ": ", candidate$candidate_id)

    cv_result <- cross_validate_superlearner_candidate(
      training_df = dataset2,
      predictor_names = predictor_names,
      fold_id = fold_id,
      candidate = candidate
    )

    best_threshold <- cv_result$best_threshold
    candidate_result <- cbind(
      SL_TUNING_GRID[candidate_index, , drop = FALSE],
      best_threshold
    )
    tuning_results <- rbind(tuning_results, candidate_result)

    cv_prediction_list[[candidate_index]] <- data.frame(
      candidate_id = candidate$candidate_id,
      row_id = seq_len(nrow(dataset2)),
      id = dataset2$id,
      outcome = dataset2[[OUTCOME_COLUMN]],
      fold = fold_id,
      cv_probability = cv_result$cv_pred
    )
  }

  best_settings <- select_best_tuning_result(tuning_results)
  cv_predictions <- do.call(rbind, cv_prediction_list)

  saveRDS(tuning_results, SL_TUNING_RESULTS_RDS)
  utils::write.csv(tuning_results, SL_TUNING_RESULTS_CSV, row.names = FALSE)
  utils::write.csv(cv_predictions, SL_CV_PREDICTIONS_CSV, row.names = FALSE)
  saveRDS(best_settings, SL_BEST_SETTINGS_RDS)
  utils::write.csv(best_settings, SL_BEST_SETTINGS_CSV, row.names = FALSE)
} else {
  tuning_results <- readRDS(SL_TUNING_RESULTS_RDS)
  best_settings <- readRDS(SL_BEST_SETTINGS_RDS)
}

print(tuning_results)
message("Best SuperLearner candidate: ", best_settings$candidate_id)
message(
  "Best F1 threshold: ", signif(best_settings$threshold, 4),
  " | F1: ", signif(best_settings$f1, 4),
  " | precision: ", signif(best_settings$precision, 4),
  " | recall: ", signif(best_settings$recall, 4)
)


#### 4. Fit Optimized SuperLearner On Full Training Dataset ####

if (!file.exists(SL_FIT_RDS) || OVERWRITE_FINAL_MODEL) {
  best_candidate_index <- match(best_settings$candidate_id, SL_TUNING_GRID$candidate_id)
  best_candidate <- candidate_from_grid(best_candidate_index)
  full_obs_weights <- make_observation_weights(dataset2[[OUTCOME_COLUMN]])

  message("Fitting final optimized SuperLearner on the full training dataset...")
  final_sl_model <- fit_superlearner_model(
    X = dataset2[, predictor_names, drop = FALSE],
    y = dataset2[[OUTCOME_COLUMN]],
    candidate = best_candidate,
    obs_weights = full_obs_weights
  )

  model_object <- list(
    model = final_sl_model,
    predictor_names = predictor_names,
    imputation_values = imputation_values,
    tuning_results = tuning_results,
    best_settings = best_settings,
    sl_library = SL_LIBRARY,
    sl_method = SL_METHOD,
    use_class_weights = USE_CLASS_WEIGHTS,
    created_at = Sys.time()
  )
  saveRDS(model_object, SL_FIT_RDS)
  message("Saved final SuperLearner model: ", SL_FIT_RDS)
} else {
  model_object <- readRDS(SL_FIT_RDS)
  message("Loaded cached final SuperLearner model: ", SL_FIT_RDS)
}


#### 5. Predict SuperLearner On Prediction Grid Data Frame ####

model_object <- readRDS(SL_FIT_RDS)
predictor_names <- model_object$predictor_names
imputation_values <- model_object$imputation_values
prediction_grid_raw <- utils::read.csv(PREDICTION_GRID_CSV, stringsAsFactors = FALSE)
prediction_grid <- prepare_prediction_grid(prediction_grid_raw, predictor_names, imputation_values)

missing_years <- setdiff(PREDICTION_YEARS, sort(unique(prediction_grid$year)))
if (length(missing_years) > 0) {
  stop("Prediction grid is missing requested years: ", paste(missing_years, collapse = ", "), call. = FALSE)
}

if (!file.exists(MODEL_PREDICTION_TABLE_CSV) || OVERWRITE_MODEL_PREDICTION_TABLE) {
  message("Predicting SuperLearner event probability over the prediction-grid table...")
  model_prediction_table <- make_superlearner_prediction_table(
    prediction_grid = prediction_grid,
    predictor_names = predictor_names,
    model_object = model_object
  )
  utils::write.csv(model_prediction_table, MODEL_PREDICTION_TABLE_CSV, row.names = FALSE)
  message("Saved model prediction table: ", MODEL_PREDICTION_TABLE_CSV)
} else {
  message("Model prediction table exists; loading cached file: ", MODEL_PREDICTION_TABLE_CSV)
  model_prediction_table <- utils::read.csv(MODEL_PREDICTION_TABLE_CSV, stringsAsFactors = FALSE)
}


#### 6. Summarize, Rasterize, And Plot Annual Predictions ####

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
  message("Rasterizing and plotting annual SuperLearner prediction summaries for ", year, "...")
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

message("Done.")
