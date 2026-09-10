# Event Ecological Niche Pipeline

This repository is a handoff-ready pipeline for predicting where and when an event might occur. The structure follows the reference code conceptually, but separates sampling, covariate extraction, model training, and prediction into reproducible steps.

The current toy study area is mainland Africa within approximately 10 degrees north/south of the equator. The canonical study-area input is `config/africacountries_nolakes.shp`, clipped to the equatorial band inside the scripts. Climate covariates use global products available in Earth Engine, including ERA5-Land aggregates for precipitation, temperature, and PET, and MODIS for NDVI.

Prediction grid cells and pseudo-absence points are created inside the core study area. Covariate extraction uses the same land shapefile with a small latitude margin so 100 km donut buffers near the study-area edge can still summarize nearby land context.

## Folder Structure

- `R_python_code/`: R scripts plus Python notebooks/helpers that orchestrate Google Earth Engine exports.
- `config/`: Africa no-lakes study-area shapefile, predictor list, and optional external raster config.
- `data/`: local CSV inputs/outputs, created locally and ignored by Git.
- `models/`: fitted model objects, created locally and ignored by Git.
- `outputs/`: manifests, validation metrics, predictions, and rasters, created locally and ignored by Git.

The GitHub repo is intentionally set up to track only `R_python_code/`,
`config/`, `README.md`, and `.gitignore`. Generated covariate exports, fitted
models, rasters, HTML reports, and local session files should be created on each
analyst's machine and are ignored by `.gitignore`.

## Pipeline

Run scripts from the `KSPH Code` repo root.

### 1. Create Dataset1

```r
source("R_python_code/01_make_dataset1.R")
```

Output:

- `data/dataset1.csv`

This table contains:

- `id`
- `year`
- `latitude`
- `longitude`
- `outcome`
- `point_type`
- `source`
- `sampling_version`
- `created_date`

The script reads presence records from `data/outcomes_csv.csv` and preserves the `type` and `country` columns. Because `data/` is ignored by Git, each analyst should create that folder locally and place their own `outcomes_csv.csv` there before running the full pipeline.

### 2. Extract Training And Prediction-Grid Covariates

Install Python requirements if needed:

```bash
python -m pip install -r R_python_code/requirements.txt
```

Recommended notebook workflow:

- `R_python_code/02_predGrid_trainSet_extraction.ipynb`

That notebook has two linked phases:

1. Create/read the prediction grid and extract prediction-grid covariates.
2. Create/read the training point buffers and extract matching training covariates.

The notebook exports large covariate tables to Google Drive because Earth
Engine can write large tables there server-side. Google Drive for Desktop then
syncs those CSVs locally for the merge cells.

Key outputs:

- `data/dataset2.csv`
- `data/prediction_grid_10km.csv`
- `data/prediction_grid_covariates_2020_2025.csv`

The active study-area controls live near the top of the notebook:

- `STUDY_AREA_NAME`
- `STUDY_AREA_FILE`
- `STUDY_AREA_BBOX`
- `STUDY_AREA_CONTEXT_BUFFER_DEGREES`

Set `STUDY_AREA_BBOX = None` to use the full extent/polygon of a
country-specific `STUDY_AREA_FILE`.

### 3. Train Full BRT Model And Predict Maps

```r
source("R_python_code/03_train_predict_brt_simple.R")
```

Outputs:

- `models`
- `outputs/predictions`

The script creates sampled control/event training datasets, fits the BRT
ensemble, predicts over the prediction-grid table, rasterizes annual summaries,
and creates derived ROR/change outputs and model diagnostics.

### 3b. Temporal Forward Validation

```r
source("R_python_code/03b_temporal_forward_validation_brt.R")
```

Outputs:

- `outputs/tfv`

This retrospective workflow trains only on years before each target year, then
predicts the held-forward year.

### 4. Interactive Reports

Run the Shiny report while working interactively:

```r
rmarkdown::run("R_python_code/04_interactive_prediction_report_Shiny.Rmd")
```

Knit the email-friendly HTML report:

```r
rmarkdown::render("R_python_code/04b_interactive_prediction_report_email.Rmd")
```

### 5. SuperLearner Aim-1 Ensemble (GBM + RF + SVM)

```r
source("R_python_code/05_train_predict_SuperLearner_CV.R")
```

Requires `data/dataset2.csv` and `data/prediction_grid_covariates_2020_2025.csv`.

This script fits the protocol Aim-1 library:

- `SL.aim1_gbm` (Bernoulli GBM)
- `SL.aim1_rf` (classification random forest)
- `SL.aim1_svm` (`kernlab` C-svc with probabilities)

Outer stratified CV scores the SuperLearner stack and each base learner. The stack is **retained only if its ROC-AUC beats the best single learner**; otherwise the discrete winner is used for final maps. Outputs land under `models/superlearner_cv/` and `outputs/predictions/superlearner_cv/`.

`03_train_predict_brt_simple.R` remains the BRT baseline path.

## Core Covariates

Spatially scaled covariates are extracted over donut buffers:

- `0_10km`
- `10_25km`
- `25_50km`

These include:

- forest cover proportion
- same-year forest loss
- one-year-prior forest loss
- two-year-prior forest loss
- forest edge proportion
- population density

Non-scaled covariates are extracted over `0_10km` only:

- annual precipitation
- annual precipitation anomaly
- standardized precipitation anomaly
- annual temperature
- annual temperature anomaly
- standardized temperature anomaly
- annual potential evapotranspiration
- annual NDVI
- annual NDVI anomaly
- standardized NDVI anomaly
- elevation

The model predictor names live in:

- `config/predictor_list.csv`

## External Rasters

To add external raster covariates, upload the raster to Earth Engine as an asset, then edit:

- `config/external_rasters.csv`

Set `enabled` to `true` and choose `buffer_mode`:

- `nonscaled`: extract only `0_10km`
- `scaled`: extract all donut buffers

## Notes

- LandScan is used with a latest-available-year rule.
- Forest-loss lag variables are intentionally missing for early years through 2003.
- Prediction grid cells are sampled at approximately 10 km resolution.
- Training and grid locations are limited to the configured study area, while covariate extraction uses a buffered land context near the study-area edge.
- `01_make_dataset1.R` requires `sf` for true polygon sampling from `africacountries_nolakes.shp`. A bounding-box fallback exists for quick testing only, but is disabled by default because it can sample ocean points.
- The Python GEE scripts require `geopandas`/`shapely` to read the same shapefile and build the land-only prediction grid.
- The older `01_extract_covariates_gee.js` file is retained as a Code Editor prototype, but the production-oriented pipeline uses the Python scripts.
