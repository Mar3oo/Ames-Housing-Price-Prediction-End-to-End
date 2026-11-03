from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "AmesHousing.csv"
MODEL_PATH = MODEL_DIR / "gradient_boosting_pipeline.joblib"


# ============================================================
# DATA CONFIGURATION
# ============================================================

TARGET_COLUMN = "SalePrice"

# Columns excluded from model features
DROP_COLUMNS = [
    "Order",
    "PID",
    "LogSalePrice",
]

# Training observations excluded after the sensitivity analysis.
# These are original DataFrame index labels.
EXCLUDED_TRAIN_INDICES = [1498, 2180]


# ============================================================
# DATA SPLITTING CONFIGURATION
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42

CV_SPLITS = 5


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "GradientBoostingRegressor"

MODEL_PARAMS = {
    "n_estimators": 363,
    "learning_rate": 0.061560888611986816,
    "max_depth": 4,
    "max_features": 0.7,
    "min_samples_leaf": 4,
    "min_samples_split": 8,
    "subsample": 0.7103165563345655,
    "random_state": RANDOM_STATE,
}


# ============================================================
# EXPERIMENT RESULTS
# ============================================================

BEST_CV_RMSE = 20619.12327136985
