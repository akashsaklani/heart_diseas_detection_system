"""Central configuration: paths, column names, hyper-parameters.

Values mirror the project presentation (80:20 split, random_state=42,
Random Forest with 100 trees).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "heart.csv"          # real UCI file (you provide)
DATA_DEMO = PROJECT_ROOT / "data" / "raw" / "heart_synthetic_demo.csv"  # fake, for demo/tests only
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "random_forest.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

TARGET = "target"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
]

# Human readable meaning of each feature (used by the web UI and docs)
FEATURE_INFO = {
    "age": "Age of the patient (years)",
    "sex": "Sex of the patient (1 = male, 0 = female)",
    "cp": "Type of chest pain (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no)",
    "restecg": "Resting ECG result (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment (0-2)",
}

# Allowed ranges for input validation: (min, max)
FEATURE_RANGES = {
    "age": (1, 120),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (50, 260),
    "chol": (50, 700),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (40, 250),
    "exang": (0, 1),
    "oldpeak": (0.0, 10.0),
    "slope": (0, 2),
}

# Split / model settings (as in the presentation)
TEST_SIZE = 0.2
RANDOM_STATE = 42
RF_PARAMS = {"n_estimators": 100, "random_state": RANDOM_STATE}
CV_FOLDS = 5

DISCLAIMER = (
    "Educational project for preliminary risk assessment only. "
    "It is not a medical diagnosis."
)
