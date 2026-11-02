"""Project-relative paths and the fixed experiment policy."""

from pathlib import Path

# Keep paths independent of the caller's current working directory.
ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42
TARGET = "fault_type"
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "gaussian_nb": "Gaussian Naive Bayes",
    "svm": "SVM",
}

# Use the archive's accompanying variable file, never API column positions.
SOURCE_URL = "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/198/steel+plates+faults"
RAW_DIRECTORY = ROOT / "data" / "raw"
TRAIN_PATH = ROOT / "data" / "processed" / "training_data.csv"
TEST_PATH = ROOT / "test_data.csv"
SCHEMA_PATH = ROOT / "metadata" / "schema.json"
SPLIT_PATH = ROOT / "metadata" / "split_manifest.json"
AUDIT_PATH = ROOT / "metadata" / "data_audit.json"
EXPERIMENT_PATH = ROOT / "metadata" / "experiment.json"
AUC_UNAVAILABLE_MESSAGE = "Seven-class macro AUC is unavailable because this upload does not contain every expected class."
