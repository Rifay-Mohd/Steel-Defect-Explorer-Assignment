"""Verify feature-only validation and inference against the real frozen pipelines."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.config import MODEL_NAMES, ROOT, TRAIN_PATH
from src.inference import predict_record
from src.validation import ValidationError, validate_record, validate_upload


@pytest.fixture
def record(batch):
    """Return a real feature-only record without creating or consulting a target."""
    return batch.features.iloc[0].to_dict()


@pytest.mark.parametrize("model_id", MODEL_NAMES)
def test_real_pipeline_single_row_parity(model_id, registry, record, schema):
    """Match direct hard predictions and every probability for all five artifacts."""
    features = validate_record(record, schema)
    pipeline = registry[model_id]
    result = predict_record(model_id, features, registry)
    expected = pipeline.predict_proba(features)[0]
    assert result["model_id"] == model_id
    assert result["prediction"] == pipeline.predict(features)[0]
    assert isinstance(result["prediction"], str)
    assert result["classes"] == pipeline.classes_.tolist()
    assert isinstance(result["probabilities"], np.ndarray)
    assert result["probabilities"].shape == (7,)
    np.testing.assert_array_equal(result["probabilities"], expected)
    assert result["predicted_probability"] == float(
        expected[result["classes"].index(result["prediction"])]
    )
    assert isinstance(result["predicted_probability"], float)
    assert result["top_probability_class"] == result["classes"][expected.argmax()]
    assert set(result) == {
        "model_id",
        "prediction",
        "classes",
        "probabilities",
        "predicted_probability",
        "top_probability_class",
        "feature_fingerprint",
    }


def test_real_svc_disagreement_preserves_hard_class(registry, batch, schema):
    """Locate a frozen SVC disagreement at runtime and retain the hard decision."""
    pipeline = registry["svm"]
    hard = pipeline.predict(batch.features)
    probabilities = pipeline.predict_proba(batch.features)
    top = pipeline.classes_[probabilities.argmax(axis=1)]
    mismatches = np.flatnonzero(hard != top)
    assert len(mismatches), "Expected a real frozen SVC calibration disagreement."

    # Verify the selected row independently, since single-row calls are the contract.
    position = int(mismatches[0])
    features = validate_record(batch.features.iloc[position].to_dict(), schema)
    result = predict_record("svm", features, registry)
    assert result["prediction"] == str(pipeline.predict(features)[0])
    assert result["prediction"] != result["top_probability_class"]
    assert result["predicted_probability"] < result["probabilities"].max()


def test_record_order_strings_and_fingerprint(record, schema, registry):
    """Canonicalize dictionary order and numeric strings while ignoring row index."""
    original = validate_record(record, schema)
    reverse = {key: str(record[key]) for key in reversed(record)}
    ordered = validate_record(reverse, schema)
    pd.testing.assert_frame_equal(original, ordered)
    assert ordered.columns.tolist() == schema["features"]
    assert ordered.shape == (1, 27)
    first = predict_record("svm", original, registry)
    ordered.index = [-42]
    second = predict_record("svm", ordered, registry)
    assert first["feature_fingerprint"] == second["feature_fingerprint"]
    assert len(bytes.fromhex(first["feature_fingerprint"])) == 32
    ordered.loc[-42, "Pixels_Areas"] += 1
    third = predict_record("svm", ordered, registry)
    assert third["feature_fingerprint"] != first["feature_fingerprint"]


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        None,
        pd.NA,
        np.nan,
        np.inf,
        -np.inf,
        "NaN",
        "inf",
        "",
        " ",
        "nonnumeric",
        "True",
        [],
        [1],
        {},
        1j,
    ],
)
def test_invalid_record_values(record, schema, value):
    """Reject nonnumeric scalars, booleans, containers, and missing/nonfinite data."""
    record["Pixels_Areas"] = value
    with pytest.raises(ValidationError, match="Feature"):
        validate_record(record, schema)


@pytest.mark.parametrize(
    "case", ["missing", "extra", "target", "indicator", "not_dict"]
)
def test_record_requires_exact_features(record, schema, case):
    """Reject missing fields, extras, labels, source indicators, and other inputs."""
    if case == "missing":
        record.pop(schema["features"][0])
    elif case == "not_dict":
        record = [record]
    else:
        name = {
            "extra": "extra",
            "target": schema["target"],
            "indicator": schema["classes"][0],
        }[case]
        record[name] = 0
    with pytest.raises(ValidationError):
        validate_record(record, schema)


@pytest.mark.parametrize(
    "changes",
    [
        {"TypeOfSteel_A300": 0, "TypeOfSteel_A400": 0},
        {"TypeOfSteel_A300": 1, "TypeOfSteel_A400": 1},
        {"TypeOfSteel_A300": 0.5, "TypeOfSteel_A400": 0.5},
        {"TypeOfSteel_A300": -1, "TypeOfSteel_A400": 2},
        {"X_Minimum": 1, "X_Maximum": 1},
        {"X_Minimum": 2, "X_Maximum": 1},
        {"Y_Minimum": 1, "Y_Maximum": 1},
        {"Y_Minimum": 2, "Y_Maximum": 1},
        {"Minimum_of_Luminosity": 2, "Maximum_of_Luminosity": 1},
    ],
)
def test_source_relationships(record, schema, changes):
    """Require binary complementary steel flags and consistent measurement bounds."""
    record.update(changes)
    with pytest.raises(ValidationError):
        validate_record(record, schema)


def test_source_compatible_values_not_training_bounds(record, schema, registry):
    """Allow negative derived indices, equal luminosities, and unseen finite values."""
    training = pd.read_csv(TRAIN_PATH)
    record.update(
        {
            "Orientation_Index": -0.5,
            "Luminosity_Index": -0.5,
            "Minimum_of_Luminosity": 20,
            "Maximum_of_Luminosity": 20,
            "Y_Maximum": float(training.Y_Maximum.max()) + 100,
        }
    )
    features = validate_record(record, schema)
    assert predict_record("svm", features, registry)["prediction"] in schema["classes"]


def test_all_source_rows_satisfy_single_record_constraints(schema):
    """Check requested relationships against every original source measurement row."""
    source = pd.read_csv(ROOT / "data/raw/Faults.NNA", sep=r"\s+", header=None)
    source = source.iloc[:, :27]
    source.columns = schema["features"]
    for record in source.to_dict(orient="records"):
        assert validate_record(record, schema).shape == (1, 27)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "two",
        "reordered",
        "extra",
        "missing",
        "bool",
        "string",
        "nan",
        "inf",
        "complex",
    ],
)
def test_inference_rechecks_feature_contract(record, schema, registry, case):
    """Fail closed when callers bypass validation with malformed DataFrames."""
    features = validate_record(record, schema)
    if case == "empty":
        features = features.iloc[:0]
    elif case == "two":
        features = pd.concat([features, features])
    elif case == "reordered":
        features = features.iloc[:, ::-1]
    elif case == "extra":
        features[schema["target"]] = "unused"
    elif case == "missing":
        features = features.iloc[:, 1:]
    else:
        features["Pixels_Areas"] = {
            "bool": True,
            "string": "1",
            "nan": np.nan,
            "inf": np.inf,
            "complex": 1j,
        }[case]
    with pytest.raises(ValueError):
        predict_record("svm", features, registry)


@pytest.mark.parametrize("model_id", ["unknown", None, [], "SVM"])
def test_unknown_model_rejected(record, schema, registry, model_id):
    """Reject invalid model identifiers before any estimator call."""
    with pytest.raises(ValueError, match="Unknown model"):
        predict_record(model_id, validate_record(record, schema), registry)


@pytest.mark.parametrize("case", ["shape", "nan", "negative", "above_one", "sum"])
def test_invalid_pipeline_probabilities(record, schema, registry, case):
    """Reject malformed probability output without normalization or metric calls."""
    matrix = np.full((1, 7), 1 / 7)
    if case == "shape":
        matrix = matrix[0]
    elif case == "sum":
        matrix *= 0.5
    else:
        matrix[0, 0] = {"nan": np.nan, "negative": -0.1, "above_one": 1.1}[case]
    with (
        patch.object(registry["svm"], "predict_proba", return_value=matrix),
        pytest.raises(ValueError, match="Probabilities"),
    ):
        predict_record("svm", validate_record(record, schema), registry)


def test_invalid_pipeline_class_and_prediction(record, schema, registry):
    """Reject duplicate fitted labels and hard predictions outside the class set."""
    features = validate_record(record, schema)
    classifier = registry["svm"].steps[-1][1]
    with (
        patch.object(classifier, "classes_", np.array(["duplicate"] * 7)),
        pytest.raises(ValueError, match="unique"),
    ):
        predict_record("svm", features, registry)
    with (
        patch.object(registry["svm"], "predict", return_value=np.array(["unknown"])),
        pytest.raises(ValueError, match="known class"),
    ):
        predict_record("svm", features, registry)


def test_bulk_contract_stays_separate(csv_records, schema):
    """Keep physical relationship restrictions exclusive to feature-only validation."""
    from tests.test_validation import encode_records

    # A labeled bulk row retains its existing finite-numeric validation policy.
    csv_records[1][csv_records[0].index("TypeOfSteel_A300")] = "0.5"
    assert validate_upload(encode_records(csv_records[:2]), schema).row_count == 1
