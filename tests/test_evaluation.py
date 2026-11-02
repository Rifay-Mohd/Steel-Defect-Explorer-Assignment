"""Independent seven-class metric oracles and rejection of invalid probabilities."""

import numpy as np
import pytest

from src.config import AUC_UNAVAILABLE_MESSAGE
from src.evaluation import evaluate_predictions


def test_perfect_oracle_aligns_shuffled_probability_columns(schema):
    """A perfect one-hot oracle stays perfect after a nontrivial class permutation."""
    classes = np.array(schema["classes"])
    truth = np.tile(classes, 3)
    perfect = np.tile(np.eye(7), (3, 1))
    order = [4, 0, 6, 2, 5, 1, 3]
    result = evaluate_predictions(truth, truth, perfect[:, order], classes[order])
    assert result["metrics"] == dict.fromkeys(
        ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"), 1.0
    )
    assert result["classes"] == classes.tolist()
    np.testing.assert_array_equal(result["probabilities"], perfect)
    np.testing.assert_array_equal(result["confusion_matrix"], np.eye(7, dtype=int) * 3)
    assert result["auc_message"] is None


@pytest.mark.parametrize("present", [1, 3, 6])
def test_missing_classes_keep_fixed_macro_denominator(schema, present):
    """Perfect observed classes contribute present/7 while missing-class AUC is None."""
    classes = schema["classes"]
    truth = classes[:present] * 2
    probabilities = np.tile(np.eye(7)[:present], (2, 1))
    result = evaluate_predictions(truth, truth, probabilities, classes)
    assert result["metrics"]["AUC"] is None
    assert result["auc_message"] == AUC_UNAVAILABLE_MESSAGE
    assert result["metrics"]["Accuracy"] == 1
    for name in ("Precision", "Recall", "F1"):
        assert result["metrics"][name] == pytest.approx(present / 7)
    assert result["confusion_matrix"].shape == (7, 7)
    assert result["confusion_matrix"].sum() == len(truth)
    for label in classes[present:]:
        assert result["report"][label] == {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0,
        }


def test_hard_predictions_are_not_replaced_by_probability_argmax(schema):
    """Keep deliberately different hard decisions, as required for calibrated SVC."""
    classes = np.array(schema["classes"])
    predictions = np.roll(classes, 1)
    result = evaluate_predictions(classes, predictions, np.eye(7), classes)
    np.testing.assert_array_equal(result["predictions"], predictions)
    assert result["metrics"]["Accuracy"] == 0
    assert result["metrics"]["F1"] == 0
    assert result["metrics"]["AUC"] == 1
    assert result["metrics"]["MCC"] == pytest.approx(-1 / 6)


@pytest.mark.parametrize(
    "case",
    [
        "nan",
        "inf",
        "negative",
        "above_one",
        "zero_sum",
        "bad_sum",
        "six_columns",
        "wrong_rows",
        "one_dimensional",
        "text",
    ],
)
def test_invalid_probabilities_are_rejected(schema, case):
    """Reject nonfinite, out-of-range, unnormalized, malformed, or nonnumeric matrices."""
    probabilities = np.eye(7)
    if case in ("nan", "inf", "negative", "above_one"):
        probabilities[0, 0] = {
            "nan": np.nan,
            "inf": np.inf,
            "negative": -0.1,
            "above_one": 1.1,
        }[case]
    elif case == "zero_sum":
        probabilities[0] = 0
    elif case == "bad_sum":
        probabilities[0, 0] = 0.9
    elif case == "six_columns":
        probabilities = probabilities[:, :6]
    elif case == "wrong_rows":
        probabilities = probabilities[:6]
    elif case == "one_dimensional":
        probabilities = probabilities[0]
    else:
        probabilities = probabilities.astype(object)
        probabilities[0, 0] = "broken"
    with pytest.raises(ValueError):
        evaluate_predictions(
            schema["classes"], schema["classes"], probabilities, schema["classes"]
        )
