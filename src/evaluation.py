"""Single metric implementation shared by offline training and application inference."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.config import AUC_UNAVAILABLE_MESSAGE


def evaluate_predictions(y_true, y_pred, probabilities, classes) -> dict:
    """Evaluate seven-class predictions, sorting labels and probability columns together.

    ``classes[j]`` names input probability column j. Macro metrics always include
    all seven labels with ``zero_division=0``. Hard predictions are preserved,
    including when SVC predictions differ from probability argmax. Missing true
    classes make only AUC unavailable, with the documented explanatory message.
    """
    # Validate before computing anything; never normalize or repair probabilities.
    labels = np.asarray(classes)
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if labels.ndim != 1 or len(labels) != 7:
        raise ValueError("classes must contain exactly seven canonical string labels.")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Class labels must be nonempty strings.")
    if len(set(labels)) != 7:
        raise ValueError("Class labels must be unique.")
    if truth.ndim != 1 or predictions.ndim != 1 or len(truth) == 0:
        raise ValueError(
            "True labels and predictions must be nonempty one-dimensional arrays."
        )
    if len(truth) != len(predictions):
        raise ValueError("True labels and predictions must have equal lengths.")
    if not np.isin(truth, labels).all() or not np.isin(predictions, labels).all():
        raise ValueError(
            "True labels and predictions must belong to the canonical classes."
        )
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.shape != (len(truth), 7):
        raise ValueError("Probabilities must have shape (number of rows, 7).")
    if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
        raise ValueError("Probabilities must be finite values between zero and one.")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("Each probability row must sum to one.")

    # sklearn multiclass AUC requires lexicographically ordered probability labels.
    order = np.argsort(labels)
    ordered_classes = labels[order].tolist()
    matrix = matrix[:, order].copy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=ordered_classes, average="macro", zero_division=0
    )
    missing_classes = sorted(set(ordered_classes) - set(truth))
    auc = None
    if not missing_classes:
        auc = float(
            roc_auc_score(
                truth,
                matrix,
                labels=ordered_classes,
                multi_class="ovr",
                average="macro",
            )
        )
    return {
        "metrics": {
            "Accuracy": float(accuracy_score(truth, predictions)),
            "AUC": auc,
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "MCC": float(matthews_corrcoef(truth, predictions)),
        },
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=ordered_classes
        ),
        "report": classification_report(
            truth,
            predictions,
            labels=ordered_classes,
            output_dict=True,
            zero_division=0,
        ),
        "classes": ordered_classes,
        "auc_message": AUC_UNAVAILABLE_MESSAGE if missing_classes else None,
        "predictions": predictions.copy(),
        "probabilities": matrix,
    }
