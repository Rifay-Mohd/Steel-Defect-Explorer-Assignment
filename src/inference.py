"""Load verified local pipelines and evaluate uploads without fitting or caching."""

import hashlib
import json
import warnings
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.config import MODEL_NAMES, ROOT
from src.evaluation import evaluate_predictions
from src.validation import ValidatedBatch, _schema_columns


def get_experiment(experiment_path: str | Path | None = None) -> dict:
    """Read fresh experiment metadata for the UI, defaulting to project ROOT."""
    path = (
        Path(experiment_path)
        if experiment_path is not None
        else ROOT / "metadata/experiment.json"
    )
    with path.open(encoding="utf-8") as handle:
        experiment = json.load(handle)
    if not isinstance(experiment, dict):
        raise ValueError("Experiment metadata must be a JSON object.")
    return experiment


def load_artifacts(
    root: str | Path = ROOT,
    *,
    schema_path: str | Path | None = None,
    experiment_path: str | Path | None = None,
    model_dir: str | Path | None = None,
) -> dict[str, Pipeline]:
    """Load the five trusted local pipelines after integrity/contract checks.

    Optional paths refer to trusted backend files, never uploaded artifacts.
    Metadata artifact names must exactly equal ``model/{known_slug}.joblib``.
    All hashes and metadata are checked before any joblib deserialization; the
    same verified bytes are loaded to avoid a second, unchecked file read.
    """
    root = Path(root)
    schema_path = (
        Path(schema_path) if schema_path is not None else root / "metadata/schema.json"
    )
    model_dir = (Path(model_dir) if model_dir is not None else root / "model").resolve()
    experiment = get_experiment(
        experiment_path
        if experiment_path is not None
        else root / "metadata/experiment.json"
    )
    schema_bytes = schema_path.read_bytes()
    schema = json.loads(schema_bytes)
    features, _, classes = _schema_columns(schema)

    # Fail closed on incompatible or incomplete experiment metadata.
    if experiment.get("status") != "complete":
        raise ValueError("Experiment is not complete.")
    if experiment.get("versions", {}).get("sklearn") != sklearn.__version__:
        raise ValueError("Installed scikit-learn version differs from the experiment.")
    if experiment.get("schema_sha256") != hashlib.sha256(schema_bytes).hexdigest():
        raise ValueError("Schema SHA-256 does not match the experiment.")
    if experiment.get("schema_version") != schema.get("schema_version"):
        raise ValueError("Schema version does not match the experiment.")
    if len(classes) != 7 or experiment.get("cv", {}).get("classes") != classes:
        raise ValueError("Experiment classes do not match the seven-class schema.")
    if experiment.get("metric_policy", {}).get("macro_classes") != classes:
        raise ValueError("Metric policy classes do not match the schema.")
    metadata = experiment.get("models")
    if not isinstance(metadata, dict) or set(metadata) != set(MODEL_NAMES):
        raise ValueError(
            "Experiment must contain exactly the five expected model slugs."
        )

    # Never derive a deserialization path from an unchecked metadata value.
    verified = {}
    for slug in MODEL_NAMES:
        entry = metadata[slug]
        if (
            not isinstance(entry, dict)
            or entry.get("artifact") != f"model/{slug}.joblib"
        ):
            raise ValueError(f"Unexpected artifact filename for {slug}.")
        if entry.get("features") != features or entry.get("classes") != classes:
            raise ValueError(f"Model metadata differs from the schema for {slug}.")
        path = model_dir / f"{slug}.joblib"
        if path.resolve().parent != model_dir:
            raise ValueError(f"Artifact path escapes the model directory for {slug}.")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
            raise ValueError(f"Artifact SHA-256 mismatch for {slug}.")
        if len(payload) != entry.get("artifact_bytes"):
            raise ValueError(f"Artifact byte count mismatch for {slug}.")
        verified[slug] = payload

    # Validate fitted pipeline attributes as well as the surrounding metadata.
    registry = {}
    for slug, payload in verified.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            pipeline = joblib.load(BytesIO(payload))
        if not isinstance(pipeline, Pipeline):
            raise ValueError(f"Artifact is not a pipeline for {slug}.")
        check_is_fitted(pipeline)
        if list(getattr(pipeline, "feature_names_in_", [])) != features:
            raise ValueError(f"Fitted feature names differ from the schema for {slug}.")
        if getattr(pipeline, "n_features_in_", None) != len(features):
            raise ValueError(
                f"Fitted feature count differs from the schema for {slug}."
            )
        if list(getattr(pipeline, "classes_", [])) != classes:
            raise ValueError(f"Fitted classes differ from the schema for {slug}.")
        if not callable(getattr(pipeline, "predict_proba", None)):
            raise ValueError(f"Pipeline does not support probabilities for {slug}.")
        registry[slug] = pipeline
    return registry


def evaluate_model(model_id: str, batch: ValidatedBatch, registry: dict) -> dict:
    """Return the shared evaluator result plus model/upload identity and a table.

    ``predictions_frame`` preserves upload order with zero-based ``row_index``,
    ``true``, ``predicted``, and ``probability_{class}`` columns in evaluator
    class order. Hard predictions always come from ``pipeline.predict``.
    """
    if (
        not isinstance(model_id, str)
        or model_id not in MODEL_NAMES
        or model_id not in registry
    ):
        raise ValueError("Unknown model ID.")
    if not isinstance(batch, ValidatedBatch) or batch.row_count == 0:
        raise ValueError("Evaluation requires a nonempty ValidatedBatch.")
    pipeline = registry[model_id]
    if batch.features.columns.tolist() != list(pipeline.feature_names_in_):
        raise ValueError("Batch features do not match the fitted pipeline.")
    if not batch.features.index.equals(batch.target.index):
        raise ValueError("Batch feature and target row indexes must match.")

    # Reuse the offline evaluator, including its probability ordering and AUC policy.
    result = evaluate_predictions(
        batch.target,
        pipeline.predict(batch.features),
        pipeline.predict_proba(batch.features),
        pipeline.classes_,
    )
    predictions_frame = pd.DataFrame(
        {
            "row_index": batch.features.index.to_numpy(copy=True),
            "true": batch.target.to_numpy(copy=True),
            "predicted": result["predictions"],
        }
    )
    for column, label in enumerate(result["classes"]):
        predictions_frame[f"probability_{label}"] = result["probabilities"][:, column]
    return {
        **result,
        "model_id": model_id,
        "batch_fingerprint": batch.fingerprint,
        "predictions_frame": predictions_frame,
    }
