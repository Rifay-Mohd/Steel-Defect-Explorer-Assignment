"""Train the five bounded pipeline searches with ``python -m scripts.train_models``."""

from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from uuid import uuid4
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    EXPERIMENT_PATH,
    MODEL_NAMES,
    RANDOM_SEED,
    ROOT,
    SCHEMA_PATH,
    SPLIT_PATH,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)
from src.data_preparation import json_bytes, sha256_bytes, write_immutable_files
from src.evaluation import evaluate_predictions

# Offline figures must not require a graphical desktop or a Streamlit process.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def model_recipes() -> dict:
    """Return the documented pipeline recipes and bounded grids in fixed model order.

    GridSearchCV's deterministic ParameterGrid order breaks equal candidate
    scores by first occurrence. Overall model ties use MODEL_NAMES order and
    are also explicitly retained in experiment metadata.
    """
    estimators = {
        "logistic_regression": LogisticRegression(
            max_iter=5000, random_state=RANDOM_SEED
        ),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        "gaussian_nb": GaussianNB(),
        "svm": SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
    }
    grids = {
        "logistic_regression": {"classifier__C": [0.1, 1, 10]},
        "decision_tree": {
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_leaf": [1, 3],
        },
        "knn": {
            "classifier__n_neighbors": [3, 5, 9],
            "classifier__weights": ["uniform", "distance"],
        },
        "gaussian_nb": {"classifier__var_smoothing": [1e-11, 1e-9, 1e-7]},
        "svm": {"classifier__C": [0.1, 1, 10], "classifier__gamma": ["scale", 0.01]},
    }
    recipes = {}
    for slug, estimator in estimators.items():
        steps = [] if slug == "decision_tree" else [("scaler", StandardScaler())]
        recipes[slug] = (Pipeline(steps + [("classifier", estimator)]), grids[slug])
    return recipes


def summarize_warnings(observed: list) -> list[dict]:
    """Group warnings by category/message/portable location and retain counts."""

    def portable_filename(filename: str) -> str:
        path = Path(filename)
        for marker in ("site-packages", "dist-packages", "sklearn"):
            if marker in path.parts:
                index = path.parts.index(marker)
                start = index if marker == "sklearn" else index + 1
                return Path(*path.parts[start:]).as_posix()
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return path.name

    counts = Counter(
        (
            item.category.__name__,
            str(item.message),
            portable_filename(item.filename),
            item.lineno,
        )
        for item in observed
    )
    return [
        {
            "category": category,
            "message": message,
            "filename": filename,
            "lineno": lineno,
            "count": count,
        }
        for (category, message, filename, lineno), count in counts.items()
    ]


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize full-precision tabular results with stable CSV line endings."""
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def verified_partition(path, partition: dict, schema: dict) -> pd.DataFrame:
    """Reject modified frozen CSVs and verify their ordered feature/label contract."""
    content = path.read_bytes()
    if sha256_bytes(content) != partition["sha256"]:
        raise ValueError(f"Frozen partition hash mismatch: {path.name}")
    frame = pd.read_csv(BytesIO(content))
    if (
        frame.columns.tolist() != schema["features"] + [TARGET]
        or len(frame) != partition["rows"]
    ):
        raise ValueError(f"Frozen partition schema or row count mismatch: {path.name}")
    if not np.isfinite(frame[schema["features"]].to_numpy(dtype=float)).all():
        raise ValueError(f"Frozen partition has nonfinite inputs: {path.name}")
    if sorted(frame[TARGET].unique()) != schema["classes"]:
        raise ValueError(
            f"Frozen partition does not contain every canonical class: {path.name}"
        )
    if frame[TARGET].value_counts().sort_index().to_dict() != partition["class_counts"]:
        raise ValueError(
            f"Frozen partition class counts differ from the manifest: {path.name}"
        )
    return frame


def confusion_png(result: dict, display_name: str) -> bytes:
    """Render a labeled count matrix for the frozen held-out partition."""
    figure, axis = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(
        result["confusion_matrix"], display_labels=result["classes"]
    ).plot(ax=axis, cmap="Blues", values_format="d", xticks_rotation=45, colorbar=False)
    axis.set_title(f"{display_name} — held-out test data (389 rows)")
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=140)
    plt.close(figure)
    return buffer.getvalue()


def verify_fresh_process() -> None:
    """Reload all saved pipelines in a fresh interpreter and compare saved outputs.

    This verifies both hard predictions and every aligned probability column,
    including SVC's separate hard-label decision rule.
    """
    program = """
import joblib
import numpy as np
import pandas as pd
from src.config import ROOT, MODEL_NAMES, TARGET, TEST_PATH
from src.evaluation import evaluate_predictions
test = pd.read_csv(TEST_PATH)
saved = pd.read_csv(ROOT / "results" / "test_predictions.csv")
for slug in MODEL_NAMES:
    pipeline = joblib.load(ROOT / "model" / (slug + ".joblib"))
    x = test.loc[:, pipeline.feature_names_in_]
    result = evaluate_predictions(test[TARGET], pipeline.predict(x), pipeline.predict_proba(x), pipeline.classes_)
    expected = saved.loc[saved["model_id"] == slug]
    np.testing.assert_array_equal(result["predictions"], expected["predicted_label"].to_numpy())
    np.testing.assert_allclose(result["probabilities"], expected[["probability__" + label for label in result["classes"]]].to_numpy(), rtol=1e-12, atol=1e-14)
    assert result["confusion_matrix"].sum() == len(test)
print("Fresh-process reload verified: all five predictions and probability matrices match.")
"""
    subprocess.run([sys.executable, "-c", program], cwd=ROOT, check=True)


def main() -> None:
    """Search only training folds, freeze all choices, then evaluate the held-out set.

    Artifacts are created exclusively and experiment.json is published last as
    the coherent-run completion marker. Existing or partial runs fail clearly
    rather than mixing artifacts from separate experiments.
    """
    results_directory = ROOT / "results"
    model_directory = ROOT / "model"
    if (
        EXPERIMENT_PATH.exists()
        or any(model_directory.glob("*.joblib"))
        or any(results_directory.glob("*.csv"))
    ):
        raise FileExistsError(
            "Existing complete or partial experiment found; refusing to overwrite or mix runs."
        )
    schema_content = SCHEMA_PATH.read_bytes()
    schema = json.loads(schema_content)
    split_content = SPLIT_PATH.read_bytes()
    manifest = json.loads(split_content)
    if manifest["schema_sha256"] != sha256_bytes(schema_content):
        raise ValueError("Schema differs from the frozen split manifest.")
    if (
        manifest["random_seed"] != RANDOM_SEED
        or manifest["audit_disposition"] != "row-level split permitted"
    ):
        raise ValueError(
            "Frozen split policy does not match the approved training recipe."
        )
    features, classes = schema["features"], schema["classes"]
    if (
        schema["target"] != TARGET
        or len(features) != 27
        or len(classes) != 7
        or classes != sorted(classes)
    ):
        raise ValueError(
            "Schema does not match the approved 27-feature/seven-class contract."
        )
    train_ids = manifest["partitions"]["training"]["source_row_ids"]
    test_ids = manifest["partitions"]["test"]["source_row_ids"]
    if (
        len(train_ids) != 1552
        or len(test_ids) != 389
        or any(type(row_id) is not int for row_id in [*train_ids, *test_ids])
        or set(train_ids) & set(test_ids)
        or set(train_ids) | set(test_ids) != set(range(1941))
    ):
        raise ValueError("Frozen split membership is invalid or overlapping.")
    train = verified_partition(TRAIN_PATH, manifest["partitions"]["training"], schema)
    x_train, y_train = train[features], train[TARGET]

    # Materialize identical training-only folds once for all five searches.
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(
            x_train, y_train
        )
    )
    scorer = make_scorer(f1_score, labels=classes, average="macro", zero_division=0)
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    )
    fitted, models, cv_frames = {}, {}, []
    outputs = {}
    for slug, (pipeline, grid) in model_recipes().items():
        started = perf_counter()
        # Capture with "always" so deprecations are counted, persisted, and reported.
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            search = GridSearchCV(
                pipeline,
                grid,
                scoring=scorer,
                cv=folds,
                n_jobs=1,
                refit=True,
                error_score="raise",
                return_train_score=True,
            )
            search.fit(x_train, y_train)
        duration = perf_counter() - started
        warning_records = summarize_warnings(observed)
        for record in warning_records:
            print(
                f"{slug}: {record['category']} x{record['count']}: {record['message']}",
                flush=True,
            )
        best = search.best_estimator_
        if (
            best.classes_.tolist() != classes
            or best.feature_names_in_.tolist() != features
        ):
            raise ValueError(f"Fitted schema/class ordering mismatch for {slug}.")
        fitted[slug] = best
        buffer = BytesIO()
        joblib.dump(best, buffer, compress=3)
        artifact = buffer.getvalue()
        artifact_path = model_directory / f"{slug}.joblib"
        outputs[artifact_path] = artifact
        models[slug] = {
            "display_name": MODEL_NAMES[slug],
            "artifact": str(artifact_path.relative_to(ROOT)),
            "sha256": sha256_bytes(artifact),
            "artifact_bytes": len(artifact),
            "classes": best.classes_.tolist(),
            "features": features,
            "best_params": search.best_params_,
            "cv_macro_f1": float(search.best_score_),
            "duration": duration,
            "duration_seconds": duration,
            "warnings": warning_records,
            "best_candidate_index": int(search.best_index_),
            "parameters": {
                key: value
                for key, value in best.get_params().items()
                if key.startswith("classifier__") or key.startswith("scaler__")
            },
        }
        frame = pd.DataFrame(search.cv_results_)
        frame.insert(0, "candidate_index", np.arange(len(frame)))
        frame.insert(0, "model_id", slug)
        frame.insert(0, "run_id", run_id)
        cv_frames.append(frame)
        print(
            f"{MODEL_NAMES[slug]}: CV macro F1={search.best_score_:.6f}; {duration:.2f}s; {search.best_params_}",
            flush=True,
        )

    # Freeze overall selection before opening or evaluating held-out rows.
    best_score = max(model["cv_macro_f1"] for model in models.values())
    selected_ties = [
        slug for slug in MODEL_NAMES if models[slug]["cv_macro_f1"] == best_score
    ]
    selected = selected_ties[0]
    test = verified_partition(TEST_PATH, manifest["partitions"]["test"], schema)
    x_test, y_test = test[features], test[TARGET]
    metric_rows, prediction_frames = [], []
    for slug, pipeline in fitted.items():
        # Preserve predict output rather than replacing it with probability argmax.
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            result = evaluate_predictions(
                y_test,
                pipeline.predict(x_test),
                pipeline.predict_proba(x_test),
                pipeline.classes_,
            )
        evaluation_warnings = summarize_warnings(observed)
        models[slug]["evaluation_warnings"] = evaluation_warnings
        for record in evaluation_warnings:
            print(
                f"{slug} evaluation: {record['category']} x{record['count']}: {record['message']}",
                flush=True,
            )
        metric_rows.append({"Model": MODEL_NAMES[slug], **result["metrics"]})
        predictions = pd.DataFrame(
            {
                "run_id": run_id,
                "model_id": slug,
                "source_row_id": test_ids,
                "true_label": y_test.to_numpy(),
                "predicted_label": result["predictions"],
            }
        )
        for index, label in enumerate(result["classes"]):
            predictions[f"probability__{label}"] = result["probabilities"][:, index]
        prediction_frames.append(predictions)
        outputs[results_directory / "classification_reports" / f"{slug}.json"] = (
            json_bytes(result["report"])
        )
        outputs[results_directory / "confusion_matrices" / f"{slug}.csv"] = csv_bytes(
            pd.DataFrame(result["confusion_matrix"], columns=classes).assign(
                true_label=classes
            )[["true_label"] + classes]
        )
        outputs[results_directory / "figures" / f"{slug}_confusion_matrix.png"] = (
            confusion_png(result, MODEL_NAMES[slug])
        )

    metrics = pd.DataFrame(
        metric_rows,
        columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
    )
    outputs[results_directory / "test_metrics.csv"] = csv_bytes(metrics)
    outputs[results_directory / "test_predictions.csv"] = csv_bytes(
        pd.concat(prediction_frames, ignore_index=True)
    )
    outputs[results_directory / "cv_results.csv"] = csv_bytes(
        pd.concat(cv_frames, ignore_index=True)
    )
    versions = {
        name: version(distribution)
        for name, distribution in {
            "sklearn": "scikit-learn",
            "numpy": "numpy",
            "scipy": "scipy",
            "pandas": "pandas",
            "streamlit": "streamlit",
            "joblib": "joblib",
            "matplotlib": "matplotlib",
        }.items()
    }
    versions["python"] = platform.python_version()
    experiment = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "complete",
        "random_seed": RANDOM_SEED,
        "schema_sha256": sha256_bytes(schema_content),
        "split_manifest_sha256": sha256_bytes(split_content),
        "dataset_sha256": manifest["source_sha256"],
        "versions": versions,
        "training_sha256": manifest["partitions"]["training"]["sha256"],
        "test_sha256": manifest["partitions"]["test"]["sha256"],
        "test_rows": len(test),
        "models": models,
        "cv_selected_model": selected,
        "cv_selected_model_ties": selected_ties,
        "cv": {
            "method": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": RANDOM_SEED,
            "n_jobs": 1,
            "selection_metric": "macro F1",
            "classes": classes,
            "zero_division": 0,
            "candidate_tie_policy": "first candidate in sklearn ParameterGrid order",
            "model_tie_policy": "first in MODEL_NAMES insertion order; all exact ties recorded",
            "folds": [
                {
                    "training_source_row_ids": [train_ids[i] for i in fit_ids],
                    "validation_source_row_ids": [train_ids[i] for i in validation_ids],
                }
                for fit_ids, validation_ids in folds
            ],
        },
        "metric_policy": {
            "macro_classes": classes,
            "zero_division": 0,
            "auc": "macro OVR",
            "hard_predictions": "pipeline.predict, never probability argmax",
        },
        "svm_probability_recipe": "StandardScaler -> SVC(kernel='rbf', probability=True, random_state=42); built-in calibration retained",
        "outputs": {
            str(path.relative_to(ROOT)): sha256_bytes(content)
            for path, content in outputs.items()
        },
        "fresh_process_reload_verified": True,
    }

    # Publish the completion marker only after artifacts pass independent reload.
    write_immutable_files(outputs)
    verify_fresh_process()
    write_immutable_files({EXPERIMENT_PATH: json_bytes(experiment)})
    print(f"Run {run_id}; CV-selected model: {MODEL_NAMES[selected]}", flush=True)
    print(metrics.to_string(index=False, float_format="%.6f"), flush=True)


if __name__ == "__main__":
    main()
