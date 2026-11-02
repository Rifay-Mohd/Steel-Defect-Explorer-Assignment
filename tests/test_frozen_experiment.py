"""Check source provenance, disjoint frozen membership, and real inference artifacts."""

from collections import Counter
from hashlib import sha256
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import MODEL_NAMES, ROOT
from src.data_preparation import audit_source
from src.inference import evaluate_model
from src.validation import validate_upload


def test_source_integrity_and_frozen_rows(schema, manifest, experiment):
    """Reconstruct both CSVs from original one-hot source rows and ordered IDs."""
    raw = ROOT / "data/raw"
    paths = {name: raw / name for name in ("Faults.NNA", "Faults27x7_var")}
    with ZipFile(raw / "steel+plates+faults.zip") as archive:
        for name, path in paths.items():
            assert archive.read(name) == path.read_bytes()
            digest = sha256(path.read_bytes()).hexdigest()
            assert digest == schema["source"]["sha256"][name]
            assert digest == manifest["source_sha256"][name]
            assert digest == experiment["dataset_sha256"][name]
    audit = audit_source(paths)
    assert audit.rows.shape == (1941, 28)
    assert audit.features == schema["features"]
    assert len(audit.features) == 27
    assert not set(audit.features) & ({schema["target"]} | set(audit.source_targets))
    assert audit.summary["exact_duplicate_rows"] == 0
    assert audit.summary["identical_feature_rows"] == 0
    assert audit.summary["conflicting_label_rows"] == 0
    assert audit.summary["missing_values"] == 0
    assert audit.summary["disposition"] == "row-level split permitted"
    expected_ids = train_test_split(
        np.arange(1941),
        test_size=0.2,
        random_state=42,
        stratify=audit.rows[schema["target"]],
    )
    memberships = []
    for (name, partition), ids, size in zip(
        ((name, manifest["partitions"][name]) for name in ("training", "test")),
        expected_ids,
        (1552, 389),
    ):
        assert partition["source_row_ids"] == ids.tolist()
        assert partition["rows"] == len(set(ids)) == size
        content = (ROOT / partition["path"]).read_bytes()
        assert sha256(content).hexdigest() == partition["sha256"]
        assert experiment[f"{name}_sha256"] == partition["sha256"]
        reconstructed = audit.rows.iloc[ids]
        assert (
            content == reconstructed.to_csv(index=False, lineterminator="\n").encode()
        )
        assert (
            reconstructed[schema["target"]].value_counts().sort_index().to_dict()
            == partition["class_counts"]
        )
        assert set(reconstructed[schema["target"]]) == set(schema["classes"])
        memberships.append(set(ids))
    assert memberships[0].isdisjoint(memberships[1])
    assert memberships[0] | memberships[1] == set(range(1941))
    assert (
        sha256((ROOT / "metadata/schema.json").read_bytes()).hexdigest()
        == manifest["schema_sha256"]
        == experiment["schema_sha256"]
    )
    assert (
        sha256((ROOT / "metadata/split_manifest.json").read_bytes()).hexdigest()
        == experiment["split_manifest_sha256"]
    )


def test_cv_folds_are_training_only(manifest, experiment, schema):
    """Reproduce every fold and exclude held-out source IDs from selection."""
    train_ids = np.array(manifest["partitions"]["training"]["source_row_ids"])
    test_ids = set(manifest["partitions"]["test"]["source_row_ids"])
    frame = pd.read_csv(ROOT / manifest["partitions"]["training"]["path"])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
        frame[schema["features"]], frame[schema["target"]]
    )
    observed = []
    assert len(experiment["cv"]["folds"]) == 5
    for recorded, (fit, validation) in zip(experiment["cv"]["folds"], folds):
        fit_ids, validation_ids = (
            recorded["training_source_row_ids"],
            recorded["validation_source_row_ids"],
        )
        assert fit_ids == train_ids[fit].tolist()
        assert validation_ids == train_ids[validation].tolist()
        assert set(fit_ids).isdisjoint(validation_ids)
        assert set(fit_ids) | set(validation_ids) == set(train_ids)
        assert test_ids.isdisjoint(fit_ids + validation_ids)
        observed.extend(validation_ids)
    assert Counter(observed) == Counter(train_ids)


@pytest.mark.parametrize("model_id", MODEL_NAMES)
def test_saved_pipeline_predictions_and_offline_metrics(
    model_id,
    registry,
    batch,
    schema,
    manifest,
    experiment,
    offline_predictions,
):
    """Require deterministic real inference matching saved row IDs, metrics, and exports."""
    pipeline = registry[model_id]
    assert pipeline.feature_names_in_.tolist() == schema["features"]
    assert pipeline.n_features_in_ == 27
    assert schema["target"] not in pipeline.feature_names_in_
    if "scaler" in pipeline.named_steps:
        train = pd.read_csv(ROOT / manifest["partitions"]["training"]["path"])
        scaler = pipeline.named_steps["scaler"]
        assert scaler.n_samples_seen_ == 1552
        np.testing.assert_allclose(
            scaler.mean_, train[schema["features"]].mean(), rtol=1e-13, atol=1e-13
        )
    first = evaluate_model(model_id, batch, registry)
    second = evaluate_model(model_id, batch, registry)
    for key in ("predictions", "probabilities", "confusion_matrix"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["metrics"] == second["metrics"]
    saved = offline_predictions.loc[offline_predictions.model_id == model_id]
    assert (
        saved.source_row_id.tolist() == manifest["partitions"]["test"]["source_row_ids"]
    )
    assert saved.run_id.unique().tolist() == [experiment["run_id"]]
    np.testing.assert_array_equal(saved.true_label, batch.target)
    np.testing.assert_array_equal(first["predictions"], saved.predicted_label)
    np.testing.assert_allclose(
        first["probabilities"],
        saved[[f"probability__{c}" for c in schema["classes"]]],
        rtol=1e-12,
        atol=1e-14,
    )
    metrics = pd.read_csv(ROOT / "results/test_metrics.csv").set_index("Model")
    for name, value in first["metrics"].items():
        assert value == pytest.approx(
            metrics.loc[MODEL_NAMES[model_id], name], rel=1e-12, abs=1e-14
        )
    exported = first["predictions_frame"]
    assert exported.row_index.tolist() == list(range(389))
    np.testing.assert_array_equal(exported["true"], batch.target)
    np.testing.assert_array_equal(exported.predicted, pipeline.predict(batch.features))
    assert first["confusion_matrix"].sum() == 389
    assert first["auc_message"] is None


@pytest.mark.parametrize("model_id", MODEL_NAMES)
def test_reordered_upload_is_bit_exact(model_id, registry, batch, test_bytes, schema):
    """Reordering raw columns preserves every feature bit and model output bit."""
    lines = test_bytes.decode().splitlines()
    reversed_csv = (
        "\n".join(",".join(reversed(line.split(","))) for line in lines) + "\n"
    ).encode()
    reordered = validate_upload(reversed_csv, schema)
    np.testing.assert_array_equal(
        batch.features.to_numpy().view(np.uint64),
        reordered.features.to_numpy().view(np.uint64),
    )
    pd.testing.assert_series_equal(batch.target, reordered.target)
    assert batch.fingerprint != reordered.fingerprint
    original = evaluate_model(model_id, batch, registry)
    result = evaluate_model(model_id, reordered, registry)
    np.testing.assert_array_equal(original["predictions"], result["predictions"])
    np.testing.assert_array_equal(original["probabilities"], result["probabilities"])
    assert original["metrics"] == result["metrics"]
