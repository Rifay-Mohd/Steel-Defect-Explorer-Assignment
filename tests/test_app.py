"""Exercise the actual app with AppTest and observe chart data and source failures."""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.config import MODEL_NAMES, ROOT, TRAIN_PATH


@pytest.fixture
def app_with_upload(monkeypatch):
    """Supply named BytesIO uploads while running unchanged app.py through runpy."""
    # AppTest runs charts on a worker thread; avoid native macOS GUI figures.
    plt.switch_backend("Agg")
    holder = {"upload": None}

    def mocked_uploader(*args, **kwargs):
        """Return the current upload without replacing validation or session logic."""
        return holder["upload"]

    monkeypatch.setattr(st, "file_uploader", mocked_uploader)
    script = (
        f"import runpy\nrunpy.run_path({str(ROOT / 'app.py')!r}, run_name='__main__')\n"
    )
    app = AppTest.from_string(script, default_timeout=60)
    return app, holder


def set_upload(holder, payload):
    """Replace the mocked upload with a realistic named, binary file-like object."""
    upload = BytesIO(payload)
    upload.name = "inspection.csv"
    holder["upload"] = upload


def assert_healthy(app):
    """Fail on both uncaught exceptions and application-reported backend failures."""
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]


def test_initial_app_requires_explicit_upload(app_with_upload):
    """Initial UI loads real resources but never silently evaluates the example CSV."""
    app, _ = app_with_upload
    app.run()
    assert_healthy(app)
    assert [tab.label for tab in app.tabs] == ["Evaluate", "Compare", "Explore"]
    assert len(app.metric) == 0
    assert "batch" not in app.session_state
    assert "results" not in app.session_state
    assert "comparison_identity" not in app.session_state
    assert len(app.session_state["resources"]["registry"]) == 5
    assert sum("Upload a valid labeled CSV" in item.value for item in app.info) == 3


def test_valid_upload_model_selection_comparison_and_invalid_replacement(
    app_with_upload,
    test_bytes,
    schema,
):
    """Exercise every model, explicit comparison, rerun caching, and invalidation."""
    app, holder = app_with_upload
    set_upload(holder, test_bytes)
    app.run()
    assert_healthy(app)
    assert len(app.metric) == 6
    assert app.session_state["batch"].row_count == 389
    assert len(app.session_state["results"]) == 1
    assert "comparison_identity" not in app.session_state
    run_id = app.session_state["resources"]["run_id"]
    fingerprint = app.session_state["batch"].fingerprint
    for model_id in MODEL_NAMES:
        app.selectbox(key="model_select").select(model_id).run()
        assert_healthy(app)
        result = app.session_state["results"][(fingerprint, model_id, run_id)]
        assert [metric.label for metric in app.metric] == list(result["metrics"])
        assert [metric.value for metric in app.metric] == [
            f"{value:.4f}" for value in result["metrics"].values()
        ]
    assert "comparison_identity" not in app.session_state
    app.button(key="evaluate_all_models").click().run()
    assert_healthy(app)
    assert app.session_state["comparison_identity"] == (fingerprint, run_id)
    assert len(app.session_state["results"]) == 5
    comparison = [
        element.value
        for element in app.dataframe
        if "Accuracy" in element.value.columns
    ]
    assert len(comparison) == 1
    assert comparison[0].index.tolist() == list(MODEL_NAMES.values())
    cached = dict(app.session_state["results"])
    app.selectbox(key="comparison_metric").select("AUC").run()
    assert_healthy(app)
    for key, result in cached.items():
        assert app.session_state["results"][key] is result
    # A new valid upload must drop the old comparison and all old fingerprint keys.
    lines = test_bytes.splitlines(keepends=True)
    set_upload(holder, lines[0] + b"".join(reversed(lines[1:])))
    app.run()
    assert_healthy(app)
    replacement_fingerprint = app.session_state["batch"].fingerprint
    assert replacement_fingerprint != fingerprint
    assert "comparison_identity" not in app.session_state
    assert len(app.session_state["results"]) == 1
    assert all(
        key[0] == replacement_fingerprint for key in app.session_state["results"]
    )
    app.button(key="evaluate_all_models").click().run()
    assert_healthy(app)
    assert len(app.session_state["results"]) == 5
    # Keep the filename unchanged: bytes, rather than filenames, identify uploads.
    set_upload(holder, b"invalid,headers\n1,2\n")
    app.run()
    assert not app.exception
    assert any("CSV rejected" in error.value for error in app.error)
    for key in (
        "batch",
        "results",
        "comparison_identity",
        "confusion_mode",
        "comparison_metric",
        "distribution_feature",
        "scatter_x",
        "scatter_y",
    ):
        assert key not in app.session_state
    assert not app.metric
    assert not app.button
    assert len(app.dataframe) == 1  # Only the sidebar schema remains.
    assert not any("VALIDATED UPLOAD" in caption.value for caption in app.caption)
    set_upload(holder, test_bytes)
    app.run()
    assert_healthy(app)
    assert app.session_state["batch"].features.columns.tolist() == schema["features"]
    assert len(app.session_state["results"]) == 1
    holder["upload"] = None
    app.run()
    assert_healthy(app)
    assert "batch" not in app.session_state
    assert "results" not in app.session_state
    assert not app.metric


# Source isolation: compare the rendered matrix with an independent training oracle.
def test_training_heatmap_is_independent_of_upload(
    app_with_upload, test_bytes, schema, monkeypatch
):
    """Render two distinct uploads and verify the same target-free training matrix."""
    matrices = []
    heatmap = sns.heatmap

    def observe_heatmap(data, **kwargs):
        """Capture labeled feature matrices while preserving actual chart rendering."""
        if isinstance(data, pd.DataFrame):
            matrices.append(data.copy())
        return heatmap(data, **kwargs)

    monkeypatch.setattr(sns, "heatmap", observe_heatmap)
    training = pd.read_csv(TRAIN_PATH)
    expected = np.corrcoef(training[schema["features"]].to_numpy(), rowvar=False)
    app, holder = app_with_upload
    set_upload(holder, test_bytes)
    figures_before = plt.get_fignums()
    app.run()
    assert_healthy(app)
    assert any("Current validated upload" in item.value for item in app.caption)
    assert any(
        "Frozen training data only · 1,552 rows · 27 numeric features" in item.value
        and "fault_type excluded" in item.value
        for item in app.caption
    )
    assert any(
        item.label == "Feature correlation · frozen training data"
        for item in app.tabs[2].expander
    )
    # Change one uploaded feature's sign: an upload-derived correlation would change.
    replacement = pd.read_csv(BytesIO(test_bytes))
    replacement[schema["features"][0]] *= -1
    set_upload(holder, replacement.to_csv(index=False).encode("utf-8"))
    app.run()
    assert_healthy(app)
    assert len(matrices) == 2
    for matrix in matrices:
        assert matrix.shape == (27, 27)
        assert matrix.columns.tolist() == schema["features"]
        assert matrix.index.tolist() == schema["features"]
        assert schema["target"] not in matrix.columns
        np.testing.assert_allclose(matrix.to_numpy(), expected, atol=1e-12)
    assert plt.get_fignums() == figures_before


# Source failures must leave upload evaluation usable and omit the reference chart.
@pytest.mark.parametrize("source_failure", ["missing", "wrong_hash"])
def test_training_source_failure_warns_without_heatmap(
    app_with_upload, test_bytes, monkeypatch, source_failure
):
    """Reject absent or substituted training data without breaking upload diagnostics."""
    read_bytes = Path.read_bytes
    matrices = []
    heatmap = sns.heatmap

    def training_source(path):
        """Simulate a missing training CSV or a valid but incorrect held-out source."""
        if path == TRAIN_PATH:
            if source_failure == "missing":
                raise FileNotFoundError(path)
            return test_bytes
        return read_bytes(path)

    def observe_heatmap(data, **kwargs):
        """Observe reference chart emission without replacing confusion rendering."""
        if isinstance(data, pd.DataFrame):
            matrices.append(data)
        return heatmap(data, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", training_source)
    monkeypatch.setattr(sns, "heatmap", observe_heatmap)
    app, holder = app_with_upload
    set_upload(holder, test_bytes)
    app.run()
    assert_healthy(app)
    assert len(app.metric) == 6
    assert not matrices
    expected = "missing or unreadable" if source_failure == "missing" else "SHA-256"
    assert any(
        "Training correlation unavailable" in item.value and expected in item.value
        for item in app.warning
    )
