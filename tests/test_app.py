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
    # Evaluate Batch must come first because the first tab is the default page.
    # The flat list follows the render tree: the nested batch tabs sit under their parent.
    assert [tab.label for tab in app.tabs] == [
        "Evaluate Batch",
        "Evaluate",
        "Compare",
        "Explore",
        "Predict One",
    ]
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
    batch_tab = next(tab for tab in app.tabs if tab.label == "Evaluate Batch")
    assert (
        not batch_tab.button
    )  # Batch actions are absent; single inference remains usable.
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
    explore_tab = next(tab for tab in app.tabs if tab.label == "Explore")
    assert any(
        item.label == "Feature correlation · frozen training data"
        for item in explore_tab.expander
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
    assert all(widget.value is None for widget in app.number_input)
    assert app.button(key="single_load_example").disabled


def test_prefilled_prediction_without_upload_and_submitted_snapshot(
    app_with_upload, schema
):
    """Use actual widgets and verify model/input edits never relabel a saved result."""
    app, _ = app_with_upload
    app.run()
    assert_healthy(app)
    training = pd.read_csv(TRAIN_PATH).iloc[0]
    assert len(app.number_input) == 25
    assert app.selectbox(key="single_model_select").value == "svm"
    assert app.selectbox(key="single_steel").value == "A400"
    for widget in app.number_input:
        feature = widget.key.removeprefix("single_")
        assert widget.value == training[feature]
        assert feature in widget.proto.help
        assert "Observed training range" in widget.proto.help
        assert "Units: not documented" in widget.proto.help
    assert any(
        item.value
        == "Prefilled demonstration measurements from training data; not an independent test"
        for item in app.caption
    )
    app.button(key="single_predict").click().run()
    assert_healthy(app)
    snapshot = app.session_state["single_result"]
    result = snapshot["result"]
    pipeline = app.session_state["resources"]["registry"]["svm"]
    features = pd.DataFrame([snapshot["inputs"]])[schema["features"]]
    np.testing.assert_array_equal(
        features.iloc[0].to_numpy(), training[schema["features"]].to_numpy()
    )
    assert result["prediction"] == pipeline.predict(features)[0]
    np.testing.assert_allclose(
        result["probabilities"], pipeline.predict_proba(features)[0]
    )
    assert (
        result["predicted_probability"]
        == result["probabilities"][result["classes"].index(result["prediction"])]
    )
    assert list(snapshot["inputs"]) == schema["features"]
    assert schema["target"] not in snapshot["inputs"]
    assert not app.metric
    assert "results" not in app.session_state
    app.selectbox(key="single_model_select").select("decision_tree").run()
    app.number_input(key="single_Edges_Index").set_value(0.12345).run()
    assert_healthy(app)
    assert app.session_state["single_result"] is snapshot
    assert any("LAST SUBMITTED / SVM" in item.value for item in app.caption)
    assert any("edits require resubmission" in item.value for item in app.caption)
    # Form changes travel together on submit, not on unrelated forced reruns.
    app.selectbox(key="single_model_select").select("decision_tree")
    app.number_input(key="single_Edges_Index").set_value(0.12345)
    app.button(key="single_predict").click().run()
    assert_healthy(app)
    assert app.session_state["single_result"]["result"]["model_id"] == "decision_tree"
    assert app.session_state["single_result"]["inputs"]["Edges_Index"] == 0.12345
    app.button(key="single_load_example").click().run()
    assert_healthy(app)
    assert app.number_input(key="single_Edges_Index").value == training["Edges_Index"]
    assert "single_result" not in app.session_state


def test_single_prediction_survives_invalid_upload_and_rejects_bad_inputs(
    app_with_upload,
):
    """Upload invalidation is isolated; invalid single submission removes old output."""
    app, holder = app_with_upload
    app.run()
    app.button(key="single_predict").click().run()
    assert_healthy(app)
    snapshot = app.session_state["single_result"]
    set_upload(holder, b"invalid,headers\n1,2\n")
    app.run()
    assert not app.exception
    assert any("CSV rejected" in item.value for item in app.error)
    assert app.session_state["single_result"] is snapshot
    app.selectbox(key="single_model_select").select("knn")
    app.button(key="single_predict").click().run()
    assert not app.exception
    assert app.session_state["single_result"]["result"]["model_id"] == "knn"
    app.number_input(key="single_X_Minimum").set_value(200)
    app.button(key="single_predict").click().run()
    assert not app.exception
    assert "single_result" not in app.session_state
    assert any("Measurements rejected" in item.value for item in app.error)
    assert not any("Predicted category:" in item.value for item in app.subheader)


def test_missing_training_allows_manual_prediction(
    app_with_upload, schema, monkeypatch
):
    """Missing examples produce empty widgets, with manual prediction still enabled."""
    example = pd.read_csv(TRAIN_PATH).iloc[0].to_dict()
    read_bytes = Path.read_bytes

    def missing_training(path):
        """Hide only the training CSV without changing model loading or inference."""
        if path == TRAIN_PATH:
            raise FileNotFoundError(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", missing_training)
    app, _ = app_with_upload
    app.run()
    assert_healthy(app)
    assert all(widget.value is None for widget in app.number_input)
    assert app.selectbox(key="single_steel").value is None
    assert app.button(key="single_load_example").disabled
    app.button(key="single_predict").click().run()
    assert not app.exception
    assert any("Measurements rejected" in item.value for item in app.error)
    for widget in app.number_input:
        feature = widget.key.removeprefix("single_")
        widget.set_value(
            int(example[feature])
            if widget.proto.format == "%d"
            else float(example[feature])
        )
    app.selectbox(key="single_steel").select("A400")
    app.button(key="single_predict").click().run()
    assert_healthy(app)
    assert set(app.session_state["single_result"]["inputs"]) == set(schema["features"])
    app.button(key="single_clear").click().run()
    assert_healthy(app)
    assert all(widget.value is None for widget in app.number_input)
    assert "single_result" not in app.session_state


def test_single_range_warning_is_advisory(app_with_upload):
    """Preserve out-of-range precision beyond display digits and show an advisory."""
    app, _ = app_with_upload
    app.run()
    app.number_input(key="single_Edges_Index").set_value(2.123456789)
    app.button(key="single_predict").click().run()
    assert_healthy(app)
    assert app.session_state["single_result"]["inputs"]["Edges_Index"] == 2.123456789
    assert any(
        "advisory; values were not clamped" in item.value for item in app.warning
    )


def test_prediction_contract_failure_clears_previous_result(
    app_with_upload, monkeypatch
):
    """Handle invalid backend probabilities without retaining a previous prediction."""
    app, _ = app_with_upload
    app.run()
    app.button(key="single_predict").click().run()
    assert_healthy(app)
    assert "single_result" in app.session_state
    pipeline = app.session_state["resources"]["registry"]["svm"]

    def invalid_probabilities(features):
        """Return invalid probabilities through the real inference contract checks."""
        return np.full((len(features), 7), np.nan)

    monkeypatch.setattr(pipeline, "predict_proba", invalid_probabilities)
    app.button(key="single_predict").click().run()
    assert not app.exception
    assert any("Prediction failed" in item.value for item in app.error)
    assert "single_result" not in app.session_state
    assert not any("Predicted category:" in item.value for item in app.subheader)


def test_resource_change_clears_single_result(app_with_upload, monkeypatch):
    """A refreshed experiment invalidates a single result even without an upload."""
    from src import inference

    app, _ = app_with_upload
    app.run()
    app.button(key="single_predict").click().run()
    assert_healthy(app)
    original = inference.get_experiment

    def refreshed_experiment(*args, **kwargs):
        """Change metadata identity while retaining the verified pipeline contract."""
        return {**original(*args, **kwargs), "ui_test_revision": 1}

    monkeypatch.setattr(inference, "get_experiment", refreshed_experiment)
    app.run()
    assert_healthy(app)
    assert "single_result" not in app.session_state


def test_svc_disagreement_preserves_hard_label_and_chart(
    app_with_upload, schema, monkeypatch
):
    """Render a real SVC disagreement and verify all seven chart probabilities."""
    app, _ = app_with_upload
    app.run()
    training = pd.read_csv(TRAIN_PATH)[schema["features"]]
    pipeline = app.session_state["resources"]["registry"]["svm"]
    predictions = pipeline.predict(training)
    probabilities = pipeline.predict_proba(training)
    top_classes = pipeline.classes_[probabilities.argmax(axis=1)]
    position = np.flatnonzero(predictions != top_classes)[0]
    example = training.iloc[position]
    charts = []
    barh = plt.Axes.barh

    def observe_barh(axis, labels, values, *args, **kwargs):
        """Capture the probability chart while rendering its actual bars."""
        charts.append((list(labels), np.asarray(values).copy()))
        return barh(axis, labels, values, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "barh", observe_barh)
    for widget in app.number_input:
        feature = widget.key.removeprefix("single_")
        widget.set_value(
            int(example[feature])
            if widget.proto.format == "%d"
            else float(example[feature])
        )
    app.selectbox(key="single_steel").select(
        "A300" if example["TypeOfSteel_A300"] == 1 else "A400"
    )
    app.button(key="single_predict").click().run()
    assert_healthy(app)
    result = app.session_state["single_result"]["result"]
    assert result["prediction"] == predictions[position]
    assert result["prediction"] != result["top_probability_class"]
    assert any("SVC probability calibration" in item.value for item in app.info)
    assert len(charts) == 1
    assert len(charts[0][0]) == 7
    np.testing.assert_allclose(charts[0][1], probabilities[position])
    assert not app.metric
    assert not plt.get_fignums()


# --- Colour coding: palette integrity, comparison highlight, result card ---------


def _luminance(hex_color):
    """Return WCAG relative luminance for a #rrggbb colour."""
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground, background):
    """Return the WCAG contrast ratio between two colours."""
    light, dark = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def test_palette_covers_exactly_the_schema_classes(schema):
    """The palette keys must equal the metadata classes, so no class is uncoloured."""
    import app as app_module

    assert set(app_module.PALETTE) == set(schema["classes"])
    assert len(schema["classes"]) == 7


def test_palette_colours_are_unique_and_text_is_readable():
    """Fills and inks are distinct per class and ink-on-fill meets 4.5:1 contrast."""
    import app as app_module

    fills = [v["fill"].lower() for v in app_module.PALETTE.values()]
    inks = [v["ink"].lower() for v in app_module.PALETTE.values()]
    assert len(set(fills)) == len(fills)
    assert len(set(inks)) == len(inks)
    for label, colors in app_module.PALETTE.items():
        assert _contrast(colors["ink"], colors["fill"]) >= 4.5, label
        # Default slate body text must also stay readable if placed on a fill.
        assert _contrast(app_module.NAVY, colors["fill"]) >= 4.5, label


def test_best_cell_styles_pick_column_maximum_and_ignore_nan():
    """Highlight the factual maximum per column, including ties and an all-NaN column."""
    import app as app_module

    frame = pd.DataFrame(
        {
            "Accuracy": [0.70, 0.90, 0.80],
            "AUC": [np.nan, np.nan, np.nan],
            "F1": [0.5, 0.6, 0.6],
            "MCC": [-0.2, -0.5, -0.9],
        },
        index=["a", "b", "c"],
    )
    styles = app_module.best_cell_styles(frame)
    marked = {
        (row, column)
        for row in frame.index
        for column in frame.columns
        if styles.loc[row, column]
    }
    # Negative MCC still marks its (least negative) maximum; AUC has none.
    assert marked == {("b", "Accuracy"), ("b", "F1"), ("c", "F1"), ("a", "MCC")}
    # The Styler renders without error and shows N/A for missing AUC.
    rendered = app_module.comparison_styler(frame).to_html()
    assert "N/A" in rendered
    assert "font-weight: 700" in rendered


def test_result_card_colour_follows_prediction_not_top_probability(
    app_with_upload, schema, monkeypatch
):
    """Card uses the hard-label colour; headline equals predict(); both chips on SVC split."""
    import app as app_module

    app, _ = app_with_upload
    app.run()
    assert_healthy(app)
    app.button(key="single_predict").click().run()
    assert_healthy(app)
    result = app.session_state["single_result"]["result"]
    pipeline = app.session_state["resources"]["registry"]["svm"]
    training = pd.read_csv(TRAIN_PATH)[schema["features"]]
    assert result["prediction"] == pipeline.predict(training.iloc[[0]])[0]
    assert any(
        item.value
        == f"Predicted category: {app_module.display_label(result['prediction'])}"
        for item in app.subheader
    )
    cards = [item.value for item in app.markdown if 'class="result-card"' in item.value]
    assert len(cards) == 1
    colors = app_module.PALETTE[result["prediction"]]
    assert f"background:{colors['fill']}" in cards[0]
    assert f"border-color:{colors['ink']}" in cards[0]
    assert f"{result['predicted_probability']:.2%}" in cards[0]
    assert "Model: SVM" in cards[0]


def test_result_card_shows_both_classes_when_prediction_and_top_probability_differ():
    """A disagreement renders two labelled chips; agreement renders only the headline."""
    import app as app_module

    base = {
        "prediction": "Bumps",
        "top_probability_class": "Stains",
        "predicted_probability": 0.31,
    }
    split = app_module.result_card_html(base, "SVM")
    assert split.count("class-chip") == 3  # headline chip plus the two comparison chips
    assert app_module.PALETTE["Stains"]["fill"] in split
    assert "Highest probability" in split
    assert "background:" + app_module.PALETTE["Bumps"]["fill"] in split
    same = app_module.result_card_html(
        {**base, "top_probability_class": "Bumps"}, "SVM"
    )
    assert same.count("class-chip") == 1
    assert "Highest probability" not in same


def test_no_good_bad_threshold_colours_in_stylesheet():
    """Colour is identification only: no red/green status classes in generated CSS."""
    import app as app_module

    css = app_module.build_css().lower()
    for word in ("pass", "fail", "danger", "success"):
        assert word not in css
