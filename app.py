"""Session-local, inference-only manufacturing dashboard for steel fault labels."""

import hashlib
import html
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit.typing import UploadedFile

from src.config import MODEL_NAMES, ROOT, SCHEMA_PATH, SPLIT_PATH, TEST_PATH, TRAIN_PATH
from src.inference import evaluate_model, get_experiment, load_artifacts, predict_record
from src.validation import (
    MAX_ROWS,
    MAX_UPLOAD_BYTES,
    ValidatedBatch,
    ValidationError,
    validate_record,
    validate_upload,
)

# Fixed presentation vocabulary; CSVs and inference retain canonical source labels.
METRICS = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")
DISPLAY_LABELS = {
    "Bumps": "Bumps",
    "Dirtiness": "Dirtiness",
    "K_Scatch": "K Scratch",
    "Other_Faults": "Other Faults",
    "Pastry": "Pastry",
    "Stains": "Stains",
    "Z_Scratch": "Z Scratch",
}
NAVY = "#293747"
PAPER = "#fafcfb"

# Single source of truth for class colour: pastel "fill" for backgrounds and a darker
# "ink" for text, borders and small chart marks. Ink on fill is >= 4.5:1 (tested).
# Colour is always paired with the class name, never the only signal.
PALETTE = {
    "Bumps": {"fill": "#cfe3f7", "ink": "#1f4e79"},
    "Dirtiness": {"fill": "#fbd9c4", "ink": "#8a3b12"},
    "K_Scatch": {"fill": "#e0d4f5", "ink": "#4a2f86"},
    "Other_Faults": {"fill": "#dcdfe4", "ink": "#3d4655"},
    "Pastry": {"fill": "#f8d4e4", "ink": "#8a2753"},
    "Stains": {"fill": "#d2efd9", "ink": "#1f6b3a"},
    "Z_Scratch": {"fill": "#f6efb8", "ink": "#6b5a00"},
}
# Metric identity accents: pure identification, deliberately not a quality scale.
METRIC_ACCENTS = {
    "Accuracy": "#b8d8f0",
    "AUC": "#d6c8f0",
    "Precision": "#f3c9dc",
    "Recall": "#f9dcc0",
    "F1": "#c6e8d0",
    "MCC": "#f2e8a8",
}
# One subtle tint per Predict One form section, for scannability only.
SECTION_TINTS = ("#e3f1ea", "#e4edf9", "#f1e8f8", "#fbf0dc", "#e6f0f2")
# Highest value per comparison column; neutral blue, not a good/bad signal.
BEST_CELL_STYLE = "background-color: #cfe3f7; color: #1f4e79; font-weight: 700"

# Source integer measurements precede the derived indices; steel uses one selector.
INTEGER_FEATURES = {
    "X_Minimum",
    "X_Maximum",
    "Y_Minimum",
    "Y_Maximum",
    "Pixels_Areas",
    "X_Perimeter",
    "Y_Perimeter",
    "Sum_of_Luminosity",
    "Minimum_of_Luminosity",
    "Maximum_of_Luminosity",
    "Length_of_Conveyer",
    "Steel_Plate_Thickness",
}
FEATURE_GROUPS = {
    "01 / Position & extent": ("X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum"),
    "02 / Area & perimeter": ("Pixels_Areas", "X_Perimeter", "Y_Perimeter"),
    "03 / Luminosity": (
        "Sum_of_Luminosity",
        "Minimum_of_Luminosity",
        "Maximum_of_Luminosity",
        "Luminosity_Index",
    ),
    "04 / Steel & conveyer": ("Length_of_Conveyer", "Steel_Plate_Thickness"),
    "05 / Shape & derived indices": (
        "Edges_Index",
        "Empty_Index",
        "Square_Index",
        "Outside_X_Index",
        "Edges_X_Index",
        "Edges_Y_Index",
        "Outside_Global_Index",
        "LogOfAreas",
        "Log_X_Index",
        "Log_Y_Index",
        "Orientation_Index",
        "SigmoidOfAreas",
    ),
}


def display_label(label: str) -> str:
    """Return a readable category name without changing its source identifier."""
    return DISPLAY_LABELS.get(label, label.replace("_", " "))


def class_style(label: str) -> dict:
    """Return the pastel fill and dark ink for a canonical class label."""
    return PALETTE.get(label, {"fill": "#e8ecef", "ink": NAVY})


def class_chip(label: str, extra: str = "") -> str:
    """Return an escaped HTML chip: palette colours plus the always-visible class name."""
    colors = class_style(label)
    return (
        f'<span class="class-chip" style="background:{colors["fill"]};'
        f'color:{colors["ink"]};border-color:{colors["ink"]}">'
        f"{html.escape(display_label(label))}{html.escape(extra)}</span>"
    )


def build_css() -> str:
    """Compose the scoped stylesheet, generating per-key rules from the palettes."""
    metric_rules = "".join(
        f".st-key-metric_card_{metric} {{ background: {color}55; "
        f"border-top: 4px solid {color}; }}\n"
        for metric, color in METRIC_ACCENTS.items()
    )
    return (
        """
        :root { --radius: 14px; --shadow: 0 1px 3px rgba(41,55,71,.10), 0 4px 12px rgba(41,55,71,.06); }
        .block-container { max-width: 1500px; padding-top: 4.5rem; padding-bottom: 3rem; }
        .control-header { background: linear-gradient(110deg, #d8f1e6 0%, #e4dcf5 52%, #d6e8f7 100%);
          border-radius: var(--radius); box-shadow: var(--shadow);
          padding: 1rem 1.3rem; margin-bottom: .8rem; color: #293747; }
        .eyebrow { font: 700 .75rem ui-monospace, monospace; letter-spacing: .16em;
          color: #0f5f56; text-transform: uppercase; }
        .control-header h1 { margin: .25rem 0; font-size: clamp(1.7rem, 3vw, 2.3rem); }
        .control-header p { max-width: 1000px; margin: .3rem 0; line-height: 1.5; }
        .category-strip { display: flex; flex-wrap: wrap; gap: .45rem; margin: .7rem 0 1rem; }
        .class-chip { display: inline-block; padding: .22rem .65rem; border-radius: 999px;
          border: 1px solid; font-size: .82rem; font-weight: 650; white-space: nowrap; }
        .section-heading { margin: 1rem 0 .4rem; padding: .35rem .8rem; border-radius: 10px;
          font-size: 1.15rem; font-weight: 700; color: #293747; }
        .result-card { border-radius: var(--radius); border: 2px solid; padding: .9rem 1.1rem;
          margin-bottom: .6rem; box-shadow: var(--shadow); overflow-wrap: anywhere; }
        .result-card .result-name { font-size: 1.25rem; margin: .1rem 0 .3rem; }
        .result-card .result-meta { font-size: .95rem; margin: .1rem 0; }
        .result-card .class-chip { margin: .1rem .3rem .1rem 0; }
        .legend-note { font-size: .85rem; color: #4a5766; }
        .st-key-single_form, .st-key-single_result { border-radius: var(--radius); box-shadow: var(--shadow); }
        .st-key-single_form { background: #f0f7f4; }
        .st-key-single_result { background: #fbfaff; }
        [class*="st-key-metric_card_"] { border-radius: var(--radius); box-shadow: var(--shadow);
          padding: .6rem .8rem; }
        [class*="st-key-metric_card_"] label, [class*="st-key-metric_card_"] p { white-space: normal; }
        [data-testid="stSidebar"] { border-right: 1px solid #d5e3dd; }
        .stButton > button, .stDownloadButton > button { border-radius: 10px; transition: box-shadow .15s, transform .15s; }
        .stButton > button:hover, .stDownloadButton > button:hover { box-shadow: var(--shadow); transform: translateY(-1px); }
        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
        input:focus-visible, [role="combobox"]:focus-within { outline: 3px solid #168578; outline-offset: 2px; }
        @media (max-width: 640px) {
          .block-container { padding-left: .8rem; padding-right: .8rem; }
          .result-card .result-name { font-size: 1.2rem; }
          .class-chip { white-space: normal; }
        }
        """
        + metric_rules
    )


def style_page() -> None:
    """Apply scoped local CSS for a pastel, accessible manufacturing control panel."""
    st.markdown(f"<style>{build_css()}</style>", unsafe_allow_html=True)


def best_cell_styles(frame: pd.DataFrame) -> pd.DataFrame:
    """Return CSS per cell marking the maximum of every column (all ties; NaN ignored)."""
    styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            styles.loc[values == values.max(), column] = BEST_CELL_STYLE
    return styles


def comparison_styler(frame: pd.DataFrame):
    """Format the comparison table and highlight the highest value in each column."""
    return frame.style.format("{:.4f}", na_rep="N/A").apply(best_cell_styles, axis=None)


def class_cell_styler(frame: pd.DataFrame, column: str, styler=None):
    """Colour display-label cells of one column with their class palette."""
    by_display = {display_label(k): v for k, v in PALETTE.items()}

    def paint(values: pd.Series) -> list[str]:
        """Return palette CSS for known class names and nothing for other rows."""
        return [
            f"background-color: {by_display[v]['fill']}; color: {by_display[v]['ink']}; "
            "font-weight: 650"
            if v in by_display
            else ""
            for v in values
        ]

    return (styler if styler is not None else frame.style).apply(paint, subset=[column])


def result_card_html(result: dict, model_name: str) -> str:
    """Build the prediction card coloured by the hard predicted class, showing both
    the predicted class and the highest-probability class when they differ."""
    colors = class_style(result["prediction"])
    top = result["top_probability_class"]
    parts = [
        f'<div class="result-card" style="background:{colors["fill"]};'
        f'border-color:{colors["ink"]};color:{colors["ink"]}">',
        f'<div class="result-name">{class_chip(result["prediction"])}</div>',
        f'<div class="result-meta">Model: {html.escape(model_name)} · '
        f"probability for predicted category: <b>{result['predicted_probability']:.2%}</b></div>",
    ]
    if result["prediction"] != top:
        parts.append(
            '<div class="result-meta">Hard prediction '
            + class_chip(result["prediction"])
            + " · Highest probability "
            + class_chip(top)
            + "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


# Session lifecycle: invalidate derived data before validating replacement bytes.
def reset_upload_state() -> None:
    """Discard upload-derived state before accepting a new or removed file."""
    for key in (
        "upload_identity",
        "batch",
        "results",
        "comparison_identity",
        "confusion_mode",
        "distribution_feature",
        "scatter_x",
        "scatter_y",
    ):
        st.session_state.pop(key, None)


def session_resources() -> dict:
    """Load verified pipelines into this session and refresh on metadata changes."""
    experiment = get_experiment()
    run_id = experiment.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Experiment metadata requires a nonempty run_id.")
    schema_bytes = SCHEMA_PATH.read_bytes()
    # Include metadata content so an edited manifest cannot reuse stale resources.
    identity = (
        run_id,
        hashlib.sha256(schema_bytes).hexdigest(),
        hashlib.sha256(
            json.dumps(experiment, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    )
    resources = st.session_state.get("resources")
    if resources is None or resources["identity"] != identity:
        reset_upload_state()
        st.session_state.pop("single_result", None)
        st.session_state.pop("resources", None)
        registry = load_artifacts(root=ROOT)
        resources = {
            "identity": identity,
            "run_id": run_id,
            "schema": json.loads(schema_bytes),
            "registry": registry,
        }
        st.session_state["resources"] = resources
    return resources


# Input workspace: trusted example download, raw upload, and explicit schema.
def render_sidebar(schema: dict | None) -> tuple[UploadedFile | None, str]:
    """Render input controls, the downloadable example, and the schema contract."""
    with st.sidebar:
        st.markdown("### Inspection workspace")
        st.caption("EVALUATE BATCH / LABELED CSV")
        st.caption(
            "Predict One works without an upload. These controls are for batch evaluation."
        )
        try:
            st.download_button(
                "Download labeled test CSV",
                TEST_PATH.read_bytes(),
                file_name="steel_defect_test_data.csv",
                mime="text/csv; charset=utf-8",
                key="download_test_csv",
                width="stretch",
            )
        except OSError:
            st.warning(
                "Example CSV is unavailable. Restore test_data.csv to enable download."
            )
        st.caption("The download is an example. Upload it explicitly to evaluate it.")
        uploaded = st.file_uploader(
            "Upload labeled test CSV (max 5 MiB)",
            type=["csv"],
            max_upload_size=MAX_UPLOAD_BYTES // (1024 * 1024),
            key="test_csv_upload",
            on_change=reset_upload_state,
            help="UTF-8 CSV: 27 numeric features plus fault_type; at most 10,000 rows.",
        )
        with st.expander("CSV schema & source labels"):
            st.markdown(
                "**27 finite numeric features + `fault_type`.** Column order may vary. "
                "Missing values, duplicate headers, extra columns, and unknown labels are rejected. "
                f"Limit: 5 MiB / {MAX_ROWS:,} rows. Labels must match exactly."
            )
            if schema is not None:
                st.code(
                    ",".join([*schema["features"], schema["target"]]), language="text"
                )
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Source label (CSV)": schema["classes"],
                            "Display label": [
                                display_label(c) for c in schema["classes"]
                            ],
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
            st.caption(
                "Explicit spelling mapping: K_Scatch → K Scratch. Downloads retain K_Scatch."
            )
        st.divider()
        st.caption("02 / SAVED CLASSIFIER")
        model_id = st.selectbox(
            "Select model",
            options=list(MODEL_NAMES),
            format_func=MODEL_NAMES.get,
            key="model_select",
        )
        st.caption("Verified saved pipelines · inference only")
    return uploaded, model_id


def current_batch(uploaded: UploadedFile | None, schema: dict) -> ValidatedBatch | None:
    """Validate original upload bytes on every rerun, clearing stale state first."""
    raw = None if uploaded is None else uploaded.getvalue()
    identity = None if raw is None else hashlib.sha256(raw).hexdigest()
    if st.session_state.get("upload_identity") != identity:
        reset_upload_state()
        st.session_state["upload_identity"] = identity
    if raw is None:
        return None
    try:
        batch = validate_upload(raw, schema)
    except ValidationError as exc:
        reset_upload_state()
        st.session_state["upload_identity"] = identity
        st.error(f"CSV rejected: {exc}")
        st.info(
            "Correct the CSV using the sidebar schema, then upload it again. No batch results are retained."
        )
        return None
    st.session_state["batch"] = batch
    return batch


def result_for(model_id: str, batch: ValidatedBatch, resources: dict) -> dict:
    """Reuse a result only for the exact upload, model, and verified experiment."""
    key = (batch.fingerprint, model_id, resources["run_id"])
    results = st.session_state.setdefault("results", {})
    if key not in results:
        results[key] = evaluate_model(model_id, batch, resources["registry"])
    return results[key]


# Shared plotting primitives keep styling local and release every figure.
def new_chart(figsize: tuple = (9, 5)):
    """Create a locally styled chart without mutating global matplotlib settings."""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    fig.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.tick_params(colors=NAVY, labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#cbd6d1")
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    return fig, ax


def show_chart(fig) -> None:
    """Render a matplotlib figure and always release its associated resources."""
    try:
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)


def render_confusion(result: dict) -> None:
    """Show counts or zero-safe row-normalized percentages in canonical order."""
    mode = st.radio(
        "Confusion matrix values",
        ["Counts", "Row-normalized (%)"],
        horizontal=True,
        key="confusion_mode",
    )
    counts = np.asarray(result["confusion_matrix"])
    normalized = mode != "Counts"
    values = counts
    if normalized:
        totals = counts.sum(axis=1, keepdims=True)
        values = np.divide(
            counts * 100.0,
            totals,
            out=np.zeros_like(counts, dtype=float),
            where=totals != 0,
        )
    labels = [display_label(c) for c in result["classes"]]
    fig, ax = new_chart((8, 6))
    try:
        sns.heatmap(
            values,
            annot=True,
            fmt=".1f" if normalized else "d",
            cmap="GnBu",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            vmin=0,
            vmax=100 if normalized else None,
            cbar_kws={"label": "% of true class" if normalized else "Records"},
        )
        ax.set(
            title="Prediction agreement",
            xlabel="Predicted category",
            ylabel="True category",
        )
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        # Tick text takes each class's dark ink so classes match the rest of the app.
        for tick, label in zip(ax.get_xticklabels(), result["classes"]):
            tick.set_color(class_style(label)["ink"])
        for tick, label in zip(ax.get_yticklabels(), result["classes"]):
            tick.set_color(class_style(label)["ink"])
        show_chart(fig)
    finally:
        plt.close(fig)
    st.caption(
        "Source: current upload. Rows without true examples remain zero when normalized."
    )


def report_frame(result: dict) -> pd.DataFrame:
    """Build a readable class report while keeping accuracy out of support columns."""
    rows = []
    for label, scores in result["report"].items():
        if not isinstance(scores, dict):
            continue
        rows.append(
            {
                "Category": display_label(label),
                "Precision": scores["precision"],
                "Recall": scores["recall"],
                "F1": scores["f1-score"],
                "Support": int(scores["support"]),
            }
        )
    return pd.DataFrame(rows)


# Evaluation uses the backend contract exclusively; presentation only rounds values.
def render_evaluate(batch: ValidatedBatch, model_id: str, resources: dict) -> None:
    """Present shared metrics, class diagnostics, and unchanged prediction exports."""
    st.subheader(MODEL_NAMES[model_id])
    st.caption(
        "CURRENT UPLOAD / Macro precision, recall and F1 include all seven classes; zero division = 0."
    )
    try:
        with st.spinner("Evaluating saved pipeline…"):
            result = result_for(model_id, batch, resources)
    except Exception:
        st.error(
            "Evaluation failed. Check that the local pipelines and metadata belong to the same complete experiment, then reload the app."
        )
        return
    for column, metric in zip(st.columns(6), METRICS):
        value = result["metrics"][metric]
        # Per-metric accent identifies the metric only; it never signals pass or fail.
        with column.container(key=f"metric_card_{metric}"):
            st.metric(metric, "N/A" if value is None else f"{value:.4f}")
    if result["auc_message"]:
        st.warning(result["auc_message"])
    st.caption("AUC: seven-class one-vs-rest macro ROC AUC. MCC ranges from −1 to +1.")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("#### Confusion matrix")
        render_confusion(result)
    with right:
        st.markdown("#### Classification report")
        st.dataframe(
            class_cell_styler(
                report_frame(result),
                "Category",
                report_frame(result).style.format(
                    {
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1": "{:.4f}",
                        "Support": "{:d}",
                    }
                ),
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "K Scratch is displayed from source label K_Scatch. Support is the number of true records per class."
        )
        st.download_button(
            "Download predictions CSV",
            result["predictions_frame"].to_csv(index=False).encode("utf-8"),
            file_name=f"steel_predictions_{model_id}.csv",
            mime="text/csv; charset=utf-8",
            key="download_predictions",
            width="stretch",
        )
        st.caption(
            "Original upload order · zero-based row_index · canonical labels · seven class probabilities"
        )


# One grouped chart: models on the x-axis, one pastel bar per metric within each group.
def metric_comparison_figure(frame: pd.DataFrame):
    """Draw all six metrics for every model in a single grouped bar chart.

    ``frame`` is indexed by model display name with one column per metric. Metric colors
    come from METRIC_ACCENTS only to identify a metric, never to rate it. An unavailable
    (NaN) value leaves a gap rather than a zero-height bar, so a missing AUC is never
    mistaken for a score of zero.
    """
    metrics = list(METRICS)
    group_width = 0.84
    bar_width = group_width / len(metrics)
    fig, ax = new_chart((12, 5.4))
    for position, metric in enumerate(metrics):
        # Center the group of bars on each model's tick.
        offsets = np.arange(len(frame)) - group_width / 2 + bar_width * (position + 0.5)
        values = frame[metric].to_numpy(dtype=float)
        bars = ax.bar(
            offsets,
            np.nan_to_num(values, nan=0.0),
            width=bar_width,
            color=METRIC_ACCENTS[metric],
            edgecolor=NAVY,
            linewidth=0.6,
            label=metric,
        )
        for bar, value in zip(bars, values):
            if np.isnan(value):
                bar.set_visible(False)
                continue
            # MCC can be negative: place its label on the outer side of the bar.
            ax.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3 if value >= 0 else -3),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                rotation=90,
                fontsize=7.5,
                color=NAVY,
            )
    lowest = np.nanmin(frame[metrics].to_numpy(dtype=float))
    ax.set_ylim(min(lowest - 0.15, 0.0), 1.18)
    ax.axhline(0, color=NAVY, linewidth=0.8)
    ax.set_xticks(
        range(len(frame)), [name.replace(" ", "\n", 1) for name in frame.index]
    )
    ax.set(title="All metrics · five saved classifiers", ylabel="Score", xlabel="Model")
    ax.legend(
        title="Metric",
        ncols=len(metrics),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
    )
    return fig


# Comparison activation is bound to the uploaded bytes and experiment run.
def render_compare(batch: ValidatedBatch, resources: dict) -> None:
    """Evaluate five models explicitly and retain comparison across metric reruns."""
    st.subheader("Same batch. Five perspectives.")
    st.caption(
        "All values are computed on the current upload with the same seven-class metric policy."
    )
    identity = (batch.fingerprint, resources["run_id"])
    if st.button("Evaluate all five models", key="evaluate_all_models", type="primary"):
        st.session_state.pop("comparison_identity", None)
        try:
            with st.spinner("Evaluating all five saved pipelines…"):
                for model_id in MODEL_NAMES:
                    result_for(model_id, batch, resources)
        except Exception:
            st.error(
                "Comparison failed. Verify all five local model artifacts against metadata/experiment.json and reload the app."
            )
        else:
            st.session_state["comparison_identity"] = identity
    if st.session_state.get("comparison_identity") != identity:
        st.info(
            "Select Evaluate all five models to build a comparison for this upload."
        )
        return
    results = [result_for(model_id, batch, resources) for model_id in MODEL_NAMES]
    frame = pd.DataFrame(
        [
            {"Model": MODEL_NAMES[result["model_id"]], **result["metrics"]}
            for result in results
        ]
    ).set_index("Model")
    st.dataframe(comparison_styler(frame), width="stretch")
    st.caption(
        "Blue cells mark the highest value in each column on the current upload. "
        "This is a factual maximum, not a quality guarantee."
    )
    if results[0]["auc_message"]:
        st.warning(results[0]["auc_message"])
    if frame["AUC"].isna().all():
        st.info("AUC is unavailable for this upload, so its bars are omitted.")
    show_chart(metric_comparison_figure(frame))
    st.caption(
        "Source: current upload. Each color is one metric, identified in the legend. "
        "This comparison does not select or retrain models."
    )


# Frozen-training reference: verify source bytes before computing feature correlations.
def verified_training(schema: dict) -> ValidatedBatch:
    """Read and validate the frozen training source against its split manifest."""
    raw = TRAIN_PATH.read_bytes()
    training = json.loads(SPLIT_PATH.read_bytes())["partitions"]["training"]
    if hashlib.sha256(raw).hexdigest() != training["sha256"]:
        raise ValueError("Training CSV SHA-256 does not match the split manifest.")
    batch = validate_upload(raw, schema)
    if batch.row_count != training["rows"]:
        raise ValueError("Training row count does not match the split manifest.")
    return batch


def render_training_correlation(schema: dict) -> None:
    """Plot signed Pearson correlations of verified training features, excluding target.

    Read project-relative configured paths on each rerun, check the frozen split
    fingerprint, and reuse CSV validation. Never use uploaded rows or fit a model.
    """
    with st.expander("Feature correlation · frozen training data", expanded=False):
        try:
            batch = verified_training(schema)
        except OSError:
            st.warning(
                "Training correlation unavailable: the frozen training CSV or split manifest "
                "is missing or unreadable. Restore data/processed/training_data.csv and "
                "metadata/split_manifest.json."
            )
            return
        except (ValueError, KeyError, TypeError) as exc:
            st.warning(f"Training correlation unavailable: {exc}")
            return
        st.caption(
            f"SOURCE / Frozen training data only · {batch.row_count:,} rows · "
            f"{len(batch.features.columns)} numeric features · {schema['target']} excluded. "
            "SHA-256 verified against the split manifest. Independent of the current upload."
        )
        correlation = batch.features.corr(method="pearson")
        fig, ax = new_chart((16, 14))
        try:
            sns.heatmap(
                correlation,
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                center=0,
                annot=False,
                square=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={"label": "Pearson r", "ticks": [-1, -0.5, 0, 0.5, 1]},
                ax=ax,
            )
            ax.set(
                title="Feature correlation · frozen training data", xlabel="", ylabel=""
            )
            plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=10)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
            show_chart(fig)
        finally:
            plt.close(fig)
        st.caption(
            "Pearson r: −1 = inverse linear relationship; 0 = none; +1 = direct linear relationship."
        )


# Explore separates current-upload measurements from the frozen-training reference.
def render_explore(batch: ValidatedBatch, schema: dict) -> None:
    """Explore uploaded measurements, then show a separate training-only reference."""
    st.subheader("Understand the inspection batch")
    st.caption(
        "SOURCE / Current validated upload · class composition, feature distribution, and "
        "two-feature inspection below include all uploaded records."
    )
    classes = schema["classes"]
    labels = [display_label(c) for c in classes]
    palette = {display_label(k): v["ink"] for k, v in PALETTE.items()}
    counts = batch.target.value_counts().reindex(classes, fill_value=0)
    summary = pd.DataFrame(
        {
            "Category": labels,
            "Records": counts.to_numpy(),
            "Percent": counts.to_numpy() / batch.row_count * 100,
        }
    )
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.markdown("#### Class composition")
        st.dataframe(
            class_cell_styler(
                summary, "Category", summary.style.format({"Percent": "{:.2f}%"})
            ),
            hide_index=True,
            width="stretch",
        )
    with right:
        fig, ax = new_chart((8, 4))
        try:
            ax.barh(
                labels,
                counts.to_numpy(),
                color=[class_style(c)["fill"] for c in classes],
                edgecolor=[class_style(c)["ink"] for c in classes],
            )
            ax.invert_yaxis()
            ax.set(
                title="Records by defect category", xlabel="Records", ylabel="Category"
            )
            show_chart(fig)
        finally:
            plt.close(fig)
    st.markdown("#### Feature distribution by category")
    feature = st.selectbox(
        "Distribution feature", schema["features"], key="distribution_feature"
    )
    plot_data = batch.features.copy()
    plot_data["Category"] = batch.target.map(display_label)
    fig, ax = new_chart((11, 4.5))
    try:
        # Boxplots handle single-record classes and constant-valued features.
        # Use Matplotlib directly: seaborn 0.13 passes the deprecated vert argument.
        box = ax.boxplot(
            [
                batch.features.loc[batch.target == label, feature].to_numpy()
                for label in classes
            ],
            positions=range(len(classes)),
            tick_labels=labels,
            orientation="vertical",
            widths=0.8,
            whis=1.5,
            patch_artist=True,
            boxprops={
                "facecolor": "#e8ecef",
                "edgecolor": "#707070",
            },
            medianprops={"color": "#707070", "solid_capstyle": "butt"},
            whiskerprops={"color": "#707070", "solid_capstyle": "butt"},
            capprops={"color": "#707070"},
            flierprops={
                "marker": "o",
                "markersize": 2,
                "markeredgecolor": "#707070",
                "markerfacecolor": "none",
            },
        )
        for patch, label in zip(box["boxes"], classes):
            patch.set(
                facecolor=class_style(label)["fill"],
                edgecolor=class_style(label)["ink"],
            )
        ax.set(
            title=f"{feature.replace('_', ' ')} · distribution across classes",
            xlabel="Defect category",
            ylabel=feature.replace("_", " "),
        )
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        show_chart(fig)
    finally:
        plt.close(fig)
    st.caption(
        "Box: middle 50%; line: median; whiskers: 1.5 × interquartile range; points: outliers."
    )
    st.markdown("#### Two-feature inspection")
    x_column, y_column = st.columns(2)
    x_feature = x_column.selectbox("X feature", schema["features"], key="scatter_x")
    y_feature = y_column.selectbox(
        "Y feature", schema["features"], index=1, key="scatter_y"
    )
    fig, ax = new_chart((11, 5.2))
    try:
        sns.scatterplot(
            data=plot_data,
            x=x_feature,
            y=y_feature,
            hue="Category",
            hue_order=labels,
            palette=palette,
            alpha=0.65,
            s=24,
            linewidth=0,
            ax=ax,
        )
        ax.set(
            title="Feature relationship · colored by true category",
            xlabel=x_feature.replace("_", " "),
            ylabel=y_feature.replace("_", " "),
        )
        ax.legend(
            title="True category",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )
        show_chart(fig)
    finally:
        plt.close(fig)

    render_training_correlation(schema)


# Single-record inference never depends on upload state or evaluation metrics.
def reset_single_inputs(example: dict | None) -> None:
    """Set real example measurements or blank inputs before widgets are rendered."""
    for features in FEATURE_GROUPS.values():
        for feature in features:
            value = None if example is None else example[feature]
            if value is not None:
                value = int(value) if feature in INTEGER_FEATURES else float(value)
            st.session_state[f"single_{feature}"] = value
    steel = None
    if example is not None:
        steel = "A300" if example["TypeOfSteel_A300"] == 1 else "A400"
    st.session_state["single_steel"] = steel
    st.session_state["single_initialized"] = True
    st.session_state.pop("single_result", None)


def render_single_result(snapshot: dict) -> None:
    """Present only the last submitted model, hard prediction, and probabilities."""
    result = snapshot["result"]
    with st.container(border=True, key="single_result"):
        st.subheader(f"Predicted category: {display_label(result['prediction'])}")
        st.markdown(
            result_card_html(result, MODEL_NAMES[result["model_id"]]),
            unsafe_allow_html=True,
        )
        st.caption(
            f"LAST SUBMITTED / {MODEL_NAMES[result['model_id']]} · "
            f"Experiment {snapshot['run_id']} · SHA-256 {result['feature_fingerprint'][:12]}"
        )
        st.caption(
            "This result belongs to the submitted model and measurements below. "
            "Input or model edits require resubmission; they do not update this result."
        )
        if result["prediction"] != result["top_probability_class"]:
            st.info(
                "The hard prediction is retained from the model's predict decision. "
                "SVC probability calibration can rank another category highest: "
                f"{display_label(result['top_probability_class'])}. "
                "The probability above is for the predicted category, not the maximum probability."
            )
        fig, ax = new_chart((9, 4))
        try:
            bars = ax.barh(
                [display_label(label) for label in result["classes"]],
                result["probabilities"],
                color=[class_style(label)["fill"] for label in result["classes"]],
                edgecolor=[
                    class_style(label)["ink"]
                    if label == result["prediction"]
                    else "#9aa7b3"
                    for label in result["classes"]
                ],
                linewidth=[
                    2.5 if label == result["prediction"] else 0.8
                    for label in result["classes"]
                ],
            )
            ax.invert_yaxis()
            ax.set(
                xlim=(0, 1.15),
                xlabel="Model probability",
                title="All seven defect categories",
            )
            ax.bar_label(
                bars,
                labels=[f"{value:.2%}" for value in result["probabilities"]],
                padding=4,
            )
            show_chart(fig)
        finally:
            plt.close(fig)
        with st.expander("Submitted model & measurement snapshot"):
            st.write(f"Model: {MODEL_NAMES[result['model_id']]}")
            st.dataframe(
                pd.DataFrame([snapshot["inputs"]]), hide_index=True, width="stretch"
            )


def render_predict_one(resources: dict) -> None:
    """Collect 27 feature-only inputs and explicitly submit independent inference."""
    schema = resources["schema"]
    st.subheader("One defect. One saved classifier.")
    st.caption(
        "Enter measurements for an already-detected defect. No CSV or true category is required."
    )
    training = None
    example = None
    try:
        training = verified_training(schema).features
        example = training.iloc[0].to_dict()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.warning(
            f"Training example unavailable: {exc}. Enter measurements manually; prediction remains available."
        )
    load_column, clear_column = st.columns(2)
    load_example = load_column.button(
        "Load example / reset", key="single_load_example", disabled=example is None
    )
    clear_inputs = clear_column.button("Clear measurements", key="single_clear")
    if clear_inputs:
        reset_single_inputs(None)
    elif load_example or not st.session_state.get("single_initialized"):
        reset_single_inputs(example)
    if example is not None:
        st.caption(
            "Prefilled demonstration measurements from training data; not an independent test"
        )
        st.caption(
            "Load example restores the actual first training row, SHA-256 verified against the split manifest."
        )
    with st.form("single_form"):
        model_id = st.selectbox(
            "Prediction model",
            list(MODEL_NAMES),
            index=list(MODEL_NAMES).index("svm"),
            format_func=MODEL_NAMES.get,
            key="single_model_select",
        )
        record = {}
        for tint, (section, features) in zip(SECTION_TINTS, FEATURE_GROUPS.items()):
            st.markdown(
                f'<div class="section-heading" style="background:{tint}">{section}</div>',
                unsafe_allow_html=True,
            )
            if section == "04 / Steel & conveyer":
                steel = st.selectbox(
                    "Steel grade",
                    ["A300", "A400"],
                    index=None,
                    key="single_steel",
                    help="Source: TypeOfSteel_A300 and TypeOfSteel_A400. A300 → (1, 0); A400 → (0, 1).",
                )
                record["TypeOfSteel_A300"] = (
                    None if steel is None else int(steel == "A300")
                )
                record["TypeOfSteel_A400"] = (
                    None if steel is None else int(steel == "A400")
                )
            columns = st.columns(3)
            for index, feature in enumerate(features):
                observed = "Observed training range unavailable."
                if training is not None:
                    observed = f"Observed training range: {training[feature].min():g} to {training[feature].max():g} (advisory only)."
                integer = feature in INTEGER_FEATURES
                record[feature] = columns[index % 3].number_input(
                    feature.replace("_", " "),
                    value=None,
                    step=1 if integer else 0.0001,
                    format="%d" if integer else "%.5f",
                    key=f"single_{feature}",
                    help=f"Source: {feature}. Units: not documented in the source. {observed}",
                )
        submitted = st.form_submit_button(
            "Predict defect category", key="single_predict", type="primary"
        )
    if submitted:
        st.session_state.pop("single_result", None)
        try:
            features = validate_record(record, schema)
            result = predict_record(model_id, features, resources["registry"])
        except ValidationError as exc:
            st.error(f"Measurements rejected: {exc}")
        except ValueError:
            st.error(
                "Prediction failed. Verify the saved pipelines and experiment metadata, then resubmit."
            )
        else:
            st.session_state["single_result"] = {
                "result": result,
                "inputs": features.iloc[0].to_dict(),
                "run_id": resources["run_id"],
            }
    snapshot = st.session_state.get("single_result")
    if snapshot is not None:
        if training is not None:
            outside = [
                feature
                for feature, value in snapshot["inputs"].items()
                if value < training[feature].min() or value > training[feature].max()
            ]
            if outside:
                st.warning(
                    "Outside observed training ranges (advisory; values were not clamped): "
                    + ", ".join(outside)
                )
        render_single_result(snapshot)


# Application entry point: configure the page before creating any UI elements.
def main() -> None:
    """Coordinate verified session resources, strict input validation, and UI tabs."""
    st.set_page_config(page_title="Steel Defect Explorer", page_icon="⚙", layout="wide")
    style_page()
    st.markdown(
        '<div class="control-header"><div class="eyebrow">Quality engineering / Inspection analytics</div>'
        "<h1>Steel Defect Explorer</h1><p>Classify an already-detected steel defect into one of seven categories "
        "using 27 tabular measurements. Compare five saved classifiers and inspect your labeled batch. "
        "This is tabular defect classification, not image analysis.</p></div>"
        '<div class="category-strip">'
        + "".join(class_chip(label) for label in PALETTE)
        + "</div>",
        unsafe_allow_html=True,
    )
    try:
        resources = session_resources()
    except Exception as exc:
        reset_upload_state()
        st.session_state.pop("single_result", None)
        st.session_state.pop("resources", None)
        render_sidebar(None)
        detail = f" Contract check: {exc}" if isinstance(exc, ValueError) else ""
        st.error("Saved experiment unavailable." + detail)
        st.info(
            "Restore the matching metadata/schema.json, metadata/experiment.json and all five model/*.joblib files. Use the experiment's recorded scikit-learn version, then reload the app."
        )
        return
    schema = resources["schema"]
    uploaded, model_id = render_sidebar(schema)
    # The first tab opens by default: the labeled batch evaluation is the primary workflow.
    batch_tab, single_tab = st.tabs(["Evaluate Batch", "Predict One"])
    with single_tab:
        render_predict_one(resources)
    with batch_tab:
        batch = current_batch(uploaded, schema)
        evaluate_tab, compare_tab, explore_tab = st.tabs(
            ["Evaluate", "Compare", "Explore"]
        )
        if batch is None:
            for tab, purpose in (
                (evaluate_tab, "evaluate a saved classifier"),
                (compare_tab, "compare all five models"),
                (explore_tab, "explore class and feature distributions"),
            ):
                with tab:
                    st.info(f"Upload a valid labeled CSV in the sidebar to {purpose}.")
        else:
            st.caption(
                f"VALIDATED UPLOAD · {batch.row_count:,} records · {len(schema['features'])} features · "
                f"{batch.target.nunique()}/7 categories · SHA-256 {batch.fingerprint[:12]} · "
                f"Experiment {resources['run_id']}"
            )
            with evaluate_tab:
                render_evaluate(batch, model_id, resources)
            with compare_tab:
                render_compare(batch, resources)
            with explore_tab:
                render_explore(batch, schema)


if __name__ == "__main__":
    main()
