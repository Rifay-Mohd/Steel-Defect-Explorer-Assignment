"""Session-local, inference-only manufacturing dashboard for steel fault labels."""

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit.typing import UploadedFile

from src.config import MODEL_NAMES, ROOT, SCHEMA_PATH, SPLIT_PATH, TEST_PATH, TRAIN_PATH
from src.inference import evaluate_model, get_experiment, load_artifacts
from src.validation import (
    MAX_ROWS,
    MAX_UPLOAD_BYTES,
    ValidatedBatch,
    ValidationError,
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
COLORS = ("#147d80", "#c08043", "#334d70", "#9c6574", "#71845a", "#5c96ad", "#8273a1")
NAVY = "#172b42"
TEAL = "#147d80"
PAPER = "#faf8f2"


def display_label(label: str) -> str:
    """Return a readable category name without changing its source identifier."""
    return DISPLAY_LABELS.get(label, label.replace("_", " "))


def style_page() -> None:
    """Apply local CSS for an accessible, warm manufacturing control panel."""
    st.markdown(
        """
        <style>
        .stApp { background: #f5f2e9; color: #172b42; }
        [data-testid="stHeader"] { background: #f5f2e9; }
        [data-testid="stSidebar"] { background: #e9eee9; border-right: 1px solid #cbd6d1; }
        .block-container { max-width: 1500px; padding-top: 4.5rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: #172b42; font-family: ui-sans-serif, system-ui, sans-serif; }
        h1 { font-weight: 800 !important; letter-spacing: -.045em; }
        h2, h3 { letter-spacing: -.025em; }
        p, label, [data-testid="stCaptionContainer"] { color: #33495b; }
        [data-testid="stMetric"] { background: #faf8f2; border: 1px solid #d5ddd5;
          border-top: 3px solid #147d80; border-radius: 8px; padding: 15px 12px; }
        [data-testid="stMetricValue"] { color: #172b42; font-variant-numeric: tabular-nums;
          font-size: clamp(1.3rem, 2.2vw, 2rem); }
        [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid #cbd6d1; }
        [data-baseweb="tab"] { color: #172b42; font-weight: 650; }
        [data-baseweb="tab-highlight"] { background-color: #147d80; }
        .stButton > button[kind="primary"] { background: #147d80; color: white; border: 0; }
        .stButton > button, .stDownloadButton > button { border-radius: 6px; }
        .control-header { border-left: 5px solid #147d80; padding: .3rem 0 .4rem 1.2rem;
          margin-bottom: 1.1rem; }
        .eyebrow { font: 700 .75rem ui-monospace, monospace; letter-spacing: .16em;
          color: #147d80; text-transform: uppercase; }
        .control-header h1 { margin: .25rem 0; font-size: clamp(2rem, 4vw, 3.4rem); }
        .control-header p { max-width: 850px; margin: .3rem 0; line-height: 1.6; }
        .category-strip { display: flex; flex-wrap: wrap; gap: .45rem; margin: 1rem 0 1.5rem; }
        .category-strip span { background: #e5ece6; color: #244c50; padding: .3rem .65rem;
          border: 1px solid #cbd8cd; border-radius: 4px; font-size: .8rem; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Session lifecycle: invalidate derived data before validating replacement bytes.
def reset_upload_state() -> None:
    """Discard upload-derived state before accepting a new or removed file."""
    for key in (
        "upload_identity",
        "batch",
        "results",
        "comparison_identity",
        "confusion_mode",
        "comparison_metric",
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
        st.caption("01 / INPUT BATCH")
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
            "Correct the CSV using the sidebar schema, then upload it again. No results are retained."
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
        column.metric(metric, "N/A" if value is None else f"{value:.4f}")
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
            report_frame(result).style.format(
                {
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}",
                    "Support": "{:d}",
                }
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
    metric = st.selectbox("Chart metric", METRICS, index=4, key="comparison_metric")
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
    st.dataframe(frame.style.format("{:.4f}", na_rep="N/A"), width="stretch")
    if results[0]["auc_message"]:
        st.warning(results[0]["auc_message"])
    if frame[metric].isna().all():
        st.info("AUC chart is unavailable for this upload. Select another metric.")
        return
    fig, ax = new_chart((10, 4.8))
    try:
        bars = ax.bar(frame.index, frame[metric], color=TEAL, width=0.6)
        ax.set_ylim((-1.15, 1.15) if metric == "MCC" else (0, 1.15))
        ax.axhline(0, color=NAVY, linewidth=0.8)
        ax.set(
            title=f"{metric} · five saved classifiers", ylabel=metric, xlabel="Model"
        )
        ax.bar_label(bars, fmt="%.4f", padding=4, color=NAVY)
        ax.set_xticks(
            range(len(frame)),
            [
                "Logistic\nRegression",
                "Decision\nTree",
                "KNN",
                "Gaussian\nNaive Bayes",
                "SVM",
            ],
        )
        show_chart(fig)
    finally:
        plt.close(fig)
    st.caption(
        "Source: current upload. This comparison does not select or retrain models."
    )


# Frozen-training reference: verify source bytes before computing feature correlations.
def render_training_correlation(schema: dict) -> None:
    """Plot signed Pearson correlations of verified training features, excluding target.

    Read project-relative configured paths on each rerun, check the frozen split
    fingerprint, and reuse CSV validation. Never use uploaded rows or fit a model.
    """
    with st.expander("Feature correlation · frozen training data", expanded=False):
        try:
            raw = TRAIN_PATH.read_bytes()
            training = json.loads(SPLIT_PATH.read_bytes())["partitions"]["training"]
            if hashlib.sha256(raw).hexdigest() != training["sha256"]:
                raise ValueError(
                    "Training CSV SHA-256 does not match the split manifest."
                )
            batch = validate_upload(raw, schema)
            if batch.row_count != training["rows"]:
                raise ValueError(
                    "Training row count does not match the split manifest."
                )
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
    palette = dict(zip(labels, COLORS))
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
            summary.style.format({"Percent": "{:.2f}%"}),
            hide_index=True,
            width="stretch",
        )
    with right:
        fig, ax = new_chart((8, 4))
        try:
            ax.barh(labels, counts.to_numpy(), color=COLORS)
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
        ax.boxplot(
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
                "facecolor": sns.desaturate("#a4cebf", 0.75),
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
        + "".join(f"<span>{label}</span>" for label in DISPLAY_LABELS.values())
        + "</div>",
        unsafe_allow_html=True,
    )
    try:
        resources = session_resources()
    except Exception as exc:
        reset_upload_state()
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
    batch = current_batch(uploaded, schema)
    evaluate_tab, compare_tab, explore_tab = st.tabs(["Evaluate", "Compare", "Explore"])
    if batch is None:
        for tab, purpose in (
            (evaluate_tab, "evaluate a saved classifier"),
            (compare_tab, "compare all five models"),
            (explore_tab, "explore class and feature distributions"),
        ):
            with tab:
                st.info(f"Upload a valid labeled CSV in the sidebar to {purpose}.")
        return
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
