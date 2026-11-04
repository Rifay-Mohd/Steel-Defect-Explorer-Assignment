# Steel Defect Explorer

## a. Problem statement

Classify an already-detected steel plate defect into one of seven fault categories using 27 numeric tabular measurements. This assignment compares five saved classification pipelines and provides an inference-only Streamlit application. The mandatory labeled-CSV evaluation workflow is available under **Evaluate Batch**; **Predict One** is an optional, feature-only single-record extension in the current local build. The dataset contains defect records, not healthy/non-defective examples; these results do not establish healthy-versus-defective detection performance. The inputs are measurements, not images, and the application does not perform image analysis.

## b. Dataset description

The source is the [UCI Steel Plates Faults dataset](https://archive.ics.uci.edu/dataset/198/steel+plates+faults), obtained from the [official ZIP archive](https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip). The retained `data/raw/Faults.NNA` contains 1,941 rows and 34 numeric columns: 27 features followed by seven one-hot target indicators. `data/raw/Faults27x7_var` determines the exact source column order. Preparation verifies binary targets with exactly one active indicator per row, derives the single `fault_type` label, and excludes all target indicators from the features.

The local audit in `metadata/data_audit.json` reports zero missing values, all values finite, zero exact duplicate rows, zero identical-feature rows, and zero conflicting-label rows. These findings permit the recorded row-level split; they do not establish independence across production batches or generalization to other plants.

| Canonical source label | All rows | Training rows | Held-out rows |
| --- | ---: | ---: | ---: |
| Bumps | 402 | 321 | 81 |
| Dirtiness | 55 | 44 | 11 |
| K_Scatch | 391 | 313 | 78 |
| Other_Faults | 673 | 538 | 135 |
| Pastry | 158 | 126 | 32 |
| Stains | 72 | 58 | 14 |
| Z_Scratch | 190 | 152 | 38 |
| **Total** | **1,941** | **1,552** | **389** |

The exact source spelling is **`K_Scatch`**. The UI displays it as **K Scratch**, but CSV labels, metadata, saved predictions, and model classes retain `K_Scatch`; `K_Scratch` is rejected as an unknown upload label.

The ordered features are recorded in `metadata/schema.json`:

```text
X_Minimum, X_Maximum, Y_Minimum, Y_Maximum, Pixels_Areas, X_Perimeter, Y_Perimeter, Sum_of_Luminosity, Minimum_of_Luminosity, Maximum_of_Luminosity, Length_of_Conveyer, TypeOfSteel_A300, TypeOfSteel_A400, Steel_Plate_Thickness, Edges_Index, Empty_Index, Square_Index, Outside_X_Index, Edges_X_Index, Edges_Y_Index, Outside_Global_Index, LogOfAreas, Log_X_Index, Log_Y_Index, Orientation_Index, Luminosity_Index, SigmoidOfAreas
```

**Citation:** Buscema, M., Terzi, S., & Tastle, W. (2010). *Steel Plates Faults* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5J88N](https://doi.org/10.24432/C5J88N). The source dataset is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Local transformations convert the one-hot targets into `fault_type` and create the documented stratified partitions.

## c. GitHub repository link

**Public repository:** [Rifay-Mohd/Steel-Defect-Explorer-Assignment](https://github.com/Rifay-Mohd/Steel-Defect-Explorer-Assignment). The published source is on the default branch, [`develop`](https://github.com/Rifay-Mohd/Steel-Defect-Explorer-Assignment/tree/develop). Repository visibility, branch availability, and required published files were verified previously.

**Live app:** [https://steel-defect-explorer-rifay.streamlit.app/](https://steel-defect-explorer-rifay.streamlit.app/). The user confirmed that this existing Streamlit Community Cloud deployment works and serves the previous build. The local pastel redesign and **Predict One** extension are pending push/publication and subsequent cloud verification; their availability at the live URL has not been verified.

## d. Models used

The completed experiment is `20261102T160049Z-21d3849d1f0b`, recorded in `metadata/experiment.json`. All searches use the same five shuffled, stratified training-only folds, macro F1 scoring over the fixed seven classes, `zero_division=0`, and `n_jobs=1`. Each winning candidate is refitted on all 1,552 training rows. Overall model selection is frozen before held-out evaluation.

`StandardScaler` is inside the Logistic Regression, KNN, Gaussian Naive Bayes, and SVM pipelines, so scaling is fitted within each training fold during CV. Decision Tree uses the original feature scales without a scaler. Logistic Regression uses `max_iter=5000`; SVM uses an RBF kernel and `probability=True`.

### Search configuration and actual training results

Parameter names below abbreviate the actual `classifier__` pipeline prefix. CV scores are the recorded best mean validation macro F1, rounded to six decimals; full precision and complete parameters are in `metadata/experiment.json`, and candidate/fold scores are in `results/cv_results.csv`.

| Model | Searched parameters | Actual best parameters | CV macro F1 |
| --- | --- | --- | ---: |
| Logistic Regression | `C=[0.1, 1, 10]` | `C=1` | 0.722047 |
| Decision Tree | `max_depth=[None, 5, 10]`, `min_samples_leaf=[1, 3]` | `max_depth=10`, `min_samples_leaf=3` | 0.737935 |
| KNN | `n_neighbors=[3, 5, 9]`, `weights=[uniform, distance]` | `n_neighbors=5`, `weights=distance` | 0.764824 |
| Gaussian Naive Bayes | `var_smoothing=[1e-11, 1e-9, 1e-7]` | `var_smoothing=1e-7` | 0.617489 |
| SVM | `C=[0.1, 1, 10]`, `gamma=[scale, 0.01]` | `C=10`, `gamma=0.01` | **0.783722** |

Equal candidate scores use the first candidate in scikit-learn `ParameterGrid` order. Equal overall model scores use `MODEL_NAMES` insertion order, with all exact ties retained in experiment metadata. The recorded overall winner is SVM without a tie.

### Six-metric held-out comparison

These are actual results on the frozen 389-row `test_data.csv`, copied from `results/test_metrics.csv` and rounded to six decimals. They are held-out scores, not CV estimates or scores from an arbitrary app upload.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.727506 | 0.937941 | 0.761262 | 0.730528 | 0.741789 | 0.648680 |
| Decision Tree | 0.748072 | 0.905572 | 0.778415 | 0.736415 | 0.755113 | 0.672818 |
| KNN | **0.768638** | 0.921967 | 0.765293 | **0.786583** | **0.774095** | **0.704385** |
| Gaussian Naive Bayes | 0.614396 | 0.924869 | 0.640667 | 0.719043 | 0.638040 | 0.554840 |
| SVM | 0.753213 | **0.949834** | **0.797132** | 0.750933 | 0.767909 | 0.681378 |

### Shared metric policy

`src/evaluation.py` is shared by offline training and labeled-batch app evaluation. Feature-only **Predict One** inference does not call the evaluator or compute evaluation metrics. Accuracy and multiclass MCC use hard predictions. Precision, recall, and F1 are unweighted macro averages over the fixed seven canonical classes with `zero_division=0`, including absent classes as zero contributions. AUC is seven-class one-vs-rest macro ROC AUC, with probability columns aligned to lexicographically sorted labels. Confusion matrices and class reports retain all seven classes.

If an uploaded batch lacks any expected true class, AUC is unavailable: the backend returns `None`, and the UI displays **N/A** with an explanation. The other five metrics remain available under the same fixed-label policy; a perfectly predicted batch containing only three classes therefore has macro precision, recall, and F1 of `3/7`, not 1.

Hard labels always come from `pipeline.predict`; probabilities come separately from `pipeline.predict_proba`. In particular, calibrated SVC probabilities can have an argmax that differs from `SVC.predict`. The evaluator preserves the original hard predictions rather than replacing them with probability argmax.

## e. Observations

| Model | Observation from this experiment |
| --- | --- |
| Logistic Regression | CV macro F1 was 0.722047 and held-out F1 was 0.741789. Its held-out AUC of 0.937941 exceeded Decision Tree and KNN, while its hard-label F1 was lower than both. |
| Decision Tree | The depth-10, minimum-leaf-3 model reached held-out F1 of 0.755113, above Logistic Regression. Its AUC of 0.905572 was the lowest of the five. |
| KNN | Distance-weighted five-neighbor KNN had the best held-out F1 (0.774095), accuracy (0.768638), recall (0.786583), and MCC (0.704385), despite ranking below SVM in training-only CV. |
| Gaussian Naive Bayes | The selected smoothing value was 1e-7. It had the lowest CV F1 (0.617489) and held-out F1 (0.638040); its AUC of 0.924869 shows that probability ranking and hard-label quality need not rank models identically. |
| SVM | SVM was the overall CV winner at 0.783722. It had the highest held-out AUC (0.949834) and precision (0.797132), with held-out F1 of 0.767909, slightly below KNN. |
| **Overall Winner** | **SVM** was selected by training-only CV macro F1 (0.783722). **KNN** had the best held-out macro F1 (0.774095); this held-out result does not change the frozen CV selection. |

**Overall CV-selected model: SVM. Best held-out macro F1: KNN.** These are distinct findings. The held-out ranking does not retroactively change the CV selection. Class imbalance and small minority-class test supports, including only 11 Dirtiness records, limit how broadly this single split should be interpreted.

## Reproduce locally

### 1. Python environment and dependencies

Run all commands from this project's root directory. Use **Python 3.12**; the completed experiment records **Python 3.12.12**. The tests also require the external **`jq` CLI** on `PATH`; it is not installed by pip. Check that Python and jq are available before creating the environment. If `.venv` already exists, activate it rather than recreating it.

```bash
python3.12 --version
jq --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-lock.txt -r requirements-dev.txt
python -m pip check
```

`requirements.txt` pins the used runtime packages: Streamlit 1.66.1, scikit-learn 1.9.1, pandas 3.1.0, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2, joblib 1.6.0, and seaborn 0.13.2. Seaborn is used by `app.py` for heatmaps, scatterplots, and color adjustment. `requirements-dev.txt` includes the runtime requirements and pytest 9.2.0. `requirements-lock.txt` is the complete observed `python -m pip freeze` snapshot of the verified local environment, including transitive packages, and constrains their versions in the command above. It is a version snapshot, not a hash-verified or cross-platform installation guarantee.

### 2. Verify or prepare the frozen dataset

```bash
python -m scripts.prepare_data
```

The preparer reuses the retained raw ZIP when present and downloads from UCI when absent. It audits the original representation and produces `data/processed/training_data.csv`, `test_data.csv`, `metadata/data_audit.json`, `metadata/schema.json`, and `metadata/split_manifest.json`. The split is stratified 80/20 with 1,552 training rows and 389 held-out rows. The manifest records ordered zero-based source-row identities, class counts, and SHA-256 fingerprints.

**Frozen-artifact warning:** preparation accepts byte-identical existing artifacts but refuses to overwrite any differing artifact. Training also verifies frozen partition hashes, schema, membership, and class counts. Do not edit the CSVs or manifests to bypass a refusal. Restore the matching recorded files for this experiment, or create a separate clean experiment directory and regenerate the preparation manifests and all downstream outputs together.

`RANDOM_SEED = 42` in `src/config.py` controls the split, shuffled CV folds, and seeded classifiers. The value 42 is a reproducibility choice, not a tuned hyperparameter. Changing it defines a new experiment: existing source-row memberships, hashes, folds, and saved results will no longer describe that configuration. Current scripts also enforce the approved row counts and split policy, so a changed dataset or experiment policy requires corresponding code review and regeneration in a clean experiment directory.

### 3. Train in a clean experiment directory

```bash
python -m scripts.train_models
```

**The supplied project already contains a complete experiment.** This command intentionally raises `FileExistsError` when `metadata/experiment.json`, any `model/*.joblib`, or any top-level `results/*.csv` exists, including partial runs. Use the existing artifacts for tests and the app. For a fresh training run, use a separate clean copy containing the source and prepared data/metadata but no previous `model/`, `results/`, or `metadata/experiment.json` outputs; keep the original experiment intact.

Training prints each model's actual best parameters, CV macro F1, elapsed seconds, captured warnings, the CV-selected model, and the six-metric held-out table. It persists the outputs below, reloads predictions and probabilities in a fresh Python process, and writes `metadata/experiment.json` last as the completion marker. New runs have new run IDs, timings, and metadata bytes even with the same seed.

| Artifact path | Contents |
| --- | --- |
| `data/raw/steel+plates+faults.zip`, `data/raw/Faults.NNA`, `data/raw/Faults27x7_var` | Retained official source archive, measurements, and column names |
| `data/processed/training_data.csv` | Frozen 1,552-row labeled training partition |
| `test_data.csv` | Frozen 389-row labeled held-out partition and downloadable app example |
| `metadata/data_audit.json`, `metadata/schema.json`, `metadata/split_manifest.json` | Audit findings, feature/label contract, split identities and hashes |
| `metadata/experiment.json` | Completion status, run ID, versions, CV folds, best parameters, warnings, model/output fingerprints, and reload-verification record |
| `model/{slug}.joblib` | Five fitted preprocessing/classifier pipelines |
| `results/cv_results.csv` | All 24 candidate searches, fold scores, ranks, and timing data |
| `results/test_metrics.csv` | Five models by six held-out metrics |
| `results/test_predictions.csv` | Run/model IDs, held-out source-row IDs, true labels, hard predictions, and seven probability columns |
| `results/classification_reports/{slug}.json` | Per-class and aggregate held-out reports |
| `results/confusion_matrices/{slug}.csv` | Labeled seven-class count matrices |
| `results/figures/{slug}_confusion_matrix.png` | Offline confusion-matrix figures |

The five slugs are `logistic_regression`, `decision_tree`, `knn`, `gaussian_nb`, and `svm`. These are actual local artifact paths; model, data, and result files are retained rather than ignored.

### 4. Run the tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q -p no:cacheprovider
```

**Independent local verification from the current project folder: 138 tests passed, with 15 warnings.** The suite was run against the current app/backend using `.venv/bin/python` with the command above. The warnings were 13 seaborn heatmap `set_bad` pending deprecations, scikit-learn's single-label metric warning, and pandas' `float_precision` deprecation in the saved-prediction fixture. `python -m pip check` passed in a previous session. Tests read the existing experiment without retraining and cover raw-source integrity, frozen memberships and CV folds, saved inference parity, training-only scaler statistics, metric oracles, upload validation boundaries, app state transitions, and the training-only correlation reference. Single-record coverage includes exact feature validation, all five real pipelines, actual SVC prediction/probability disagreement, example/reset/clear controls, submitted snapshots, advisory range warnings, and failure-state cleanup.

The Streamlit tests execute the real `app.py` through `AppTest`, supplying uploads by mocking `st.file_uploader`'s returned object with a named `BytesIO`. Normal-path tests use real model loading, validation, inference, charts, controls, and session state; targeted tests also simulate source failures, metadata changes, and invalid probability output. This is local automated test evidence; it does not verify browser upload transport, the real uploader's `on_change` callback, or a remotely deployed application.

### 5. Launch Streamlit

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The current local UI uses a light pastel theme with mint input panels, lavender results, pale-blue headers, and teal accents. Its top-level tabs are **Evaluate Batch**, which opens by default, and **Predict One**. The app performs inference only and does not retrain models.

**Evaluate Batch — mandatory labeled-CSV workflow:** download the labeled test CSV from the sidebar and explicitly upload it; the application does not silently evaluate the example. Within **Evaluate Batch**, choose a saved model in **Evaluate**, click **Evaluate all five models** in **Compare** (one table plus a single grouped chart showing all six metrics for every model, with one color per metric), or inspect class and feature distributions in **Explore**. Comparisons use the current upload and do not select models. Prediction downloads preserve upload order, zero-based `row_index`, canonical true/predicted labels, and seven class probabilities.

Uploads must be UTF-8 CSVs, optionally with a leading BOM, containing exactly the 27 finite numeric feature columns plus `fault_type`. Column order may vary; target labels must match the seven source labels exactly. The limits are 5 MiB and 10,000 data rows. Missing/extra/duplicate headers, target-indicator columns, unknown labels, missing/nonfinite/nonnumeric features, blank records, and malformed records are rejected. No imputation or label repair occurs. Replacing, invalidating, or removing an upload clears its derived results and comparison state; resources and results are session-local.

The **Explore** tab also provides a training-only Pearson correlation heatmap across all 27 numeric features, excluding `fault_type`. Its source is the frozen `data/processed/training_data.csv` (1,552 rows), verified by SHA-256 against `metadata/split_manifest.json`: `258afa4b264d1c4dce75f71ed68d894398d305da3f36376a4a9a86b8bd47ad92`. This reference is independent of the current upload and excludes held-out rows.

**Predict One — optional feature-only extension:** enter one record without a CSV upload or true category. The form represents exactly the same 27 schema features, grouped into **Position & extent**, **Area & perimeter**, **Luminosity**, **Steel & conveyer**, and **Shape & derived indices**. There are 25 numeric inputs plus one steel-grade selector that supplies the two complementary source fields: A300 maps to `(TypeOfSteel_A300=1, TypeOfSteel_A400=0)`, and A400 to `(0, 1)`. No `fault_type`, one-hot target indicators, or fabricated targets are supplied to single-record inference.

The initial demonstration values are the actual first row's features from the frozen training CSV, after its SHA-256 and row count are checked against the split manifest. These are training measurements, explicitly labeled as a demonstration, not an independent test. **Load example / reset** and **Clear measurements** sit outside the submission form; loading the example restores those measurements, while clearing empties them, and both clear the previous single-record result. If the training reference is unavailable or fails verification, example loading is disabled and manual prediction remains available.

The form's **Prediction model** defaults to **SVM**, the training-only CV-selected model; any of the five saved pipelines can be chosen independently of the sidebar's batch model. Click **Predict defect category** to submit. Single-record validation requires exactly 27 finite numeric values, binary complementary steel flags, strictly increasing X/Y bounds, and minimum luminosity no greater than maximum luminosity. Observed training ranges appear in numeric-input help as advisory information; source units are not documented. A submitted value outside an observed range produces a warning but is not clamped or rejected solely for being outside that range.

The result shows the hard predicted category, its estimated model probability, all seven estimated class probabilities, and an expandable submitted-model/measurement snapshot. These probabilities are model estimates, not confidence guarantees or measured single-record accuracy. With no true label, **Predict One** computes no Accuracy, AUC, Precision, Recall, F1, MCC, confusion matrix, or classification report. The hard label comes from `pipeline.predict`, separately from `predict_proba`; if SVC calibration ranks another category highest, the UI explains the mismatch and retains the hard decision. The displayed predicted-category probability is not substituted with the maximum probability.

Results are marked **LAST SUBMITTED** with the submitted model, experiment, and feature fingerprint. Editing form inputs or the prediction model does not update that snapshot: resubmit to obtain a new result. An invalid or failed submission clears the previous single-record output. Batch upload replacement/removal does not clear an independent single-record result; a refreshed experiment/resource identity does.

## Model compatibility and recorded warnings

Load only trusted local joblib artifacts from the matching experiment. Joblib deserialization can execute code; uploaded model files are not accepted. Integrity checks compare artifact hashes and byte counts, schema and class metadata, completion status, fitted pipeline contracts, and the exact installed scikit-learn version before serving inference. Hash checks establish consistency with trusted metadata, not the safety of an untrusted replacement of both metadata and models. Saved models are version-coupled; preserve the recorded environment and use scikit-learn **1.9.1** for this experiment.

The experiment records **31 `FutureWarning` occurrences** from SVM fitting under scikit-learn 1.9.1: `probability` was deprecated in 1.9 and is scheduled for removal in 1.11, with `CalibratedClassifierCV(SVC(), ensemble=False)` named as the replacement. See the [SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html). The frozen recipe retains built-in probability calibration, and the warning category, message, location, and count are persisted in `metadata/experiment.json`. Adopting a different calibration recipe would require a new experiment and regenerated results.

## Local status

The local experiment is complete, all five pipelines and recorded outputs are present, and independent verification of the current local app/backend yielded **146 passed, 15 warnings**. The pastel redesign and optional **Predict One** extension are implemented locally and covered by automated tests; pushing/publishing the updated UI and verifying it on Streamlit Community Cloud remain pending.

The existing [live app](https://steel-defect-explorer-rifay.streamlit.app/) is deployed and user-confirmed working on the previous build. That confirmation does not establish deployment of the new local changes. Local browser upload, evaluation, and five-model comparison were tested in a previous session; retained screenshots are [evaluation](results/figures/local_app_evaluation.png) and [comparison](results/figures/local_app_comparison.png). Those screenshots predate the project folder move and the current redesign, so they are previous-version browser evidence, not verification of the new theme or **Predict One**. Browser verification of the updated UI from the current folder remains pending. The public GitHub repository and required files on `develop` were verified previously. BITS Virtual Lab execution evidence and the final submission PDF remain pending.
