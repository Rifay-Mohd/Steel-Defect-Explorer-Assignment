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

A single stratified 80/20 split uses `RANDOM_SEED = 42` (1,552 training / 389 held-out rows); shuffled five-fold training cross-validation and seeded classifiers use the same seed.

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
| Naive Bayes | `var_smoothing=[1e-11, 1e-9, 1e-7]` | `var_smoothing=1e-7` | 0.617489 |
| Support Vector Machine | `C=[0.1, 1, 10]`, `gamma=[scale, 0.01]` | `C=10`, `gamma=0.01` | **0.783722** |

Equal candidate scores use the first candidate in scikit-learn `ParameterGrid` order. Equal overall model scores use `MODEL_NAMES` insertion order, with all exact ties retained in experiment metadata. The recorded overall winner is SVM without a tie.

### Six-metric held-out comparison

These are actual results on the frozen 389-row `test_data.csv`, copied from `results/test_metrics.csv` and rounded to six decimals. They are held-out scores, not CV estimates or scores from an arbitrary app upload.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.727506 | 0.937941 | 0.761262 | 0.730528 | 0.741789 | 0.648680 |
| Decision Tree | 0.748072 | 0.905572 | 0.778415 | 0.736415 | 0.755113 | 0.672818 |
| KNN | **0.768638** | 0.921967 | 0.765293 | **0.786583** | **0.774095** | **0.704385** |
| Naive Bayes | 0.614396 | 0.924869 | 0.640667 | 0.719043 | 0.638040 | 0.554840 |
| Support Vector Machine | 0.753213 | **0.949834** | **0.797132** | 0.750933 | 0.767909 | 0.681378 |

### Shared metric policy

`src/evaluation.py` is shared by offline training and labeled-batch app evaluation. Feature-only **Predict One** inference does not call the evaluator or compute evaluation metrics. Accuracy and multiclass MCC use hard predictions. Precision, recall, and F1 are unweighted macro averages over the fixed seven canonical classes with `zero_division=0`, including absent classes as zero contributions. AUC is seven-class one-vs-rest macro ROC AUC, with probability columns aligned to lexicographically sorted labels. Confusion matrices and class reports retain all seven classes.

If an uploaded batch lacks any expected true class, AUC is unavailable: the backend returns `None`, and the UI displays **N/A** with an explanation. The other five metrics remain available under the same fixed-label policy; a perfectly predicted batch containing only three classes therefore has macro precision, recall, and F1 of `3/7`, not 1.

Hard labels always come from `pipeline.predict`; probabilities come separately from `pipeline.predict_proba`. In particular, calibrated SVC probabilities can have an argmax that differs from `SVC.predict`. The evaluator preserves the original hard predictions rather than replacing them with probability argmax.

### Observations

| Model | Observation from this experiment |
| --- | --- |
| Logistic Regression | CV macro F1 was 0.722047 and held-out F1 was 0.741789. Its held-out AUC of 0.937941 exceeded Decision Tree and KNN, while its hard-label F1 was lower than both. |
| Decision Tree | The depth-10, minimum-leaf-3 model reached held-out F1 of 0.755113, above Logistic Regression. Its AUC of 0.905572 was the lowest of the five. |
| KNN | Distance-weighted five-neighbor KNN had the best held-out F1 (0.774095), accuracy (0.768638), recall (0.786583), and MCC (0.704385), despite ranking below SVM in training-only CV. |
| Naive Bayes | The selected smoothing value was 1e-7. It had the lowest CV F1 (0.617489) and held-out F1 (0.638040); its AUC of 0.924869 shows that probability ranking and hard-label quality need not rank models identically. |
| Support Vector Machine | SVM was the overall CV winner at 0.783722. It had the highest held-out AUC (0.949834) and precision (0.797132), with held-out F1 of 0.767909, slightly below KNN. |
| **Overall Winner** | **SVM** was selected by training-only CV macro F1 (0.783722). **KNN** had the best held-out macro F1 (0.774095); this held-out result does not change the frozen CV selection. |

> **Overall CV-selected model: SVM**
>
> **Best held-out macro F1: KNN**

These are distinct findings. The held-out ranking does not retroactively change the CV selection. Class imbalance and small minority-class test supports, including only 11 Dirtiness records, limit how broadly this single split should be interpreted.

## Reproduce locally

Run from the project root using **Python 3.12**. If `.venv` already exists, skip its creation and activate it.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-lock.txt -r requirements-dev.txt
python -m streamlit run app.py
```

Open [localhost:8501](http://localhost:8501), upload `test_data.csv` in the default **Evaluate Batch** tab, or use **Predict One** for a single record.

### Optional checks

Tests require the external `jq` CLI on `PATH` (installed separately from pip).

```bash
jq --version
python -m scripts.prepare_data
python -m pytest tests -q
```

### Optional fresh training

Requires a clean experiment copy with source and prepared data/metadata but no existing `model/`, `results/`, or `metadata/experiment.json` outputs; training does not overwrite existing outputs.

```bash
python -m scripts.train_models
```
