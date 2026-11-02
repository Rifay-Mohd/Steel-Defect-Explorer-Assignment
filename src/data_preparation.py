"""Acquire, audit, and freeze the original UCI Steel Plates Faults representation."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    AUDIT_PATH,
    RANDOM_SEED,
    RAW_DIRECTORY,
    ROOT,
    SCHEMA_PATH,
    SOURCE_PAGE,
    SOURCE_URL,
    SPLIT_PATH,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)


@dataclass
class DataAudit:
    """Validated labeled source rows, ordered schema, and serializable audit findings."""

    rows: pd.DataFrame
    features: list[str]
    source_targets: list[str]
    classes: list[str]
    summary: dict


def sha256_bytes(content: bytes) -> str:
    """Return the standard SHA-256 hexadecimal content fingerprint."""
    return sha256(content).hexdigest()


def json_bytes(value: dict) -> bytes:
    """Serialize metadata deterministically with strict, finite JSON numbers."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def write_immutable_files(outputs: dict[Path, bytes]) -> None:
    """Preflight every output and create exclusively, accepting identical existing bytes.

    Existing differing files are never replaced. Exclusive creation also guards
    against another process creating a conflicting file after the initial check.
    """
    for path, content in outputs.items():
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite differing artifact: {path}")
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(content)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(
                    f"Artifact changed concurrently: {path}"
                ) from None


def acquire_source() -> dict[str, Path]:
    """Fetch the official HTTPS ZIP and safely preserve only its two known raw files.

    The original ZIP is retained too. No archive member name becomes an output
    path: only exact allowlisted entries are read, with bounded archive sizes.
    """
    archive_path = RAW_DIRECTORY / "steel+plates+faults.zip"
    if archive_path.exists():
        content = archive_path.read_bytes()
    else:
        request = Request(SOURCE_URL, headers={"User-Agent": "SteelDefectExplorer/1.0"})
        with urlopen(request, timeout=60) as response:
            if not response.url.startswith("https://archive.ics.uci.edu/"):
                raise ValueError(
                    "Dataset download redirected away from official UCI HTTPS."
                )
            content = response.read(5_000_001)
    if len(content) > 5_000_000:
        raise ValueError("Source archive exceeds the expected download size bound.")

    # Read individual members rather than using extract or extractall.
    names = ("Faults.NNA", "Faults27x7_var")
    outputs = {archive_path: content}
    with ZipFile(BytesIO(content)) as archive:
        for name in names:
            matches = [info for info in archive.infolist() if info.filename == name]
            if len(matches) != 1 or matches[0].file_size > 2_000_000:
                raise ValueError(
                    f"Expected exactly one bounded archive member named {name}."
                )
            outputs[RAW_DIRECTORY / name] = archive.read(matches[0])
    write_immutable_files(outputs)
    return {name: RAW_DIRECTORY / name for name in names}


def audit_source(source_files: dict[str, Path]) -> DataAudit:
    """Validate source column order, numeric/one-hot integrity, and duplicate identity.

    Duplicate and conflicting-label findings are returned for documentation; the
    split gate refuses either finding without an explicitly revised policy.
    """
    variable_bytes = source_files["Faults27x7_var"].read_bytes()
    numeric_bytes = source_files["Faults.NNA"].read_bytes()
    columns = variable_bytes.decode("utf-8-sig").split()
    if len(columns) != 34 or len(set(columns)) != 34:
        raise ValueError("Faults27x7_var must name exactly 34 unique columns.")
    features, targets = columns[:27], columns[27:]
    known_targets = {
        "Pastry",
        "Z_Scratch",
        "Stains",
        "Dirtiness",
        "Bumps",
        "Other_Faults",
    }
    if set(targets) not in (
        known_targets | {"K_Scatch"},
        known_targets | {"K_Scratch"},
    ):
        raise ValueError(
            "Source target names differ from the documented seven classes; investigate."
        )
    frame = pd.read_csv(BytesIO(numeric_bytes), sep=r"\s+", header=None)
    if frame.shape != (1941, 34):
        raise ValueError(f"Expected source shape (1941, 34), observed {frame.shape}.")
    frame.columns = columns
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError("Source includes nonnumeric values; no coercion is permitted.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Source includes missing or infinite values.")
    indicators = frame[targets].to_numpy()
    if not np.isin(indicators, [0, 1]).all() or not (indicators.sum(axis=1) == 1).all():
        raise ValueError(
            "Source targets must be binary with exactly one active indicator per row."
        )
    labels = np.asarray(targets)[indicators.astype(bool).nonzero()[1]]
    if set(labels) != set(targets):
        raise ValueError("Not all seven source classes are represented.")

    # Source row identities are zero-based, before any split or CSV serialization.
    rows = frame[features].copy()
    rows[TARGET] = labels
    exact_mask = frame.duplicated(keep=False)
    feature_mask = rows.duplicated(subset=features, keep=False)
    conflicting = (
        rows.groupby(features, sort=False, dropna=False)[TARGET].transform("nunique")
        > 1
    )
    summary = {
        "source_shape": list(frame.shape),
        "feature_count": len(features),
        "features": features,
        "source_target_order": targets,
        "classes": sorted(targets),
        "class_counts": rows[TARGET].value_counts().sort_index().to_dict(),
        "missing_values": int(frame.isna().sum().sum()),
        "all_values_finite": True,
        "binary_targets": True,
        "one_active_target_per_row": True,
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "exact_duplicate_source_row_ids": np.flatnonzero(exact_mask).tolist(),
        "identical_feature_rows": int(feature_mask.sum()),
        "conflicting_label_rows": int(conflicting.sum()),
        "conflicting_source_row_ids": np.flatnonzero(conflicting).tolist(),
        "source_row_id_convention": "zero-based row index in original Faults.NNA",
        "source_sha256": {
            "Faults.NNA": sha256_bytes(numeric_bytes),
            "Faults27x7_var": sha256_bytes(variable_bytes),
        },
        "disposition": "stop"
        if exact_mask.any() or conflicting.any()
        else "row-level split permitted",
    }
    return DataAudit(rows, features, targets, sorted(targets), summary)


def prepare_data() -> DataAudit:
    """Acquire and audit data, then freeze the seed-42 1,552/389 stratified split."""
    audit = audit_source(acquire_source())
    write_immutable_files({AUDIT_PATH: json_bytes(audit.summary)})
    if audit.summary["disposition"] == "stop":
        raise ValueError(
            "Duplicate/conflicting source rows found; see data_audit.json and request policy approval."
        )
    train_ids, test_ids = train_test_split(
        np.arange(len(audit.rows)),
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=audit.rows[TARGET],
    )
    if (len(train_ids), len(test_ids)) != (1552, 389) or set(train_ids) & set(test_ids):
        raise ValueError("Unexpected split size or overlapping source row identities.")
    train, test = audit.rows.iloc[train_ids], audit.rows.iloc[test_ids]
    for partition in (train, test):
        if set(partition[TARGET]) != set(audit.classes):
            raise ValueError("Both frozen partitions must contain all seven classes.")

    # Deterministic bytes make repeated preparation a harmless identity check.
    train_bytes = train.to_csv(index=False, lineterminator="\n").encode("utf-8")
    test_bytes = test.to_csv(index=False, lineterminator="\n").encode("utf-8")
    schema = {
        "schema_version": 1,
        "features": audit.features,
        "classes": audit.classes,
        "target": TARGET,
        "source_target_order": audit.source_targets,
        "source_to_display": {
            label: label.replace("_", " ") for label in audit.classes
        },
        "input_policy": {
            "numeric": True,
            "finite": True,
            "missing_values_allowed": False,
        },
        "source": {
            "page": SOURCE_PAGE,
            "download_url": SOURCE_URL,
            "representation": "Faults.NNA with Faults27x7_var column order",
            "doi": "10.24432/C5J88N",
            "license": "CC BY 4.0",
            "sha256": audit.summary["source_sha256"],
        },
    }
    schema_bytes = json_bytes(schema)
    manifest = {
        "schema_version": 1,
        "random_seed": RANDOM_SEED,
        "test_size": 0.2,
        "split_method": "sklearn.model_selection.train_test_split, stratify=fault_type",
        "source_sha256": audit.summary["source_sha256"],
        "schema_sha256": sha256_bytes(schema_bytes),
        "audit_disposition": audit.summary["disposition"],
        "source_row_id_convention": audit.summary["source_row_id_convention"],
        "partitions": {
            "training": {
                "path": str(TRAIN_PATH.relative_to(ROOT)),
                "rows": len(train),
                "source_row_ids": train_ids.tolist(),
                "sha256": sha256_bytes(train_bytes),
                "class_counts": train[TARGET].value_counts().sort_index().to_dict(),
            },
            "test": {
                "path": str(TEST_PATH.relative_to(ROOT)),
                "rows": len(test),
                "source_row_ids": test_ids.tolist(),
                "sha256": sha256_bytes(test_bytes),
                "class_counts": test[TARGET].value_counts().sort_index().to_dict(),
            },
        },
    }
    write_immutable_files(
        {
            TRAIN_PATH: train_bytes,
            TEST_PATH: test_bytes,
            SCHEMA_PATH: schema_bytes,
            SPLIT_PATH: json_bytes(manifest),
        }
    )
    return audit
