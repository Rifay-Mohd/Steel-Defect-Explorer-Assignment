"""Strict, uncached validation of labeled CSV uploads."""

import csv
import hashlib
from dataclasses import dataclass
from io import StringIO
from numbers import Real
from typing import cast

import numpy as np
import pandas as pd

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ROWS = 10_000


class ValidationError(ValueError):
    """Indicate an upload that does not satisfy the input contract."""


@dataclass
class ValidatedBatch:
    """Hold schema-ordered features, unchanged labels, and the raw-byte SHA-256."""

    features: pd.DataFrame
    target: pd.Series
    fingerprint: str

    @property
    def row_count(self) -> int:
        """Return the number of uploaded data records, excluding the header."""
        return len(self.features)


def _schema_columns(schema: dict) -> tuple[list[str], str, list[str]]:
    """Check schema column/label definitions before using them for validation."""
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a dictionary.")
    features, target, classes = (
        schema.get("features"),
        schema.get("target"),
        schema.get("classes"),
    )
    for name, values in (("features", features), ("classes", classes)):
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Schema {name} must be unique nonempty strings.")
    features = cast(list[str], features)
    classes = cast(list[str], classes)
    if not isinstance(target, str) or not target or target in features:
        raise ValueError("Schema target must be a nonempty, separate column name.")
    return features, target, classes


def validate_record(record: dict, schema: dict) -> pd.DataFrame:
    """Return one finite float64 row in the schema's exact 27-feature order.

    Numeric strings are accepted; booleans, missing/extra fields, and nonnumeric
    or nonfinite values are rejected. Only source-compatible steel, coordinate,
    and luminosity relationships are enforced, never observed training ranges.
    """
    features, _, classes = _schema_columns(schema)
    if len(features) != 27 or len(classes) != 7:
        raise ValueError("Single-record schema requires 27 features and seven classes.")
    if not isinstance(record, dict):
        raise ValidationError("Record must be a feature dictionary.")
    missing, extra = set(features) - set(record), set(record) - set(features)
    if missing or extra:
        raise ValidationError(
            f"Record features do not match schema; missing={sorted(missing)}, "
            f"extra={sorted(map(str, extra))}."
        )

    # Parse scalar values explicitly so bools and container coercions cannot pass.
    values = []
    for feature in features:
        value = record[feature]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, str)):
            raise ValidationError(f"Feature {feature} must be a numeric scalar.")
        try:
            number = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValidationError(
                f"Feature {feature} must be finite numeric data."
            ) from exc
        if not np.isfinite(number):
            raise ValidationError(f"Feature {feature} must be finite numeric data.")
        values.append(number)
    frame = pd.DataFrame([values], columns=features, dtype=np.float64)

    # Enforce physical relationships without clipping or imposing training bounds.
    row = frame.iloc[0]
    steel = ["TypeOfSteel_A300", "TypeOfSteel_A400"]
    pairs = [
        ("X_Minimum", "X_Maximum"),
        ("Y_Minimum", "Y_Maximum"),
        ("Minimum_of_Luminosity", "Maximum_of_Luminosity"),
    ]
    if not set(steel + [name for pair in pairs for name in pair]).issubset(features):
        raise ValueError("Single-record schema lacks required source features.")
    if not row[steel].isin([0.0, 1.0]).all() or row[steel].sum() != 1.0:
        raise ValidationError("Steel flags must be binary and complementary.")
    for minimum, maximum in pairs[:2]:
        if row[minimum] >= row[maximum]:
            raise ValidationError(f"{minimum} must be less than {maximum}.")
    if row[pairs[2][0]] > row[pairs[2][1]]:
        raise ValidationError("Minimum luminosity must not exceed maximum luminosity.")
    return frame


def validate_upload(csv_bytes: bytes, schema: dict) -> ValidatedBatch:
    """Validate UTF-8 CSV bytes without repairing labels, rows, or column names.

    A leading UTF-8 BOM is accepted. Blank records, duplicate original headers,
    unequal record widths, missing/extra columns, nonnumeric or nonfinite
    features, and unknown targets are rejected. Row indexes are zero-based
    upload record positions; numeric parsing uses pandas' round-trip converter.
    """
    features, target, classes = _schema_columns(schema)
    if not isinstance(csv_bytes, bytes):
        raise ValidationError("Upload must be CSV bytes.")
    if len(csv_bytes) > MAX_UPLOAD_BYTES:
        raise ValidationError("Upload exceeds the 5 MiB limit.")
    try:
        text = csv_bytes.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValidationError("Upload must be valid UTF-8.") from exc
    if not text or "\x00" in text:
        raise ValidationError("Upload is empty or contains NUL characters.")

    # Inspect original records before pandas can rename headers or skip rows.
    reader = csv.reader(StringIO(text, newline=""), strict=True)
    try:
        header = next(reader, None)
        if not header or any(not name.strip() for name in header):
            raise ValidationError("CSV requires a nonempty header.")
        if len(set(header)) != len(header):
            raise ValidationError("CSV header contains duplicate column names.")
        expected = set(features) | {target}
        missing, extra = expected - set(header), set(header) - expected
        if missing or extra:
            raise ValidationError(
                f"CSV columns do not match schema; missing={sorted(missing)}, "
                f"extra={sorted(extra)}."
            )
        row_count = 0
        target_position = header.index(target)
        feature_positions = [header.index(feature) for feature in features]
        for row in reader:
            row_count += 1
            if row_count > MAX_ROWS:
                raise ValidationError("Upload exceeds the 10,000-row limit.")
            if not row or all(not value.strip() for value in row):
                raise ValidationError(f"Blank CSV record at data row {row_count}.")
            if len(row) != len(header):
                raise ValidationError(f"Incorrect field count at data row {row_count}.")
            if row[target_position] not in classes:
                raise ValidationError(
                    f"Missing or unknown target at data row {row_count}."
                )
            # pandas can cast boolean words to floats; reject them as CSV tokens.
            try:
                numeric_values = [
                    float(row[position]) for position in feature_positions
                ]
            except ValueError as exc:
                raise ValidationError(
                    f"Nonnumeric or missing feature at data row {row_count}."
                ) from exc
            if not np.isfinite(numeric_values).all():
                raise ValidationError(f"Nonfinite feature at data row {row_count}.")
        if row_count == 0:
            raise ValidationError("CSV must contain at least one data row.")
    except csv.Error as exc:
        raise ValidationError("Malformed CSV syntax.") from exc

    # Parse numeric values with pandas' converter to preserve saved precision.
    try:
        frame = pd.read_csv(
            StringIO(text, newline=""),
            engine="c",
            dtype={**dict.fromkeys(features, np.float64), target: str},
            keep_default_na=False,
            skip_blank_lines=False,
            on_bad_lines="error",
        )
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValidationError(
            "Features must contain only finite numeric values."
        ) from exc
    if len(frame) != row_count or frame.columns.tolist() != header:
        raise ValidationError("CSV parsers disagree on the uploaded records.")
    ordered_features = frame.loc[:, features].copy()
    if not np.isfinite(ordered_features.to_numpy()).all():
        raise ValidationError("Features must not contain missing or nonfinite values.")
    if not frame[target].isin(classes).all():
        raise ValidationError("Targets must exactly match schema classes.")
    return ValidatedBatch(
        features=ordered_features,
        target=frame[target].copy(),
        fingerprint=hashlib.sha256(csv_bytes).hexdigest(),
    )
