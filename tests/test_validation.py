"""Exercise original CSV records before pandas can rename, skip, or coerce them."""

import csv
from hashlib import sha256
from io import StringIO

import numpy as np
import pytest

from src.validation import MAX_ROWS, MAX_UPLOAD_BYTES, ValidationError, validate_upload


def encode_records(records):
    """Serialize controlled CSV tokens while retaining intentional duplicate headers."""
    stream = StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(records)
    return stream.getvalue().encode()


@pytest.mark.parametrize(
    "token",
    [
        "True",
        "False",
        "true",
        "FALSE",
        "NaN",
        "nan",
        "inf",
        "-inf",
        "Infinity",
        "1e309",
        "text",
        "",
        " ",
    ],
)
def test_invalid_numeric_tokens_rejected(csv_records, schema, token):
    """Reject booleans, missing data, text, NaNs, and infinite numeric tokens."""
    csv_records[1][0] = token
    with pytest.raises(ValidationError, match="feature"):
        validate_upload(encode_records(csv_records[:2]), schema)


@pytest.mark.parametrize(
    "case",
    ["duplicate", "missing", "extra", "indicator", "all_indicators", "header_space"],
)
def test_invalid_original_headers_rejected(csv_records, schema, case):
    """Reject duplicate raw headers and every route for target-indicator leakage."""
    records = csv_records[:2]
    if case == "duplicate":
        records[0][1] = records[0][0]
        message = "duplicate"
    else:
        message = "columns"
        if case == "missing":
            records = [row[1:] for row in records]
        elif case == "header_space":
            records[0][0] += " "
        else:
            if case == "all_indicators":
                extras = schema["source_target_order"]
            elif case == "indicator":
                extras = [schema["classes"][0]]
            else:
                extras = ["unexpected"]
            records[0].extend(extras)
            records[1].extend(["0"] * len(extras))
    with pytest.raises(ValidationError, match=message):
        validate_upload(encode_records(records), schema)


@pytest.mark.parametrize(
    "target", ["", "UNKNOWN", "K_Scratch", " Bumps", "Bumps ", "bumps", "NaN"]
)
def test_unknown_or_missing_targets_rejected(csv_records, schema, target):
    """Require exact canonical source labels without trimming or spelling repair."""
    csv_records[1][csv_records[0].index(schema["target"])] = target
    with pytest.raises(ValidationError, match="target"):
        validate_upload(encode_records(csv_records[:2]), schema)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "header_only",
        "blank_row",
        "short_row",
        "long_row",
        "unclosed_quote",
        "nul",
        "invalid_utf8",
    ],
)
def test_malformed_or_empty_csv_rejected(csv_records, schema, case):
    """Reject empty uploads, malformed syntax, invalid encoding, and bad row widths."""
    header, row = csv_records[:2]
    payloads = {
        "empty": b"",
        "header_only": encode_records([header]),
        "blank_row": encode_records([header, row, []]),
        "short_row": encode_records([header, row[:-1]]),
        "long_row": encode_records([header, row + ["0"]]),
        "unclosed_quote": encode_records([header]) + b'"unterminated',
        "nul": encode_records([header, row]) + b"\x00",
        "invalid_utf8": b"\xff" + encode_records([header, row]),
    }
    with pytest.raises(ValidationError):
        validate_upload(payloads[case], schema)


def test_row_cap_accepts_boundary_rejects_next_record(csv_records, schema):
    """Exercise the actual 10,000-record cap without reducing production limits."""
    header, row = csv_records[:2]
    accepted = validate_upload(encode_records([header] + [row] * MAX_ROWS), schema)
    assert accepted.row_count == MAX_ROWS
    with pytest.raises(ValidationError, match="10,000-row"):
        validate_upload(encode_records([header] + [row] * (MAX_ROWS + 1)), schema)


def test_byte_cap_accepts_boundary_rejects_next_byte(csv_records, schema):
    """Pad bounded numeric fields to exactly 5 MiB and reject one additional byte."""
    header, row = csv_records[:2]
    records = [header] + [row.copy() for _ in range(100)]
    padding, remainder = divmod(MAX_UPLOAD_BYTES - len(encode_records(records)), 100)
    for index, record in enumerate(records[1:]):
        record[0] = " " * (padding + (index < remainder)) + record[0]
    boundary = encode_records(records)
    assert len(boundary) == MAX_UPLOAD_BYTES
    assert validate_upload(boundary, schema).row_count == 100
    with pytest.raises(ValidationError, match="5 MiB"):
        validate_upload(boundary + b"\n", schema)


def test_bom_and_row_order_preserve_numeric_bits(csv_records, schema):
    """Accept a BOM and preserve difficult floats and original record positions."""
    records = csv_records[:3]
    records[1][0] = "0.10000000000000002"
    records[2][0] = "-0.0"
    payload = b"\xef\xbb\xbf" + encode_records(records)
    batch = validate_upload(payload, schema)
    expected = np.array([0.10000000000000002, -0.0])
    np.testing.assert_array_equal(
        batch.features.iloc[:, 0].to_numpy().view(np.uint64), expected.view(np.uint64)
    )
    assert batch.features.index.tolist() == batch.target.index.tolist() == [0, 1]
    assert batch.target.tolist() == [row[-1] for row in records[1:]]
    assert batch.fingerprint == sha256(payload).hexdigest()
