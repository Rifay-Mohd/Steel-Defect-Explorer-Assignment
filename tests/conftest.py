"""Read-only fixtures for the frozen experiment; never prepare data or fit models."""

import json
import subprocess

import pandas as pd
import pytest

from src.config import ROOT, TEST_PATH
from src.inference import load_artifacts
from src.validation import validate_upload


def read_metadata(path):
    """Use jq to validate and select the complete trusted metadata document."""
    # Machine-readable JSON requires raw output; snip truncates long JSON lines.
    output = subprocess.check_output(["jq", "-ce", ".", str(path)], text=True)
    return json.loads(output)


@pytest.fixture(scope="session")
def schema():
    """Return the persisted canonical feature and label contract."""
    return read_metadata(ROOT / "metadata/schema.json")


@pytest.fixture(scope="session")
def manifest():
    """Return source-row membership and frozen partition fingerprints."""
    return read_metadata(ROOT / "metadata/split_manifest.json")


@pytest.fixture(scope="session")
def experiment():
    """Return the completed experiment and recorded training-fold identities."""
    return read_metadata(ROOT / "metadata/experiment.json")


@pytest.fixture(scope="session")
def registry():
    """Load real fitted pipelines through production integrity checks."""
    return load_artifacts()


@pytest.fixture(scope="session")
def test_bytes():
    """Read the exact frozen held-out CSV without reserialization."""
    return TEST_PATH.read_bytes()


@pytest.fixture(scope="session")
def batch(test_bytes, schema):
    """Validate the complete held-out partition through the upload contract."""
    return validate_upload(test_bytes, schema)


@pytest.fixture
def csv_records(test_bytes):
    """Return original CSV records as tokens for focused malformed-input cases."""
    import csv
    from io import StringIO

    return list(csv.reader(StringIO(test_bytes.decode("utf-8"))))


@pytest.fixture(scope="session")
def offline_predictions():
    """Read saved probabilities with exact float round-trip parsing."""
    return pd.read_csv(
        ROOT / "results/test_predictions.csv", float_precision="round_trip"
    )
