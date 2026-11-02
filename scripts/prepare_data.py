"""Prepare immutable local data with ``python -m scripts.prepare_data``."""

from src.data_preparation import prepare_data


def main() -> None:
    """Run the source audit and frozen split, printing the actual findings."""
    audit = prepare_data()
    print(
        f"Audited {audit.summary['source_shape']}: {len(audit.features)} features, seven one-hot targets."
    )
    print(f"Source target order: {audit.source_targets}")
    print(f"Class counts: {audit.summary['class_counts']}")
    print(
        f"Exact duplicates: {audit.summary['exact_duplicate_rows']}; conflicting rows: {audit.summary['conflicting_label_rows']}."
    )
    print("Frozen stratified split: 1552 training / 389 test rows; random seed 42.")


if __name__ == "__main__":
    main()
