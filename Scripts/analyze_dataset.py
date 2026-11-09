import argparse
import csv
import math
from collections import Counter
from pathlib import Path


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "ddos_csv_100000.csv"
MISSING_VALUES = {"", "nan", "na", "n/a", "null", "none", "<na>"}


def analyze_dataset(path: Path, label_column: str) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as dataset:
        reader = csv.reader(dataset)
        columns = [column.strip() for column in next(reader, [])]
        if not columns or any(not column for column in columns):
            raise ValueError("The CSV must have a header with column names.")
        if len(columns) != len(set(columns)):
            raise ValueError("The CSV has duplicate column names.")
        if label_column not in columns:
            raise ValueError(f"Label column {label_column!r} not found: {columns}")

        label_index = columns.index(label_column)
        missing = Counter({column: 0 for column in columns})
        infinite = Counter({column: 0 for column in columns})
        labels = Counter()
        rows = 0
        rows_with_missing = 0
        short_rows = 0
        for row in reader:
            if not row:
                continue
            if len(row) > len(columns):
                raise ValueError(f"Record at line {reader.line_num} has extra fields.")
            if len(row) < len(columns):
                short_rows += 1
                row += [""] * (len(columns) - len(row))
            rows += 1
            has_missing = False
            for column, value in zip(columns, row):
                value = value.strip()
                if value.casefold() in MISSING_VALUES:
                    missing[column] += 1
                    has_missing = True
                else:
                    try:
                        if math.isinf(float(value)):
                            infinite[column] += 1
                    except ValueError:
                        pass
            rows_with_missing += int(has_missing)
            label = row[label_index].strip()
            if label.casefold() not in MISSING_VALUES:
                labels[label] += 1

    return {
        "columns": columns,
        "rows": rows,
        "rows_with_missing": rows_with_missing,
        "short_rows": short_rows,
        "missing": missing,
        "infinite": infinite,
        "labels": labels,
        "label_column": label_column,
    }


def print_report(report: dict, threshold: float) -> None:
    rows = report["rows"]
    label_column = report["label_column"]
    features = [column for column in report["columns"] if column != label_column]
    print(f"Records: {rows:,}")
    print(f"Columns: {len(report['columns'])} | Features: {len(features)}")
    print(f"Features: {', '.join(features)}")
    print(f"Records with missing values: {report['rows_with_missing']:,}")
    print(f"Records with missing fields: {report['short_rows']:,}")

    print("\nMISSING VALUES / NaN (including labels)")
    affected = {column: count for column, count in report["missing"].items() if count}
    if not affected:
        print("No missing values found.")
    for column, count in affected.items():
        print(f"{column}: {count:,} ({count / rows:.2%})")
    empty_features = [column for column in features if rows and report["missing"][column] == rows]
    print(f"Entirely empty features: {', '.join(empty_features) or 'none'}")
    print("Columns missing from the schema: a list of expected features is required to check.")

    print("\nINFINITE VALUES (not counted as NaN)")
    affected_infinite = {column: count for column, count in report["infinite"].items() if count}
    if not affected_infinite:
        print("No infinite values found.")
    for column, count in affected_infinite.items():
        print(f"{column}: {count:,} ({count / rows:.2%})")

    print(f"\nLABELS: {label_column}")
    labels = report["labels"]
    valid_labels = sum(labels.values())
    print(f"Observed classes: {len(labels)} | Missing labels: {report['missing'][label_column]:,}")
    for label, count in labels.most_common():
        print(f"{label!r}: {count:,} ({count / valid_labels:.2%} of valid labels)")
    if not rows:
        print("Class balance: cannot be assessed; dataset is empty.")
    elif not labels:
        print("Class balance: cannot be assessed; all labels are missing.")
    elif len(labels) == 1:
        print("Class balance: only one observed class; no class diversity.")
    else:
        ratio = max(labels.values()) / min(labels.values())
        print(f"Largest/smallest class ratio: {ratio:.2f}:1")
        status = "IMBALANCED" if ratio >= threshold else "no imbalance under the criterion"
        print(f"Class balance: {status} (configured threshold: {threshold:g}:1).")
    print("The meaning of the labels depends on the dataset documentation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze missing features and label distribution in a CSV.")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--label", default="Label", help="Label column name (default: Label).")
    parser.add_argument("--imbalance-threshold", type=float, default=2.0, help="Largest/smallest class ratio (default: 2).")
    args = parser.parse_args()
    if not math.isfinite(args.imbalance_threshold) or args.imbalance_threshold <= 1:
        parser.error("--imbalance-threshold must be a finite number greater than 1.")
    try:
        report = analyze_dataset(args.dataset, args.label)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Dataset: {args.dataset.resolve()}\n")
    print_report(report, args.imbalance_threshold)


if __name__ == "__main__":
    main()