from __future__ import annotations

import argparse
import ast
import csv
import glob
import math
import os
from collections import Counter
from dataclasses import dataclass, field


GENE_COLUMNS = {"Net", "gene", "Gene", "genotype", "architecture", "arch"}
NUMERIC_COLUMNS = {"params", "valid_psnr", "train_psnr", "psnr", "PSNR", "Valid_PSNR"}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def extend(self, other: "ValidationReport") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


def parse_gene(value: str, *, expected_len: int) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        raise ValueError("empty gene field")

    if text.startswith("[") or text.startswith("("):
        gene = [int(v) for v in ast.literal_eval(text)]
    else:
        gene = [int(v.strip()) for v in text.split(",") if v.strip()]

    if len(gene) != expected_len:
        raise ValueError(f"expected {expected_len} genes, got {len(gene)}")

    out = tuple(gene)
    bad_values = [v for v in out if v < 0 or v > 7]
    if bad_values:
        raise ValueError(f"gene values outside [0, 7]: {sorted(set(bad_values))}")

    return out


def row_is_header(row: list[str], *, expected_len: int) -> bool:
    if not row:
        return False

    clean_row = [cell.strip().lstrip("\ufeff") for cell in row]
    if any(cell in GENE_COLUMNS for cell in clean_row):
        return True

    if len(clean_row) != expected_len:
        return True

    try:
        parse_gene(",".join(clean_row), expected_len=expected_len)
    except Exception:
        return True

    return False


def validate_numeric_cell(
    *,
    report: ValidationReport,
    path: str,
    line_no: int,
    column: str,
    value: str,
) -> None:
    try:
        number = float(value)
    except Exception:
        report.error(f"{path}:{line_no}: column '{column}' is not numeric: {value!r}")
        return

    if not math.isfinite(number):
        if column == "train_psnr":
            report.warning(
                f"{path}:{line_no}: column '{column}' is non-finite: {value!r}"
            )
            return

        report.error(f"{path}:{line_no}: column '{column}' is not finite: {value!r}")
        return

    if column == "params" and number < 0:
        report.error(f"{path}:{line_no}: params must be non-negative, got {value!r}")


def validate_csv_file(
    path: str,
    *,
    expected_len: int,
    fail_on_duplicates: bool,
) -> ValidationReport:
    report = ValidationReport()

    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        report.error(f"{path}: empty CSV file")
        return report

    has_header = row_is_header(rows[0], expected_len=expected_len)
    header = [cell.strip().lstrip("\ufeff") for cell in rows[0]] if has_header else []
    data_rows = rows[1:] if has_header else rows

    if has_header:
        gene_idx = next((idx for idx, name in enumerate(header) if name in GENE_COLUMNS), None)
        if gene_idx is None:
            report.error(
                f"{path}: no architecture column found. Expected one of {sorted(GENE_COLUMNS)}"
            )
            return report
        numeric_indices = [
            (idx, name)
            for idx, name in enumerate(header)
            if name in NUMERIC_COLUMNS
        ]
    else:
        gene_idx = None
        numeric_indices = []

    genes: list[tuple[int, ...]] = []

    for offset, row in enumerate(data_rows, start=2 if has_header else 1):
        if not row or all(not cell.strip() for cell in row):
            continue

        try:
            if has_header:
                gene = parse_gene(row[gene_idx], expected_len=expected_len)
            else:
                gene = parse_gene(",".join(row), expected_len=expected_len)
            genes.append(gene)
        except Exception as exc:
            report.error(f"{path}:{offset}: invalid architecture: {exc}")

        for idx, column in numeric_indices:
            if idx >= len(row):
                report.error(f"{path}:{offset}: missing numeric column '{column}'")
                continue
            validate_numeric_cell(
                report=report,
                path=path,
                line_no=offset,
                column=column,
                value=row[idx],
            )

    duplicates = sum(count - 1 for count in Counter(genes).values() if count > 1)
    if duplicates:
        message = f"{path}: contains {duplicates} duplicate architecture rows"
        if fail_on_duplicates:
            report.error(message)
        else:
            report.warning(message)

    print(
        f"[DATA] {path}: rows={len(data_rows)}, architectures={len(genes)}, "
        f"unique={len(set(genes))}, header={has_header}"
    )

    return report


def validate_data_dir(
    data_dir: str,
    *,
    expected_len: int,
    fail_on_duplicates: bool,
) -> ValidationReport:
    report = ValidationReport()
    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    if not csv_paths:
        report.error(f"No CSV files found in {data_dir!r}")
        return report

    for path in csv_paths:
        report.extend(
            validate_csv_file(
                path,
                expected_len=expected_len,
                fail_on_duplicates=fail_on_duplicates,
            )
        )

    return report


def validate_models_dir(
    models_dir: str,
    *,
    expected_len: int,
) -> ValidationReport:
    report = ValidationReport()
    model_paths = sorted(glob.glob(os.path.join(models_dir, "*.pkl")))

    if not model_paths:
        report.error(f"No .pkl model files found in {models_dir!r}")
        return report

    try:
        import joblib
        import numpy as np
    except Exception as exc:
        report.error(f"Model validation requires joblib and numpy: {exc}")
        return report

    dummy = np.zeros((1, expected_len), dtype=np.float32)

    for path in model_paths:
        name = os.path.basename(path)
        try:
            model = joblib.load(path)
        except Exception as exc:
            report.error(f"{path}: failed to load model: {exc}")
            continue

        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None and int(n_features) != expected_len:
            report.error(
                f"{path}: expected n_features_in_={expected_len}, got {n_features}"
            )

        try:
            pred = model.predict(dummy)
            pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
        except Exception as exc:
            report.error(f"{path}: prediction failed on {expected_len}-gene input: {exc}")
            continue

        if pred_arr.size == 0:
            report.error(f"{path}: prediction output is empty")
        elif not np.all(np.isfinite(pred_arr)):
            report.error(f"{path}: prediction output contains non-finite values")

        print(
            f"[MODEL] {name}: type={type(model).__name__}, "
            f"n_features_in_={n_features}, pred_shape={tuple(np.asarray(pred).shape)}"
        )

    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate NAS4BASS curated CSV data and surrogate models."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing CSV files")
    parser.add_argument("--models-dir", default="models", help="Directory containing .pkl models")
    parser.add_argument("--expected-len", type=int, default=28, help="Expected decoded gene length")
    parser.add_argument("--skip-data", action="store_true", help="Skip CSV validation")
    parser.add_argument("--skip-models", action="store_true", help="Skip .pkl model validation")
    parser.add_argument(
        "--fail-on-duplicates",
        action="store_true",
        help="Treat duplicate architecture rows as errors instead of warnings",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = ValidationReport()

    if not args.skip_data:
        report.extend(
            validate_data_dir(
                args.data_dir,
                expected_len=args.expected_len,
                fail_on_duplicates=args.fail_on_duplicates,
            )
        )

    if not args.skip_models:
        report.extend(
            validate_models_dir(
                args.models_dir,
                expected_len=args.expected_len,
            )
        )

    for warning in report.warnings:
        print(f"[WARNING] {warning}")

    for error in report.errors:
        print(f"[ERROR] {error}")

    if report.errors:
        print(f"[FAIL] validation failed with {len(report.errors)} error(s)")
        return 1

    print(f"[OK] validation completed with {len(report.warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
