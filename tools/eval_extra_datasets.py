from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def discover_images(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    return sorted(item for item in root.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)


def import_tensorflow():
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - depends on local TF install
        raise RuntimeError(
            "TensorFlow is required for extra dataset evaluation. Run this script "
            f"in the training environment. Import error: {exc}"
        ) from exc
    return tf


def load_hr_image(path: Path, *, channels: int):
    tf = import_tensorflow()
    raw = tf.io.read_file(str(path))
    image = tf.io.decode_image(raw, channels=channels, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def crop_to_scale(image, *, scale: int):
    tf = import_tensorflow()
    shape = tf.shape(image)
    height = (shape[0] // scale) * scale
    width = (shape[1] // scale) * scale
    return image[:height, :width, :]


def downsample(image, *, scale: int, method: str):
    tf = import_tensorflow()
    shape = tf.shape(image)
    lr_size = [shape[0] // scale, shape[1] // scale]
    return tf.image.resize(image, lr_size, method=method, antialias=True)


def evaluate_image(model, image_path: Path, *, scale: int, channels: int, downsample_method: str) -> dict:
    tf = import_tensorflow()
    hr = crop_to_scale(load_hr_image(image_path, channels=channels), scale=scale)
    lr = downsample(hr, scale=scale, method=downsample_method)
    sr = model(tf.expand_dims(lr, axis=0), training=False)[0]
    sr = tf.clip_by_value(sr, 0.0, 1.0)
    sr = sr[: tf.shape(hr)[0], : tf.shape(hr)[1], :]
    psnr = tf.image.psnr(hr, sr, max_val=1.0)
    ssim = tf.image.ssim(hr, sr, max_val=1.0)
    return {
        "image": image_path.name,
        "psnr": float(psnr.numpy()),
        "ssim": float(ssim.numpy()),
    }


def load_model(path: str | Path):
    from srir_training.models import load_custom_keras_model

    return load_custom_keras_model(path)


def discover_model_paths(models_dir: str | Path) -> list[Path]:
    root = Path(models_dir)
    if not root.exists():
        raise FileNotFoundError(f"Models directory not found: {root}")
    candidates = sorted(root.rglob("best.keras"))
    if not candidates:
        candidates = sorted(root.rglob("*.keras"))
    if not candidates:
        raise FileNotFoundError(f"No .keras models found under {root}")
    return candidates


def evaluate_model_on_dataset(
    *,
    model_path: Path,
    dataset_name: str,
    dataset_dir: Path,
    sample_id: str,
    scale: int,
    channels: int,
    downsample_method: str,
) -> tuple[list[dict], dict]:
    import_tensorflow()
    model = load_model(model_path)
    rows = []
    for image_path in discover_images(dataset_dir):
        metrics = evaluate_image(
            model,
            image_path,
            scale=scale,
            channels=channels,
            downsample_method=downsample_method,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "model_path": str(model_path.as_posix()),
                "dataset": dataset_name,
                **metrics,
            }
        )

    if rows:
        summary = {
            "sample_id": sample_id,
            "model_path": str(model_path.as_posix()),
            "dataset": dataset_name,
            "n_images": len(rows),
            "psnr_mean": sum(float(row["psnr"]) for row in rows) / len(rows),
            "ssim_mean": sum(float(row["ssim"]) for row in rows) / len(rows),
        }
    else:
        summary = {
            "sample_id": sample_id,
            "model_path": str(model_path.as_posix()),
            "dataset": dataset_name,
            "n_images": 0,
            "psnr_mean": float("nan"),
            "ssim_mean": float("nan"),
        }
    return rows, summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained SRIR/BASS models on Set5, Set14, and BSD100 with trainer-consistent RGB PSNR/SSIM."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-path", default=None)
    group.add_argument("--models-dir", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--scale", type=int, choices=[2, 3, 4], required=True)
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--downsample-method", default="bicubic")
    parser.add_argument("--set5-dir", required=True)
    parser.add_argument("--set14-dir", required=True)
    parser.add_argument("--bsd100-dir", required=True)
    parser.add_argument("--output-csv", default="results/zerocost_50_stratified_random/extra_datasets/eval_extra_datasets.csv")
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_paths = [Path(args.model_path)] if args.model_path else discover_model_paths(args.models_dir)
    datasets = {
        "Set5": Path(args.set5_dir),
        "Set14": Path(args.set14_dir),
        "BSD100": Path(args.bsd100_dir),
    }

    all_rows = []
    summaries = []
    for model_path in model_paths:
        sample_id = args.sample_id or model_path.parent.parent.name or model_path.stem
        for dataset_name, dataset_dir in datasets.items():
            rows, summary = evaluate_model_on_dataset(
                model_path=model_path,
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                sample_id=sample_id,
                scale=args.scale,
                channels=args.channels,
                downsample_method=args.downsample_method,
            )
            all_rows.extend(rows)
            summaries.append(summary)

    output_csv = Path(args.output_csv)
    write_csv(
        output_csv,
        all_rows,
        ["sample_id", "model_path", "dataset", "image", "psnr", "ssim"],
    )

    summary_path = Path(args.summary_json) if args.summary_json else output_csv.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[EVAL] models={len(model_paths)} rows={len(all_rows)}")
    print(f"[EVAL] output_csv={output_csv}")
    print(f"[EVAL] summary_json={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
