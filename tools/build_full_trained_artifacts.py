from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


CHANNELS = (16, 32, 48, 64, 16, 32, 48, 64)
KERNELS = (1, 3, 5, 7, 1, 3, 5, 7)
REPEATS = (1, 2, 3, 4, 1, 2, 3, 4)
OPERATIONS = (
    "Conv2D",
    "Dilated Conv d=2",
    "Dilated Conv d=3",
    "Dilated Conv d=4",
    "Depthwise separable",
    "Inverted bottleneck E2",
    "Conv2D transpose",
    "Identity",
)

SUMMARY_FIELDS = (
    "gene_index",
    "params",
    "st1_best_val_psnr",
    "st2_best_val_psnr",
    "st1_best_model",
    "st2_best_model",
    "best_overall_psnr",
    "best_overall_model",
)

COLORS = {
    "Pareto Front": "#2667A8",
    "BASS-50": "#D97706",
    "Big Models": "#138A72",
}
PALE_COLORS = {
    "Pareto Front": "#A9C5E0",
    "BASS-50": "#F3C78B",
    "Big Models": "#9ACFC2",
}
MARKERS = {"Pareto Front": "circle", "BASS-50": "triangle", "Big Models": "square"}


@dataclass(frozen=True)
class Architecture:
    collection: str
    sample_id: str
    gene: tuple[int, ...]
    params: int
    output_dir: Path


@dataclass(frozen=True)
class ResultPoint:
    collection: str
    sample_id: str
    psnr: float
    params: int


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_gene(path: Path) -> tuple[int, ...]:
    payload = load_json(path)
    gene = tuple(int(value) for value in payload["gene"])
    if len(gene) != 28 or any(value < 0 or value > 7 for value in gene):
        raise ValueError(f"Invalid 28-value BASS gene in {path}")
    return gene


def decoded_units(gene: tuple[int, ...]) -> tuple[int, list[list[tuple[int, int, int]]]]:
    channels = CHANNELS[gene[0]]
    units: list[list[tuple[int, int, int]]] = []
    for branch_index in range(3):
        branch: list[tuple[int, int, int]] = []
        offset = 1 + branch_index * 9
        for unit_index in range(3):
            start = offset + unit_index * 3
            operation, kernel_index, repeat_index = gene[start : start + 3]
            branch.append((operation, KERNELS[kernel_index], REPEATS[repeat_index]))
        units.append(branch)
    return channels, units


def count_parameters(gene: tuple[int, ...], scale: int = 2) -> int:
    channels, branches = decoded_units(gene)
    params = 3 * 3 * 3 * channels + channels

    for branch in branches:
        for operation, kernel, repeat in branch:
            if operation <= 3 or operation == 6:
                params += repeat * (kernel * kernel * channels * channels + channels)
            elif operation == 4:
                depthwise = kernel * kernel * channels + channels
                pointwise = channels * channels + channels
                params += repeat * (depthwise + pointwise)
            elif operation == 5:
                expanded = channels * 2
                expand = channels * expanded + expanded
                depthwise = kernel * kernel * expanded + expanded
                project = kernel * kernel * expanded * channels + channels
                params += repeat * (expand + depthwise + project)

    pre_shuffle_channels = 3 * scale * scale
    params += 3 * 3 * channels * pre_shuffle_channels + pre_shuffle_channels
    params += 3 * 3 * 3 * 3 + 3
    return params


def organize_console_logs(strategic_dir: Path) -> int:
    moved = 0
    for source in sorted(strategic_dir.glob("bass_*_x2.console.log")):
        run_id = source.name.removesuffix(".console.log")
        destination_dir = strategic_dir / run_id
        if not destination_dir.is_dir():
            raise FileNotFoundError(f"Missing destination directory for {source.name}")
        destination = destination_dir / source.name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        source.replace(destination)
        moved += 1
    return moved


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def build_strategic_summary(repo_root: Path, strategic_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    gene_dir = repo_root / "data" / "architectures" / "bass_50_sample" / "genes"

    for index in range(1, 51):
        sample_id = f"bass_{index:04d}"
        outer_dir = strategic_dir / f"{sample_id}_x2"
        result_path = outer_dir / sample_id / "result.json"
        history_path = outer_dir / sample_id / "train_log.csv"
        best_model = outer_dir / sample_id / "best.keras"
        gene_path = gene_dir / f"{sample_id}.json"
        for required in (result_path, history_path, best_model, gene_path):
            if not required.is_file():
                raise FileNotFoundError(required)

        result = load_json(result_path)
        if result.get("status") != "complete":
            raise ValueError(f"{sample_id} is not complete: {result.get('status')}")

        gene = load_gene(gene_path)
        params = int(result["params"])
        computed_params = count_parameters(gene)
        if params != computed_params:
            raise ValueError(
                f"Parameter mismatch for {sample_id}: result={params}, gene={computed_params}"
            )

        with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
            history = list(csv.DictReader(handle))
        if not history:
            raise ValueError(f"Empty training history for {sample_id}")
        history_best = max(float(row["val_psnr"]) for row in history)
        best_psnr = float(result["best_val_psnr"])
        if not math.isclose(history_best, best_psnr, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"Best PSNR mismatch for {sample_id}: result={best_psnr}, history={history_best}"
            )

        model_ref = _relative(best_model, repo_root)
        rows.append(
            {
                "gene_index": str(index),
                "params": str(params),
                "st1_best_val_psnr": repr(best_psnr),
                "st2_best_val_psnr": "",
                "st1_best_model": model_ref,
                "st2_best_model": "",
                "best_overall_psnr": repr(best_psnr),
                "best_overall_model": model_ref,
            }
        )

    summary_path = strategic_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if tuple(rows[0]) != SUMMARY_FIELDS:
        raise ValueError(f"Unexpected summary schema in {path}")
    return rows


def architecture_catalog(repo_root: Path) -> list[Architecture]:
    trained_root = repo_root / "DNNs" / "Full Trained"
    catalog: list[Architecture] = []

    pareto_rows = load_summary(trained_root / "Pareto Front" / "summary.csv")
    for row in pareto_rows:
        index = int(row["gene_index"])
        sample_id = f"pareto_{index:04d}"
        gene = load_gene(
            repo_root / "data" / "architectures" / "pareto20" / "genes" / f"{sample_id}.json"
        )
        catalog.append(
            Architecture(
                "Pareto Front",
                sample_id,
                gene,
                int(row["params"]),
                trained_root / "Pareto Front" / f"gene_{index:03d}",
            )
        )

    strategic_rows = load_summary(trained_root / "strategic50" / "summary.csv")
    for row in strategic_rows:
        index = int(row["gene_index"])
        sample_id = f"bass_{index:04d}"
        gene = load_gene(
            repo_root
            / "data"
            / "architectures"
            / "bass_50_sample"
            / "genes"
            / f"{sample_id}.json"
        )
        catalog.append(
            Architecture(
                "BASS-50",
                sample_id,
                gene,
                int(row["params"]),
                trained_root / "strategic50" / f"{sample_id}_x2",
            )
        )

    big_rows = load_summary(trained_root / "Big Models" / "summary.csv")
    for row in big_rows:
        index = int(row["gene_index"])
        sample_id = f"big_{index:04d}"
        gene = load_gene(
            repo_root / "data" / "architectures" / "big_models" / "genes" / f"{sample_id}.json"
        )
        catalog.append(
            Architecture(
                "Big Models",
                sample_id,
                gene,
                int(row["params"]),
                trained_root / "Big Models" / f"gene_{index:03d}",
            )
        )

    expected = {"Pareto Front": 20, "BASS-50": 50, "Big Models": 9}
    actual = {name: sum(item.collection == name for item in catalog) for name in expected}
    if actual != expected:
        raise ValueError(f"Unexpected architecture counts: {actual}")
    for item in catalog:
        if count_parameters(item.gene) != item.params:
            raise ValueError(f"Gene and summary parameters differ for {item.sample_id}")
        if not item.output_dir.is_dir():
            raise FileNotFoundError(item.output_dir)
    return catalog


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf") if bold else Path("C:/Windows/Fonts/calibri.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = "#17202A",
    spacing: int = 7,
) -> None:
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=spacing)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.multiline_text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - bounds[1]),
        text,
        font=font,
        fill=fill,
        align="center",
        spacing=spacing,
    )


def _box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    fill: str,
    outline: str = "#34495E",
    font: ImageFont.ImageFont,
    radius: int = 16,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=3)
    _centered_text(draw, box, text, font)


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: str = "#52616B",
    width: int = 5,
) -> None:
    draw.line((start, end), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 18
    spread = 0.55
    points = [
        end,
        (
            int(end[0] - length * math.cos(angle - spread)),
            int(end[1] - length * math.sin(angle - spread)),
        ),
        (
            int(end[0] - length * math.cos(angle + spread)),
            int(end[1] - length * math.sin(angle + spread)),
        ),
    ]
    draw.polygon(points, fill=fill)


def _operation_fill(operation: int) -> str:
    if operation == 0:
        return "#DCEBFA"
    if operation in (1, 2, 3):
        return "#D9F0F0"
    if operation == 4:
        return "#DFF1DF"
    if operation == 5:
        return "#E9E1F5"
    if operation == 6:
        return "#F9E5CC"
    return "#E8EAED"


def draw_architecture(item: Architecture) -> Path:
    width, height = 2800, 1100
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font = _font(54, bold=True)
    subtitle_font = _font(30)
    box_font = _font(29, bold=True)
    detail_font = _font(27)
    branch_font = _font(26, bold=True)

    draw.text((70, 45), f"{item.collection}: {item.sample_id}", font=title_font, fill="#17202A")
    channels, branches = decoded_units(item.gene)
    subtitle = f"BASS search-space architecture | x2 | {item.params:,} parameters | shared width: {channels}"
    draw.text((72, 115), subtitle, font=subtitle_font, fill="#52616B")

    center_y = 610
    input_box = (55, center_y - 65, 235, center_y + 65)
    stem_box = (310, center_y - 75, 570, center_y + 75)
    unit_x = (720, 1090, 1460)
    branch_y = (340, 610, 880)
    unit_w, unit_h = 300, 150
    merge_box = (1830, center_y - 75, 2000, center_y + 75)
    recon_box = (2080, center_y - 95, 2400, center_y + 95)
    output_box = (2500, center_y - 65, 2745, center_y + 65)

    _arrow(draw, (input_box[2], center_y), (stem_box[0], center_y))
    for y in branch_y:
        draw.line((stem_box[2], center_y, 645, center_y, 645, y, unit_x[0], y), fill="#52616B", width=5)
        _arrow(draw, (unit_x[0] - 40, y), (unit_x[0], y))
        for x1, x2 in zip(unit_x[:-1], unit_x[1:]):
            _arrow(draw, (x1 + unit_w, y), (x2, y))
        draw.line((unit_x[-1] + unit_w, y, 1780, y, 1780, center_y), fill="#52616B", width=5)
    _arrow(draw, (1780, center_y), (merge_box[0], center_y))
    _arrow(draw, (merge_box[2], center_y), (recon_box[0], center_y))
    _arrow(draw, (recon_box[2], center_y), (output_box[0], center_y))

    _box(draw, input_box, "LR\nRGB", fill="#F2F4F6", font=box_font)
    _box(draw, stem_box, f"Stem Conv2D\n3x3, {channels} channels", fill="#DCEBFA", font=box_font)
    _box(draw, merge_box, "Elementwise\nadd", fill="#F2F4F6", font=box_font)
    _box(
        draw,
        recon_box,
        "Conv2D 3x3 -> 12\nPixelShuffle x2",
        fill="#F4E7F5",
        font=box_font,
    )
    _box(draw, output_box, "Conv2D 3x3\nRGB + sigmoid", fill="#F2F4F6", font=box_font)

    for branch_index, (y, branch) in enumerate(zip(branch_y, branches), start=1):
        draw.text((600, y - 105), f"Branch {branch_index}", font=branch_font, fill="#34495E")
        for x, (operation, kernel, repeat) in zip(unit_x, branch):
            box = (x, y - unit_h // 2, x + unit_w, y + unit_h // 2)
            if operation == 7:
                text = "Identity"
            else:
                text = f"{OPERATIONS[operation]}\nk={kernel}, repeat={repeat}"
            _box(draw, box, text, fill=_operation_fill(operation), font=detail_font)

    output_path = item.output_dir / "architecture.png"
    image.save(output_path, format="PNG", dpi=(300, 300), optimize=True)
    return output_path


def summary_points(trained_root: Path) -> list[ResultPoint]:
    points: list[ResultPoint] = []
    sources = (
        ("Pareto Front", trained_root / "Pareto Front" / "summary.csv", "pareto", 4),
        ("BASS-50", trained_root / "strategic50" / "summary.csv", "bass", 4),
        ("Big Models", trained_root / "Big Models" / "summary.csv", "big", 4),
    )
    for collection, path, prefix, digits in sources:
        for row in load_summary(path):
            index = int(row["gene_index"])
            points.append(
                ResultPoint(
                    collection,
                    f"{prefix}_{index:0{digits}d}",
                    float(row["best_overall_psnr"]),
                    int(row["params"]),
                )
            )
    if len(points) != 79:
        raise ValueError(f"Expected 79 summary points, got {len(points)}")
    return points


def nondominated(points: Iterable[ResultPoint]) -> list[ResultPoint]:
    candidates = list(points)
    front: list[ResultPoint] = []
    for point in candidates:
        x = -point.psnr
        dominated = any(
            (-other.psnr <= x and other.params <= point.params)
            and (-other.psnr < x or other.params < point.params)
            for other in candidates
        )
        if not dominated:
            front.append(point)
    return sorted(front, key=lambda point: -point.psnr)


def _format_parameters(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return str(value)


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    marker: str,
    x: int,
    y: int,
    radius: int,
    fill: str,
    *,
    outline: str = "#FFFFFF",
    width: int = 4,
) -> None:
    if marker == "circle":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=width)
    elif marker == "triangle":
        draw.polygon(
            ((x, y - radius - 2), (x - radius, y + radius), (x + radius, y + radius)),
            fill=fill,
            outline=outline,
        )
        draw.line(
            ((x, y - radius - 2), (x - radius, y + radius), (x + radius, y + radius), (x, y - radius - 2)),
            fill=outline,
            width=width,
        )
    else:
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=width)


def _intersects(box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]]) -> bool:
    return any(
        not (
            box[2] < other[0]
            or box[0] > other[2]
            or box[3] < other[1]
            or box[1] > other[3]
        )
        for other in boxes
    )


def draw_scatter(
    points: list[ResultPoint],
    output_path: Path,
    *,
    negative_psnr: bool,
    highlight_front: bool,
    log_parameters: bool,
) -> list[ResultPoint]:
    width, height = 2400, 1600
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font = _font(54, bold=True)
    subtitle_font = _font(30)
    axis_font = _font(34, bold=True)
    tick_font = _font(27)
    legend_font = _font(29)
    label_font = _font(22, bold=True)

    plot_left, plot_top, plot_right, plot_bottom = 235, 225, 2290, 1370
    x_values = [(-point.psnr if negative_psnr else point.psnr) for point in points]
    x_min, x_max = min(x_values), max(x_values)
    x_padding = (x_max - x_min) * 0.035
    x_min -= x_padding
    x_max += x_padding
    if log_parameters:
        y_values = [math.log10(point.params) for point in points]
        y_min, y_max = min(y_values) - 0.08, max(y_values) + 0.08
    else:
        y_min = 0.0
        y_max = float(math.ceil(max(point.params for point in points) / 500_000) * 500_000)

    def px(value: float) -> int:
        return round(plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left))

    def py(params: int) -> int:
        value = math.log10(params) if log_parameters else float(params)
        return round(plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top))

    if negative_psnr:
        title = "Global non-dominated front across all trained architectures"
        subtitle = "Minimization view: -PSNR and parameters"
        x_label = "-PSNR (dB)"
    else:
        title = "Validation performance and model size"
        subtitle = "79 fully trained x2 architectures"
        x_label = "Validation PSNR (dB)"
    subtitle += "; logarithmic parameter axis" if log_parameters else "; linear parameter axis"
    draw.text((plot_left, 55), title, font=title_font, fill="#17202A")
    draw.text((plot_left, 130), subtitle, font=subtitle_font, fill="#52616B")

    x_step = 0.5
    first_tick = math.ceil(x_min / x_step) * x_step
    x_ticks: list[float] = []
    tick = first_tick
    while tick <= x_max + 1e-9:
        x_ticks.append(tick)
        tick += x_step

    if log_parameters:
        y_tick_values = (
            2_000,
            5_000,
            10_000,
            20_000,
            50_000,
            100_000,
            200_000,
            500_000,
            1_000_000,
            2_000_000,
            5_000_000,
        )
    else:
        y_tick_values = tuple(range(0, int(y_max) + 1, 500_000))
    for value in y_tick_values:
        scaled_value = math.log10(value) if log_parameters else float(value)
        if y_min <= scaled_value <= y_max:
            y = py(value)
            draw.line((plot_left, y, plot_right, y), fill="#D9DEE3", width=2)
            label = _format_parameters(value)
            bounds = draw.textbbox((0, 0), label, font=tick_font)
            draw.text((plot_left - 25 - (bounds[2] - bounds[0]), y - 15), label, font=tick_font, fill="#52616B")

    for value in x_ticks:
        x = px(value)
        draw.line((x, plot_top, x, plot_bottom), fill="#E6E9EC", width=2)
        label = f"{value:.1f}"
        bounds = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (bounds[2] - bounds[0]) / 2, plot_bottom + 24), label, font=tick_font, fill="#52616B")

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#263238", width=4)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#263238", width=4)

    x_bounds = draw.textbbox((0, 0), x_label, font=axis_font)
    draw.text(
        ((plot_left + plot_right - (x_bounds[2] - x_bounds[0])) / 2, 1480),
        x_label,
        font=axis_font,
        fill="#263238",
    )
    y_label_image = Image.new("RGBA", (520, 70), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label_image)
    y_axis_label = "Trainable parameters (log scale)" if log_parameters else "Trainable parameters"
    y_draw.text((0, 0), y_axis_label, font=axis_font, fill="#263238")
    rotated = y_label_image.rotate(90, expand=True)
    image.paste(rotated, (35, (plot_top + plot_bottom - rotated.height) // 2), rotated)

    front = nondominated(points)
    front_set = {(point.collection, point.sample_id) for point in front}
    if highlight_front:
        coordinates = [(px(-point.psnr), py(point.params)) for point in front]
        draw.line(coordinates, fill="#1F2933", width=7, joint="curve")

    for point in points:
        is_front = (point.collection, point.sample_id) in front_set
        value_x = -point.psnr if negative_psnr else point.psnr
        color = COLORS[point.collection]
        if highlight_front and not is_front:
            color = PALE_COLORS[point.collection]
        _draw_marker(
            draw,
            MARKERS[point.collection],
            px(value_x),
            py(point.params),
            17 if is_front and highlight_front else 14,
            color,
            outline="#1F2933" if is_front and highlight_front else "#FFFFFF",
            width=5 if is_front and highlight_front else 3,
        )

    legend_x = plot_right - 330 if highlight_front else plot_left + 35
    legend_y = plot_top + 35
    for offset, collection in enumerate(("Pareto Front", "BASS-50", "Big Models")):
        y = legend_y + offset * 52
        _draw_marker(draw, MARKERS[collection], legend_x, y + 15, 13, COLORS[collection], width=3)
        draw.text((legend_x + 32, y), collection, font=legend_font, fill="#263238")
    if highlight_front:
        y = legend_y + 3 * 52
        draw.line((legend_x - 14, y + 15, legend_x + 15, y + 15), fill="#1F2933", width=6)
        draw.text((legend_x + 32, y), "Global Pareto front", font=legend_font, fill="#263238")

    if highlight_front:
        occupied: list[tuple[int, int, int, int]] = [
            (px(-point.psnr) - 24, py(point.params) - 24, px(-point.psnr) + 24, py(point.params) + 24)
            for point in points
        ]
        candidates = ((18, -42), (18, 18), (-145, -42), (-145, 18), (25, -72), (-160, -72))
        for point in front:
            if point.collection == "Pareto Front" and point.sample_id not in {
                "pareto_0001",
                "pareto_0020",
            }:
                continue
            x, y = px(-point.psnr), py(point.params)
            text = point.sample_id
            bounds = draw.textbbox((0, 0), text, font=label_font)
            text_w, text_h = bounds[2] - bounds[0], bounds[3] - bounds[1]
            selected: tuple[int, int, int, int] | None = None
            for dx, dy in candidates:
                box = (x + dx, y + dy, x + dx + text_w + 12, y + dy + text_h + 8)
                if (
                    box[0] >= plot_left
                    and box[2] <= plot_right
                    and box[1] >= plot_top
                    and box[3] <= plot_bottom
                    and not _intersects(box, occupied)
                ):
                    selected = box
                    break
            if selected is None:
                continue
            draw.rounded_rectangle(selected, radius=7, fill="#FFFFFF", outline="#B8C0C7", width=2)
            draw.text((selected[0] + 6, selected[1] + 2), text, font=label_font, fill="#263238")
            occupied.append(selected)

    image.save(output_path, format="PNG", dpi=(300, 300), optimize=True)
    return front


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Organize and visualize the fully trained BASS architecture collections."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    trained_root = repo_root / "DNNs" / "Full Trained"
    strategic_dir = trained_root / "strategic50"

    moved = organize_console_logs(strategic_dir)
    rows = build_strategic_summary(repo_root, strategic_dir)
    catalog = architecture_catalog(repo_root)
    images = [draw_architecture(item) for item in catalog]
    points = summary_points(trained_root)
    draw_scatter(
        points,
        trained_root / "all_architectures_psnr_vs_parameters.png",
        negative_psnr=False,
        highlight_front=False,
        log_parameters=True,
    )
    front = draw_scatter(
        points,
        trained_root / "global_pareto_front_neg_psnr_vs_parameters.png",
        negative_psnr=True,
        highlight_front=True,
        log_parameters=True,
    )
    draw_scatter(
        points,
        trained_root / "all_architectures_psnr_vs_parameters_linear.png",
        negative_psnr=False,
        highlight_front=False,
        log_parameters=False,
    )
    draw_scatter(
        points,
        trained_root / "global_pareto_front_neg_psnr_vs_parameters_linear.png",
        negative_psnr=True,
        highlight_front=True,
        log_parameters=False,
    )
    print(
        f"Moved {moved} console logs; wrote {len(rows)} summary rows; "
        f"generated {len(images)} architecture diagrams and 4 comparison plots; "
        f"global front contains {len(front)} points."
    )


if __name__ == "__main__":
    main()
