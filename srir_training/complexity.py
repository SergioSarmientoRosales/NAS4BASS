from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelComplexity:
    params: int
    conv_like_layers: int
    conv2d_layers: int
    depthwise_layers: int
    transpose_layers: int
    max_channels: int
    sum_channels: int
    max_kernel_area: int


def inspect_model_complexity(model) -> ModelComplexity:
    conv_like_layers = 0
    conv2d_layers = 0
    depthwise_layers = 0
    transpose_layers = 0
    max_channels = 1
    sum_channels = 0
    max_kernel_area = 1

    for layer in model.layers:
        name = type(layer).__name__.lower()
        is_conv2d = "conv2d" in name and "transpose" not in name and "depthwise" not in name
        is_depthwise = "depthwise" in name
        is_transpose = "transpose" in name
        if not (is_conv2d or is_depthwise or is_transpose):
            continue

        conv_like_layers += 1
        conv2d_layers += int(is_conv2d)
        depthwise_layers += int(is_depthwise)
        transpose_layers += int(is_transpose)

        filters = getattr(layer, "filters", None)
        if filters is None:
            filters = getattr(layer, "depth_multiplier", 1)
        try:
            filters = int(filters)
        except Exception:
            filters = 1
        max_channels = max(max_channels, filters)
        sum_channels += max(1, filters)

        kernel_size = getattr(layer, "kernel_size", None)
        if kernel_size:
            try:
                max_kernel_area = max(max_kernel_area, int(kernel_size[0]) * int(kernel_size[1]))
            except Exception:
                pass

    return ModelComplexity(
        params=int(model.count_params()),
        conv_like_layers=int(conv_like_layers),
        conv2d_layers=int(conv2d_layers),
        depthwise_layers=int(depthwise_layers),
        transpose_layers=int(transpose_layers),
        max_channels=int(max_channels),
        sum_channels=int(sum_channels),
        max_kernel_area=int(max_kernel_area),
    )


def round_batch(value: int, *, min_batch: int, max_batch: int) -> int:
    value = max(min_batch, min(max_batch, int(value)))
    if value <= min_batch:
        return min_batch
    power = 2 ** int(math.floor(math.log2(value)))
    return max(min_batch, min(max_batch, power))


def estimate_batch_size(
    *,
    free_mb: int,
    complexity: ModelComplexity,
    patch_size: int,
    scale: int,
    precision: str,
    vram_fraction: float,
    min_batch: int,
    max_batch: int,
) -> int:
    usable_mb = max(256.0, float(free_mb) * float(vram_fraction))
    bytes_per_value = 2.0 if precision == "mixed_float16" else 4.0
    lr_patch = max(1, patch_size // max(1, scale))

    activation_proxy = (
        lr_patch
        * lr_patch
        * max(1, complexity.max_channels)
        * max(1, complexity.conv_like_layers)
        * bytes_per_value
    )
    param_proxy = max(1, complexity.params) * bytes_per_value * 4.0
    kernel_proxy = max(1, complexity.max_kernel_area) * max(1, complexity.sum_channels) * bytes_per_value
    per_sample_mb = max(1.0, (activation_proxy + 0.05 * param_proxy + kernel_proxy) / (1024.0 * 1024.0))

    raw_batch = int(usable_mb / per_sample_mb)
    return round_batch(raw_batch, min_batch=min_batch, max_batch=max_batch)
