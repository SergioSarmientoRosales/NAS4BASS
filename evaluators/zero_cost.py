from __future__ import annotations

import hashlib
import math

import tensorflow as tf

from evaluators.base import BaseEvaluator
from search_space.search_space import decode
from search_space.model_builder import get_model

from evaluators.metrics.params_score import compute_param_score
from evaluators.metrics.synflow import compute_synflow
from evaluators.metrics.fisher import compute_fisher
from evaluators.metrics.grad_norm import compute_grad_norm
from evaluators.metrics.snip import compute_snip
from evaluators.metrics.jacob_cov import compute_jacob_cov
from evaluators.metrics.grasp import compute_grasp
from evaluators.metrics.l2_norm import compute_l2_norm
from evaluators.metrics.plain import compute_plain
from evaluators.metrics.nwot import compute_nwot
from evaluators.metrics.zen import compute_zen_score
from evaluators.metrics.zico import compute_zico


class ZeroCostEvaluator(BaseEvaluator):
    def __init__(
        self,
        metric_name: str = "synflow",
        score_transform: str = "raw",
        verbose: bool = True,
        input_shape: tuple[int, int, int] = (64, 64, 3),
        batch_size: int = 8,
        upscale_factor: int = 2,
        seed: int = 1,
        deterministic_arch_seed: bool = True,
    ):
        self.metric_name = metric_name
        self.score_transform = score_transform
        self.verbose = verbose
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.upscale_factor = upscale_factor
        self.seed = int(seed)
        self.deterministic_arch_seed = bool(deterministic_arch_seed)

        self._dummy_inputs = None
        self._dummy_targets = None
        self._dummy_loss_fn = tf.keras.losses.MeanSquaredError()

        valid_metrics = {
            "param_score",
            "synflow",
            "nwot",
            "fisher",
            "grad_norm",
            "grasp",
            "jacob_cov",
            "l2_norm",
            "plain",
            "snip",
            "zen",
            "zico",
        }
        if self.metric_name not in valid_metrics:
            raise ValueError(
                f"Unknown zero-cost metric '{self.metric_name}'. "
                f"Available options: {sorted(valid_metrics)}"
            )

        valid_transforms = {"raw", "div_params", "neg_raw", "neg_div_params"}
        if self.score_transform not in valid_transforms:
            raise ValueError(
                f"Unknown score transform '{self.score_transform}'. "
                f"Available options: {sorted(valid_transforms)}"
            )

    def _architecture_seed(self, decoded_ind: list[int]) -> int:
        payload = f"{self.seed}|{self.metric_name}|{','.join(map(str, decoded_ind))}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _set_architecture_seed(self, decoded_ind: list[int]) -> None:
        if self.deterministic_arch_seed:
            tf.random.set_seed(self._architecture_seed(decoded_ind))

    def _get_shared_sr_context(self):
        if self._dummy_inputs is None or self._dummy_targets is None:
            tf.random.set_seed(self.seed)

            h, w, c = self.input_shape
            hr_h = h * self.upscale_factor
            hr_w = w * self.upscale_factor

            self._dummy_inputs = tf.random.uniform(
                shape=(self.batch_size, h, w, c),
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32,
            )
            self._dummy_targets = tf.random.uniform(
                shape=(self.batch_size, hr_h, hr_w, c),
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32,
            )

        return self._dummy_inputs, self._dummy_targets, self._dummy_loss_fn

    def _compute_raw_score(self, model: tf.keras.Model) -> float:
        inputs, targets, loss_fn = self._get_shared_sr_context()

        if self.metric_name == "param_score":
            return float(compute_param_score(model=model))

        elif self.metric_name == "synflow":
            return float(
                compute_synflow(
                    model=model,
                    input_shape=self.input_shape,
                    batch_size=self.batch_size,
                )
            )

        elif self.metric_name == "fisher":
            return float(
                compute_fisher(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    loss_fn=loss_fn,
                )
            )

        elif self.metric_name == "grad_norm":
            return float(
                compute_grad_norm(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    loss_fn=loss_fn,
                )
            )

        elif self.metric_name == "grasp":
            return float(
                compute_grasp(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    loss_fn=loss_fn,
                )
            )

        elif self.metric_name == "snip":
            return float(
                compute_snip(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    loss_fn=loss_fn,
                )
            )

        elif self.metric_name == "jacob_cov":
            return float(
                compute_jacob_cov(
                    model=model,
                    inputs=inputs,
                )
            )

        elif self.metric_name == "l2_norm":
            return float(compute_l2_norm(model=model))

        elif self.metric_name == "plain":
            return float(
                compute_plain(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    loss_fn=loss_fn,
                )
            )

        elif self.metric_name == "nwot":
            return float(
                compute_nwot(
                    model=model,
                    inputs=inputs,
                )
            )

        elif self.metric_name == "zen":
            return float(
                compute_zen_score(
                    model=model,
                    inputs=inputs,
                )
            )

        elif self.metric_name == "zico":
            return float(
                compute_zico(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    loss_fn=loss_fn,
                )
            )

        raise ValueError(f"Unknown zero-cost metric '{self.metric_name}'")

    def _apply_score_transform(self, raw_score: float, params: int) -> float:
        raw_score = float(raw_score)
        params = int(params)

        if not math.isfinite(raw_score) or params <= 0:
            return float("nan")

        if self.score_transform == "raw":
            return raw_score
        if self.score_transform == "div_params":
            return raw_score / float(params)
        if self.score_transform == "neg_raw":
            return -raw_score
        if self.score_transform == "neg_div_params":
            return -raw_score / float(params)

        raise RuntimeError(f"Unhandled score transform '{self.score_transform}'")

    def evaluate_with_params(self, decoded_ind: list[int], n_eval: int) -> dict:
        del n_eval

        self._get_shared_sr_context()
        self._set_architecture_seed(decoded_ind)

        genotype = decode(decoded_ind)
        model = get_model(genotype, upscale_factor=self.upscale_factor)

        try:
            params = int(model.count_params())
            raw_score = self._compute_raw_score(model)
            score = self._apply_score_transform(raw_score, params)

        finally:
            try:
                del model
            except Exception:
                pass
            tf.keras.backend.clear_session()

        return {
            "score": float(score),
            "params": int(params),
            "details": {
                "metric_name": self.metric_name,
                "raw_score": float(raw_score),
                "score_transform": self.score_transform,
                "deterministic_arch_seed": self.deterministic_arch_seed,
                "score_type": "zero_cost",
            },
        }

    def evaluate(self, decoded_ind: list[int], n_eval: int) -> dict:
        result = self.evaluate_with_params(decoded_ind, n_eval)
        return {
            "score": float(result["score"]),
            "details": result["details"],
        }
