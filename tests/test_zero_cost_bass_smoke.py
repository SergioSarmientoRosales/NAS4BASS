from __future__ import annotations

import math
import os
import unittest


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_TF = None
_TF_IMPORT_ERROR = None


def load_tensorflow_or_skip(test_case: unittest.TestCase):
    global _TF, _TF_IMPORT_ERROR

    if _TF is not None:
        return _TF
    if _TF_IMPORT_ERROR is not None:
        test_case.skipTest(_TF_IMPORT_ERROR)

    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None:
        numpy_major = int(str(np.__version__).split(".", 1)[0])
        if numpy_major >= 2:
            _TF_IMPORT_ERROR = (
                "TensorFlow 2.16.x requires numpy<2; "
                f"found numpy {np.__version__}."
            )
            test_case.skipTest(_TF_IMPORT_ERROR)

    try:
        import tensorflow as tf
    except Exception as exc:
        _TF_IMPORT_ERROR = f"TensorFlow is required for zero-cost smoke tests: {exc}"
        test_case.skipTest(_TF_IMPORT_ERROR)

    _TF = tf
    return _TF


class ZeroCostBassSmokeTests(unittest.TestCase):
    def setUp(self):
        tf = load_tensorflow_or_skip(self)

        tf.keras.backend.clear_session()
        tf.random.set_seed(7)
        self.tf = tf

    def tearDown(self):
        self.tf.keras.backend.clear_session()

    @staticmethod
    def genome_covering_all_primitives() -> list[int]:
        ops = [0, 1, 2, 3, 4, 5, 6, 7, 0]
        units = []
        for op in ops:
            units.extend([op, 1, 0])
        return [0] + units

    @staticmethod
    def compact_conv_genome() -> list[int]:
        units = []
        for _ in range(9):
            units.extend([0, 1, 0])
        return [0] + units

    def build_model(self, genome, upscale_factor=2):
        from search_space.model_builder import PixelShuffle, get_model
        from search_space.search_space import decode

        model = get_model(decode(genome), upscale_factor=upscale_factor)
        dummy = self.tf.ones((1, 8, 8, 3), dtype=self.tf.float32)
        output = model(dummy, training=False)
        return model, output, PixelShuffle

    def test_model_builder_covers_representative_bass_operations(self):
        model, output, pixel_shuffle_cls = self.build_model(
            self.genome_covering_all_primitives()
        )

        layer_types = {type(layer) for layer in model.layers}
        self.assertIn(self.tf.keras.layers.Conv2D, layer_types)
        self.assertIn(self.tf.keras.layers.Conv2DTranspose, layer_types)
        self.assertIn(self.tf.keras.layers.DepthwiseConv2D, layer_types)
        self.assertTrue(any(isinstance(layer, pixel_shuffle_cls) for layer in model.layers))
        self.assertEqual(tuple(output.shape), (1, 16, 16, 3))
        self.assertEqual(
            model.layers[-1].activation,
            self.tf.keras.activations.sigmoid,
        )

    def test_model_builder_derives_pixel_shuffle_channels_from_scale(self):
        _, output, _ = self.build_model(
            self.compact_conv_genome(),
            upscale_factor=3,
        )

        self.assertEqual(tuple(output.shape), (1, 24, 24, 3))

    def test_weight_collectors_include_convolutional_parameters(self):
        from evaluators.metrics.grad_norm import _collect_target_weights

        model, _, _ = self.build_model(self.genome_covering_all_primitives())
        weights = _collect_target_weights(model)

        self.assertGreater(len(weights), 0)
        self.assertEqual(len({id(weight) for weight in weights}), len(weights))
        self.assertTrue(any("depthwise" in str(getattr(w, "path", getattr(w, "name", ""))) for w in weights))

    def test_selected_zero_cost_metrics_return_scalar_values(self):
        from evaluators.metrics.l2_norm import compute_l2_norm
        from evaluators.metrics.params_score import compute_param_score
        from evaluators.metrics.synflow import compute_synflow

        model, _, _ = self.build_model(self.compact_conv_genome())
        inputs = self.tf.ones((2, 8, 8, 3), dtype=self.tf.float32)

        param_score = compute_param_score(model=model, inputs=inputs)
        scores = [
            param_score,
            compute_l2_norm(model=model, inputs=inputs),
            compute_synflow(model=model, inputs=inputs),
        ]

        for score in scores:
            self.assertIsInstance(float(score), float)
            self.assertTrue(math.isfinite(float(score)))

        self.assertLess(param_score, 0.0)

    def test_zero_cost_evaluator_is_stable_for_repeated_architecture(self):
        from evaluators.zero_cost import ZeroCostEvaluator

        arch_a = self.compact_conv_genome()
        arch_b = self.genome_covering_all_primitives()
        evaluator = ZeroCostEvaluator(
            metric_name="l2_norm",
            input_shape=(8, 8, 3),
            batch_size=2,
            seed=11,
        )

        score_a_first = evaluator.evaluate_with_params(arch_a, n_eval=1)["score"]
        _ = evaluator.evaluate_with_params(arch_b, n_eval=2)["score"]
        score_a_second = evaluator.evaluate_with_params(arch_a, n_eval=3)["score"]

        self.assertAlmostEqual(score_a_first, score_a_second, places=6)


if __name__ == "__main__":
    unittest.main()
