from __future__ import annotations

import unittest

import pandas as pd

from tools.analyze_zerocost_rework import (
    apply_transforms,
    calculate_seed_metrics,
    summarize_seed_metrics,
    transformation_stability,
)
from tools.run_zerocost_rework import parse_metrics, parse_seed_spec


class ZeroCostReworkTests(unittest.TestCase):
    def test_seed_ranges_are_inclusive_and_unique(self):
        self.assertEqual(parse_seed_spec("1:3,7,9:11"), [1, 2, 3, 7, 9, 10, 11])
        with self.assertRaises(ValueError):
            parse_seed_spec("1:3,3")

    def test_metric_parser_rejects_unknown_proxy(self):
        self.assertEqual(parse_metrics("l2_norm,param_score"), ["l2_norm", "param_score"])
        with self.assertRaises(ValueError):
            parse_metrics("l2_norm,not_a_proxy")

    def test_complete_seed_metrics_preserve_pairing(self):
        manifest = pd.DataFrame(
            {
                "scenario": ["expanded50"] * 4,
                "architecture_id": ["a", "b", "c", "d"],
                "valid_psnr": [30.0, 31.0, 32.0, 33.0],
                "params_real": [100, 200, 400, 800],
                "complexity_bin": [1, 1, 2, 2],
            }
        )
        rows = []
        for seed in (1, 2):
            for index, architecture_id in enumerate(("a", "b", "c", "d"), start=1):
                rows.append(
                    {
                        "scenario": "expanded50",
                        "architecture_id": architecture_id,
                        "seed": seed,
                        "proxy": "l2_norm",
                        "valid_psnr": 29.0 + index,
                        "params_built": 100 * (2 ** (index - 1)),
                        "raw_score": float(index + seed / 100.0),
                        "validity_flag": True,
                    }
                )
        transformed = apply_transforms(pd.DataFrame(rows))
        metrics = calculate_seed_metrics(transformed, {"expanded50": manifest})
        raw = metrics[metrics["transformation"] == "raw"]
        self.assertEqual(len(raw), 2)
        self.assertTrue(raw["complete"].all())
        self.assertTrue((raw["kendall_tau_b"] == 1.0).all())
        self.assertTrue((raw["within_bin_kendall_tau_b"] == 1.0).all())
        self.assertTrue(raw["partial_spearman_log_params"].isna().all())

        summary = summarize_seed_metrics(
            metrics,
            bootstrap_seed=17,
            bootstrap_resamples=100,
        )
        raw_summary = summary[summary["transformation"] == "raw"].iloc[0]
        self.assertEqual(int(raw_summary["n_complete_seeds"]), 2)
        self.assertAlmostEqual(float(raw_summary["kendall_tau_b_mean"]), 1.0)

        winners, stability = transformation_stability(metrics)
        self.assertEqual(len(winners), 2)
        raw_stability = stability[stability["transformation"] == "raw"].iloc[0]
        self.assertEqual(int(raw_stability["winning_seeds"]), 2)
        self.assertAlmostEqual(float(raw_stability["winning_seed_fraction"]), 1.0)


if __name__ == "__main__":
    unittest.main()
