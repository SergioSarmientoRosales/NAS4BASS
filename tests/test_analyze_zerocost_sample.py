from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np


class AnalyzeZeroCostSampleTests(unittest.TestCase):
    def test_partial_spearman_reduces_to_crude_when_params_uncorrelated(self):
        from tools.analyze_zerocost_sample import partial_spearman

        score = np.asarray([1, 2, 3, 4, 5], dtype=float)
        target = np.asarray([1, 2, 3, 4, 5], dtype=float)
        log_params = np.asarray([1, 5, 4, 3, 2], dtype=float)

        partial, score_params = partial_spearman(score, target, log_params)

        self.assertAlmostEqual(score_params, 0.0)
        self.assertAlmostEqual(partial, 1.0)

    def test_bootstrap_returns_ordered_interval(self):
        from tools.analyze_zerocost_sample import percentile_bootstrap

        score = np.asarray([1, 2, 3, 4, 5, 6], dtype=float)
        target = np.asarray([1, 2, 3, 4, 5, 6], dtype=float)
        params = np.asarray([10, 20, 30, 40, 50, 60], dtype=float)

        low, high = percentile_bootstrap(
            score,
            target,
            params,
            metric_name="spearman",
            n_resamples=100,
            seed=7,
        )

        self.assertLessEqual(low, high)
        self.assertGreaterEqual(high, 0.0)

    def test_fixture_pipeline_writes_non_empty_outputs(self):
        from tools.analyze_zerocost_sample import analyze, write_csv_rows

        repo_root = Path(__file__).resolve().parents[1]
        fixture_dir = repo_root / "tests" / "fixtures" / "zerocost_sample"

        global_rows, band_rows, summary = analyze(
            sample_csv=str(fixture_dir / "sample_architectures.csv"),
            score_csvs=[
                str(fixture_dir / "scores_seed_1.csv"),
                str(fixture_dir / "scores_seed_2.csv"),
            ],
            target_column="valid_psnr",
            params_column="params_real",
            bootstrap_resamples=25,
            seed=11,
        )

        self.assertGreater(len(global_rows), 0)
        self.assertGreater(len(band_rows), 0)
        self.assertIn("synflow", summary["predictors"])
        self.assertIn("neg_div_params", summary["transforms"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "global.csv"
            write_csv_rows(output, global_rows, list(global_rows[0]))
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 0)


if __name__ == "__main__":
    unittest.main()
