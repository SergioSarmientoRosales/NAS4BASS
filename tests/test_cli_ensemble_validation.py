from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class DummyModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, x):
        return np.full((len(x),), self.value, dtype=np.float64)


class CliAndEnsembleTests(unittest.TestCase):
    def test_main_help_uses_lightweight_import_path(self):
        repo_root = Path(__file__).resolve().parents[1]

        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--ensemble-weights", result.stdout)
        self.assertIn("cmopso", result.stdout)
        self.assertIn("imia", result.stdout)
        self.assertIn("sms_emoa", result.stdout)

    def test_search_registry_exposes_new_methods(self):
        from core.registry import SEARCH_CLI_CHOICES, build_search_method

        self.assertIn("cmopso", SEARCH_CLI_CHOICES)
        self.assertIn("imia", SEARCH_CLI_CHOICES)
        self.assertIn("sms_emoa", SEARCH_CLI_CHOICES)
        self.assertIn("sms-emoa", SEARCH_CLI_CHOICES)

        dummy_problem = object()
        cases = {
            "cmopso": "CMOPSO",
            "imia": "IMIA",
            "sms-emoa": "SMSEMOA",
        }
        for search_name, class_name in cases.items():
            search = build_search_method(
                search_name,
                problem=dummy_problem,
                pop_size=10,
                n_gen=1,
            )
            self.assertEqual(type(search).__name__, class_name)

    def test_early_stop_rejects_searches_without_support(self):
        from core.registry import build_search_method

        with self.assertRaises(ValueError):
            build_search_method(
                "cmopso",
                problem=object(),
                pop_size=10,
                n_gen=1,
                early_stop=True,
            )

    def test_parse_named_and_positional_ensemble_weights(self):
        from main import parse_ensemble_weights

        self.assertEqual(parse_ensemble_weights("0.25,0.75"), [0.25, 0.75])
        self.assertEqual(
            parse_ensemble_weights("xgb=0.25,rf=0.75"),
            {"xgb": 0.25, "rf": 0.75},
        )

        with self.assertRaises(ValueError):
            parse_ensemble_weights("xgb=0.25,0.75")

    def test_selected_models_preserve_requested_order_for_named_weights(self):
        from predictors.ensemble import SurrogateEnsemble
        from predictors.selectors import select_surrogate_models

        models = [DummyModel(1.0), DummyModel(10.0), DummyModel(100.0)]
        model_names = ["xgb", "rf", "svr"]

        selected_models, selected_names = select_surrogate_models(
            models=models,
            model_names=model_names,
            selected_names=["rf", "xgb"],
        )

        self.assertEqual(selected_names, ["rf", "xgb"])

        ensemble = SurrogateEnsemble(
            models=selected_models,
            model_names=selected_names,
            method="weighted_mean",
            weights={"xgb": 0.25, "rf": 0.75},
        )

        pred_info = ensemble.predict_with_stats(np.zeros((1, 28), dtype=np.float32))

        self.assertAlmostEqual(pred_info["ensemble_prediction"], 7.75)
        self.assertEqual(pred_info["ensemble_weights"], {"rf": 0.75, "xgb": 0.25})

    def test_named_weights_must_match_active_models(self):
        from predictors.ensemble import SurrogateEnsemble

        with self.assertRaises(ValueError):
            SurrogateEnsemble(
                models=[DummyModel(1.0)],
                model_names=["xgb"],
                method="weighted_mean",
                weights={"rf": 1.0},
            )


class ArtifactValidationTests(unittest.TestCase):
    def test_bass_descriptor_uses_composition_features(self):
        from tools.sample_bass_architectures import descriptor_matrix

        gene = [0]
        for _ in range(9):
            gene.extend([0, 1, 0])

        matrix = descriptor_matrix([tuple(gene)], standardize=False)

        self.assertEqual(matrix.shape, (1, 26))
        self.assertAlmostEqual(matrix[0, 0], 1.0)
        self.assertAlmostEqual(matrix[0, 8 + 1], 1.0)
        self.assertAlmostEqual(matrix[0, 16], 1.0)
        self.assertAlmostEqual(matrix[0, 24], 16 / 64)

    def test_max_min_can_reuse_pool_standardized_features(self):
        from tools.sample_bass_architectures import greedy_max_min_select

        genes = [tuple([0] * 28) for _ in range(3)]
        features = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 4.0],
            ],
            dtype=np.float64,
        )

        selected, distances = greedy_max_min_select(
            genes,
            n_select=2,
            features=features,
        )

        self.assertEqual(selected, [0, 2])
        self.assertAlmostEqual(distances[1], 4.0)

    def test_proxy_alignment_matrix_counts_complexity_by_param_quintile(self):
        from tools.validate_bass_sample import proxy_alignment_matrix

        complexity_bins = [0, 0, 1, 1, 2]
        params_real = [10, 100, 1000, 10000, 100000]

        matrix = proxy_alignment_matrix(complexity_bins, params_real, n_bins=3)

        self.assertEqual(len(matrix), 3)
        self.assertEqual(sum(sum(row) for row in matrix), 5)
        self.assertEqual(sum(matrix[0]), 2)
        self.assertEqual(sum(matrix[1]), 2)
        self.assertEqual(sum(matrix[2]), 1)

    def test_validate_csv_accepts_header_and_reports_duplicates(self):
        from tools.validate_artifacts import validate_csv_file

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "models.csv"
            gene = list(range(8)) * 3 + [0, 1, 2, 3]

            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Net", "params", "valid_psnr"])
                writer.writerow([str(gene), "1234", "31.5"])
                writer.writerow([str(gene), "1234", "31.5"])

            report = validate_csv_file(
                str(path),
                expected_len=28,
                fail_on_duplicates=False,
            )

        self.assertEqual(report.errors, [])
        self.assertEqual(len(report.warnings), 1)
        self.assertIn("duplicate architecture", report.warnings[0])

    def test_bass_sampler_writes_trainer_manifest_without_psnr(self):
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            arch_csv = tmp / "architectures.csv"
            manifest_csv = tmp / "manifest.csv"
            metadata_json = tmp / "metadata.json"
            genes_dir = tmp / "genes"

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/sample_bass_architectures.py",
                    "--n",
                    "6",
                    "--pool-size",
                    "80",
                    "--seed",
                    "123",
                    "--output-csv",
                    str(arch_csv),
                    "--trainer-manifest",
                    str(manifest_csv),
                    "--metadata-json",
                    str(metadata_json),
                    "--gene-json-dir",
                    str(genes_dir),
                ],
                cwd=repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)

            with arch_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            genes = {row["Net"] for row in rows}
            self.assertEqual(len(rows), 6)
            self.assertEqual(len(genes), 6)
            self.assertFalse(any("psnr" in column.lower() for column in rows[0]))
            self.assertIn("selection_min_distance", rows[0])
            self.assertIn("selection_min_distance_raw", rows[0])
            self.assertIn("params_real", rows[0])

            with manifest_csv.open(newline="", encoding="utf-8") as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual(len(manifest), 6)
            self.assertIn("--bass-gene-file", manifest[0]["train_command"])
            self.assertIn("tools/eval_extra_datasets.py", manifest[0]["eval_extra_command"])
            self.assertTrue(Path(manifest[0]["bass_gene_file"]).exists())

            with metadata_json.open(encoding="utf-8") as handle:
                metadata = __import__("json").load(handle)
            self.assertEqual(metadata["selection_method"], "stratified_random")
            self.assertEqual(sum(metadata["pool_band_counts"].values()), 80)


if __name__ == "__main__":
    unittest.main()
