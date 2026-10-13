import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from stipsd.completion import METHODS, SCENARIOS, complete, make_mask
from stipsd.data import load_data, split_indices, fit_scaler, transform_images, normalize_targets, inverse_targets
from stipsd.metrics import image_metrics, source_metrics

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.ids = load_data(ROOT / "examples/sample_data.npz")

    def test_packaged_data_and_metadata(self):
        self.assertEqual(self.x.shape, (64, 40, 40))
        self.assertEqual(self.y.shape, (64, 3))
        self.assertEqual(len(set(self.ids)), 64)
        metadata = json.loads((ROOT / "examples/data_metadata.json").read_text())
        self.assertEqual(metadata["source_type"], "simulated_concentration_fields")

    def test_split_disjoint_reproducible(self):
        split = split_indices(64)
        self.assertEqual([len(v) for v in split.values()], [44, 9, 11])
        self.assertEqual(len(set(np.concatenate(list(split.values())))), 64)
        for key in split:
            np.testing.assert_array_equal(split[key], split_indices(64)[key])

    def test_scaler_excludes_held_out_values(self):
        train = split_indices(64)["train"]
        expected = fit_scaler(self.x[train], self.y[train])
        mutated = self.x.copy()
        mutated[split_indices(64)["test"]] = 1e20
        self.assertEqual(expected, fit_scaler(mutated[train], self.y[train]))
        transformed = transform_images(mutated, expected)
        self.assertLessEqual(transformed.max(), 1)
        np.testing.assert_allclose(inverse_targets(normalize_targets(self.y, expected), expected), self.y, rtol=1e-5, atol=.01)

    def test_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.npz"
            x = self.x.copy()
            x[1] = x[0]
            np.savez(path, images=x, targets=self.y, sample_ids=self.ids)
            with self.assertRaisesRegex(ValueError, "Duplicate images"):
                load_data(path)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(7).uniform(0, 1, (40, 40)).astype(np.float32)

    def test_mask_direction_and_determinism(self):
        spatial = make_mask((40, 40), "spatial_strip", .3, np.random.default_rng(7))
        temporal = make_mask((40, 40), "temporal_truncation", .3, np.random.default_rng(7))
        np.testing.assert_array_equal(spatial[:, 0], spatial[:, -1])
        self.assertFalse(temporal[:, :28].any())
        self.assertTrue(temporal[:, 28:].all())
        for scenario in SCENARIOS:
            first = make_mask((40, 40), scenario, .3, np.random.default_rng(7))
            second = make_mask((40, 40), scenario, .3, np.random.default_rng(7))
            np.testing.assert_array_equal(first, second)
            self.assertLess(abs(first.mean() - .3), .04)

    def test_all_methods_preserve_observations_and_ignore_hidden_truth(self):
        for scenario in SCENARIOS:
            mask = make_mask((40, 40), scenario, .3, np.random.default_rng(1))
            alternative = self.image.copy()
            alternative[mask] = np.nan
            for method in METHODS:
                with self.subTest(scenario=scenario, method=method):
                    first = complete(self.image, mask, method)
                    second = complete(alternative, mask, method)
                    np.testing.assert_array_equal(first, second)
                    np.testing.assert_array_equal(first[~mask], self.image[~mask])
                    self.assertTrue(np.isfinite(first).all())
                    self.assertTrue((first >= 0).all() and (first <= 1).all())

    def test_no_missing_identity(self):
        mask = np.zeros((40, 40), dtype=bool)
        for method in METHODS:
            np.testing.assert_array_equal(complete(self.image, mask, method), self.image)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            complete(self.image, np.ones((40, 40), dtype=bool), "linear")
        with self.assertRaises(ValueError):
            complete(self.image, np.zeros((40, 40), dtype=np.uint8), "linear")
        with self.assertRaises(ValueError):
            complete(self.image, np.zeros((40, 40), dtype=bool), "STIPSD")
        with self.assertRaises(ValueError):
            make_mask((40, 40), "mixed", 1, np.random.default_rng(1))


class MetricTests(unittest.TestCase):
    def test_identity_ssim(self):
        image = np.random.default_rng(9).random((40, 40)).astype(np.float32)
        metrics = image_metrics(image, image, np.zeros_like(image, dtype=bool))
        self.assertEqual(metrics["completion_RMSE_all"], 0)
        self.assertAlmostEqual(metrics["completion_SSIM_all"], 1)

    def test_mape_range_definition_and_zero(self):
        values = source_metrics([2., 4., 10.], [3., 6., 15.], [10., 10., 10.])
        self.assertAlmostEqual(values["mean_APE_pct"], 50)
        self.assertAlmostEqual(values["mean_range_NAE_pct"], 80 / 3)
        zero = source_metrics([0., 4., 10.], [1., 6., 15.], [0., 10., 10.])
        self.assertTrue(np.isnan(zero["APE_T_pct"]))
        self.assertTrue(np.isnan(zero["range_NAE_T_pct"]))
        self.assertTrue(np.isnan(zero["mean_APE_pct"]))


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency not installed")
class OptionalTorchTests(unittest.TestCase):
    def test_resnet_output_shape(self):
        import torch
        from stipsd.resnet_model import create_model
        torch.set_num_threads(2)
        with torch.no_grad():
            self.assertEqual(tuple(create_model().eval()(torch.zeros(2, 1, 40, 40)).shape), (2, 3))

    def test_adr_derivatives_and_sign(self):
        import torch
        from stipsd.physics import adr_residual
        x = torch.arange(6.)[None, :, None].expand(1, 6, 6)
        t = torch.arange(6.)[None, None, :].expand(1, 6, 6)
        # C=x^2+3t: dC/dt=3, dC/dx=2x, d2C/dx2=2.
        actual = adr_residual(x**2 + 3*t, .4, .2, 0)
        expected = 3 + .8*x[:, 1:-1, 1:-1] - .4
        torch.testing.assert_close(actual, expected)

    def test_adr_projection(self):
        from stipsd.physics import correct
        image = np.random.default_rng(9).random((40, 40)).astype(np.float32)
        mask = make_mask((40, 40), "mixed", .3, np.random.default_rng(1))
        settings = {"u_pixel": .4, "d_pixel": .2, "k_pixel": .01, "iterations": 2, "learning_rate": .02}
        actual = correct(image, mask, settings)
        np.testing.assert_array_equal(actual[~mask], image[~mask])
        self.assertTrue(np.isfinite(actual).all())


class IntegrationTests(unittest.TestCase):
    def test_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "run"
            command = [sys.executable, str(ROOT / "run.py"), "--out", str(target)]
            result = subprocess.run(command, cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((target / "run_manifest.json").read_text())
            self.assertEqual(manifest["source_strategy"], "not_run")
            self.assertEqual(manifest["completion_rows"], 165)
            with (target / "completion_metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 165)
            repeat = subprocess.run(command, cwd=tmp, capture_output=True, text=True)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertIn("not empty", repeat.stderr)


if __name__ == "__main__":
    unittest.main()
