import unittest

from tools.analyze_ude_rationality import (
    classify_rationality_level,
    compute_weighted_score,
    parse_rigorous_metrics,
    score_from_rigorous_metrics,
)


class TestUdeRationalityDataDriven(unittest.TestCase):
    def test_weighted_score(self):
        scores = {
            "uncertainty_calibration": 70,
            "ordinal_metric_consistency": 60,
            "affine_stability": 50,
            "joint_depth_structure": 40,
        }
        weights = {
            "uncertainty_calibration": 0.35,
            "ordinal_metric_consistency": 0.30,
            "affine_stability": 0.20,
            "joint_depth_structure": 0.15,
        }
        self.assertAlmostEqual(compute_weighted_score(scores, weights), 58.5, places=3)

    def test_level_classification(self):
        self.assertEqual(classify_rationality_level(85), "合理")
        self.assertEqual(classify_rationality_level(70), "部分合理")
        self.assertEqual(classify_rationality_level(50), "不合理")

    def test_parse_rigorous_metrics(self):
        sample = """
RIGOROUS DEPTH BRANCH DIAGNOSTIC RESULTS

ordinal_accuracy:
  mean: 3.2500 | std: 0.8000 | range: [1.2000, 5.1000]

uncertainty_error_correlation:
  mean: 0.2800 | std: 0.0500 | range: [0.1900, 0.3500]
"""
        metrics = parse_rigorous_metrics(sample)
        self.assertAlmostEqual(metrics["ordinal_accuracy"], 3.25, places=3)
        self.assertAlmostEqual(metrics["uncertainty_error_correlation"], 0.28, places=3)

    def test_scoring_pipeline(self):
        metrics = {
            "uncertainty_error_correlation": 0.25,
            "error_high_uncertainty": 0.06,
            "error_low_uncertainty": 0.045,
            "ordinal_accuracy": 0.85,
            "metric_bone_depth_error_mm": 0.03,
            "affine_r2_mean": 0.65,
            "alpha_mean": 1.08,
            "alpha_std": 0.18,
            "beta_mean": 0.02,
            "beta_std": 0.03,
            "grad_mag_mean": 0.012,
            "grad_mag_std": 0.01,
            "joint_depth_variance": 0.008,
            "depth_at_joint_mean": 0.62,
        }
        result = score_from_rigorous_metrics(metrics)
        self.assertIn("final_score", result)
        self.assertIn("level", result)
        self.assertGreaterEqual(result["final_score"], 0.0)
        self.assertLessEqual(result["final_score"], 100.0)


if __name__ == "__main__":
    unittest.main()
