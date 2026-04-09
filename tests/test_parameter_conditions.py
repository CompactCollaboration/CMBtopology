"""Tests for Euclidean topology parameter condition validation."""

import unittest

from topology.src.tools import validate_euclidean_parameter_conditions


class TestEuclideanParameterConditions(unittest.TestCase):
    def test_valid_reference_case(self):
        result = validate_euclidean_parameter_conditions(
            L1=1.0,
            L2=1.0,
            L3=1.0,
            alpha=90.0,
            beta=90.0,
            gamma=0.0,
            angles_in_degrees=True,
            on_fail="none",
        )
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["failed_conditions"], [])

    def test_invalid_alpha_range(self):
        result = validate_euclidean_parameter_conditions(
            L1=1.0,
            L2=1.0,
            L3=1.0,
            alpha=0.0,
            beta=90.0,
            gamma=0.0,
            angles_in_degrees=True,
            on_fail="none",
        )
        self.assertFalse(result["is_valid"])
        self.assertTrue(any("condition_1" in item for item in result["failed_conditions"]))

    def test_invalid_cosine_bound_condition_3(self):
        result = validate_euclidean_parameter_conditions(
            L1=1.0,
            L2=1.0,
            L3=1.0,
            alpha=10.0,
            beta=90.0,
            gamma=0.0,
            angles_in_degrees=True,
            on_fail="none",
        )
        self.assertFalse(result["is_valid"])
        self.assertTrue(any("condition_3" in item for item in result["failed_conditions"]))

    def test_error_mode_raises(self):
        with self.assertRaises(ValueError):
            validate_euclidean_parameter_conditions(
                L1=1.0,
                L2=1.0,
                L3=1.0,
                alpha=0.0,
                beta=90.0,
                gamma=0.0,
                angles_in_degrees=True,
                on_fail="error",
            )


if __name__ == "__main__":
    unittest.main()
