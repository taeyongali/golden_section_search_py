import math
import unittest

from golden_section_search_py import (
    GoldenSectionSearchError,
    GoldenSectionSearchResult,
    golden_section_search,
)


class TestGoldenSectionSearch(unittest.TestCase):
    def test_quadratic_minimum(self):
        # f(x) = (x - 3)^2 + 1 has its minimum at x = 3, f = 1.
        result = golden_section_search(lambda x: (x - 3.0) ** 2 + 1.0, -10.0, 10.0)
        self.assertIsInstance(result, GoldenSectionSearchResult)
        self.assertLess(abs(result.x - 3.0), 1e-4)
        self.assertLess(abs(result.fx - 1.0), 1e-6)
        self.assertGreater(result.iterations, 0)
        self.assertGreater(result.evaluations, 0)

    def test_returns_dataclass_fields(self):
        result = golden_section_search(lambda x: x * x, -1.0, 1.0)
        self.assertTrue(hasattr(result, "x"))
        self.assertTrue(hasattr(result, "fx"))
        self.assertTrue(hasattr(result, "iterations"))
        self.assertTrue(hasattr(result, "evaluations"))

    def test_tight_tolerance_improves_accuracy(self):
        loose = golden_section_search(lambda x: (x - 2.0) ** 2, -5.0, 5.0, tol=1e-2)
        tight = golden_section_search(lambda x: (x - 2.0) ** 2, -5.0, 5.0, tol=1e-10)
        self.assertLess(abs(tight.x - 2.0), abs(loose.x - 2.0) + 1e-12)
        self.assertGreaterEqual(tight.iterations, loose.iterations)

    def test_monotone_function_returns_endpoint(self):
        # A strictly increasing function has its minimum at the left end.
        result = golden_section_search(lambda x: x, 0.0, 1.0)
        self.assertLess(abs(result.x - 0.0), 1e-3)
        self.assertLess(abs(result.fx - 0.0), 1e-3)

    def test_monotone_decreasing_function_returns_endpoint(self):
        result = golden_section_search(lambda x: -x, 0.0, 1.0)
        self.assertLess(abs(result.x - 1.0), 1e-3)
        self.assertLess(abs(result.fx + 1.0), 1e-3)

    def test_flat_function(self):
        # Every point is a minimum; the algorithm should still terminate
        # and return a finite value inside the interval.
        result = golden_section_search(lambda x: 0.0, -1.0, 1.0)
        self.assertTrue(math.isfinite(result.x))
        self.assertGreaterEqual(result.x, -1.0)
        self.assertLessEqual(result.x, 1.0)
        self.assertEqual(result.fx, 0.0)

    def test_max_iter_caps_iterations(self):
        result = golden_section_search(
            lambda x: (x - 1.0) ** 2, -1.0, 1.0, tol=1e-15, max_iter=3
        )
        self.assertEqual(result.iterations, 3)
        self.assertLessEqual(result.evaluations, 5)

    def test_invalid_bounds_raise(self):
        with self.assertRaises(GoldenSectionSearchError):
            golden_section_search(lambda x: x * x, 1.0, 0.0)

    def test_equal_bounds_raise(self):
        with self.assertRaises(GoldenSectionSearchError):
            golden_section_search(lambda x: x * x, 1.0, 1.0)

    def test_non_positive_tolerance_raises(self):
        with self.assertRaises(GoldenSectionSearchError):
            golden_section_search(lambda x: x * x, 0.0, 1.0, tol=0.0)
        with self.assertRaises(GoldenSectionSearchError):
            golden_section_search(lambda x: x * x, 0.0, 1.0, tol=-1e-6)

    def test_invalid_max_iter_raises(self):
        with self.assertRaises(GoldenSectionSearchError):
            golden_section_search(lambda x: x * x, 0.0, 1.0, max_iter=0)

    def test_evaluation_count_matches_calls(self):
        calls = []

        def f(x: float) -> float:
            calls.append(x)
            return (x - 0.5) ** 2

        result = golden_section_search(f, 0.0, 1.0, tol=1e-6)
        self.assertEqual(result.evaluations, len(calls))

    def test_result_is_frozen(self):
        result = golden_section_search(lambda x: x * x, -1.0, 1.0)
        with self.assertRaises(Exception):
            result.x = 0.0  # type: ignore[misc]

    def test_asymmetric_interval(self):
        result = golden_section_search(lambda x: (x - 9.5) ** 2, 9.0, 10.0)
        self.assertLess(abs(result.x - 9.5), 1e-4)


if __name__ == "__main__":
    unittest.main()
