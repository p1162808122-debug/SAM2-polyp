import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.find_max_batch_size import even_batch_sizes, select_largest_stable


class BatchSizeProbePlanningTest(unittest.TestCase):
    def test_candidates_are_even_and_inclusive(self):
        self.assertEqual(even_batch_sizes(2, 8), (2, 4, 6, 8))

    def test_candidate_range_rejects_odd_or_reversed_bounds(self):
        with self.assertRaises(ValueError):
            even_batch_sizes(1, 8)
        with self.assertRaises(ValueError):
            even_batch_sizes(2, 7)
        with self.assertRaises(ValueError):
            even_batch_sizes(8, 2)

    def test_selection_uses_largest_success_with_at_least_one_gb_headroom(self):
        results = [
            {"batch_size": 2, "status": "ok", "estimated_headroom_mb": 6200},
            {"batch_size": 4, "status": "ok", "estimated_headroom_mb": 2300},
            {"batch_size": 6, "status": "ok", "estimated_headroom_mb": 800},
        ]
        self.assertEqual(select_largest_stable(results, 1024), 4)

    def test_selection_keeps_last_success_when_next_candidate_ooms(self):
        results = [
            {"batch_size": 2, "status": "ok", "estimated_headroom_mb": 5200},
            {"batch_size": 4, "status": "ok", "estimated_headroom_mb": 3500},
            {"batch_size": 6, "status": "oom", "estimated_headroom_mb": 0},
        ]
        self.assertEqual(select_largest_stable(results, 1024), 4)

    def test_selection_returns_none_when_batch_two_fails(self):
        results = [
            {"batch_size": 2, "status": "oom", "estimated_headroom_mb": 0},
        ]
        self.assertIsNone(select_largest_stable(results, 1024))


if __name__ == "__main__":
    unittest.main()
