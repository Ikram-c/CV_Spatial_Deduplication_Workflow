"""Zero-inference regression tests for fractional geometry and transitivity."""
import unittest

from deduplication import (
    intersection_area, overlapping_annotation_pairs, deduplicated_counts,
)
from examples.run_demo import run


class DeduplicationTests(unittest.TestCase):
    def test_fractional_intersection_is_not_rounded_away(self):
        a = (0.49, 0.0, 1.01, 1.0, 1, 101)
        b = (1.00, 0.0, 1.50, 1.0, 1, 102)
        self.assertAlmostEqual(intersection_area(a, b), 0.01)
        self.assertEqual(overlapping_annotation_pairs([a], [b]), [(101, 102)])

    def test_touching_negative_and_disjoint_rectangles(self):
        self.assertEqual(intersection_area((-2, -2, -1, -1), (-1, -2, 0, -1)), 0)
        self.assertEqual(intersection_area((0, 0, 2, 2), (3, 3, 4, 4)), 0)

    def test_transitivity_and_category_isolation(self):
        categories = {101: 1, 102: 1, 103: 1, 104: 1, 201: 2}
        edges = [(101, 102), (102, 103), (101, 201), (104, 999)]
        self.assertEqual(deduplicated_counts(categories, edges), {1: 2, 2: 1})

    def test_different_categories_do_not_create_overlap_edges(self):
        a = (0.0, 0.0, 2.0, 2.0, 1, 101)
        b = (0.0, 0.0, 2.0, 2.0, 2, 201)
        self.assertEqual(overlapping_annotation_pairs([a], [b]), [])

    def test_demo_counts_chain_once(self):
        import json
        from pathlib import Path
        data = json.loads((Path(__file__).resolve().parents[1] / 'examples/overlapping_frames.json').read_text())
        result = run(data)
        self.assertEqual(result['raw_counts'], {'1': 4, '2': 1})
        self.assertEqual(result['unique_counts'], {'1': 2, '2': 1})
        self.assertEqual(result['overlap_edges'], [[101, 102], [102, 103]])
        self.assertEqual(result['duplicates_removed'], 2)


if __name__ == '__main__':
    unittest.main()
