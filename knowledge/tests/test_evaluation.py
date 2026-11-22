from django.test import SimpleTestCase

from knowledge.evaluation import suggest_threshold


class SuggestThresholdTests(SimpleTestCase):
    def test_separable_picks_midpoint(self):
        self.assertEqual(suggest_threshold([0.3, 0.4], [0.6, 0.7]), (0.5, True))

    def test_overlap_is_recall_first(self):
        self.assertEqual(suggest_threshold([0.3, 0.5], [0.45, 0.7]), (0.52, False))

    def test_no_hits_gives_no_suggestion(self):
        self.assertEqual(suggest_threshold([], [0.6]), (None, False))