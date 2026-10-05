"""Check the trajectory report figures against the data they are drawn from."""

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_figures", ROOT / "scripts" / "build_trajectory_figures.py")
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)

EDITION = "2026-10-04"


class TrajectoryFigureTests(unittest.TestCase):
    def setUp(self):
        self.traces = figures.read_traces()
        self.history = json.loads(
            figures.DEFAULT_HISTORY.read_text(encoding="utf-8"))

    def test_committed_figures_match_the_data(self):
        """The drawn figures still reproduce from the archive they cite.

        This is the guard that matters: if a trace is corrected or the drawing
        changes, the committed SVGs go stale silently, and the report shows
        numbers that no longer match its own tables.
        """
        folder = figures.FIGURES / EDITION
        for name, (w, h, body, label) in figures.build(self.traces, self.history).items():
            with self.subTest(figure=name):
                path = folder / f"{name}.svg"
                self.assertTrue(path.is_file(), path)
                self.assertEqual(path.read_text(encoding="utf-8"),
                                 figures.wrap(w, h, body, label))

    def test_every_charted_run_is_in_the_archive(self):
        missing = [run for run in figures.RUN_ORDER if run not in self.traces]
        self.assertEqual(missing, [])

    def test_accuracy_maps_to_the_drawn_scale(self):
        """A value at the top of the axis sits on the top edge of the plot.

        The charts are hand-placed, so a wrong scale would draw a plausible but
        false picture. This pins the mapping at both ends.
        """
        _, _, body, _ = figures.trace_panel(
            self.traces, "E08", "scale check", 10, ymax=50, w=430, h=300)
        top, bottom = 40, 300 - 44  # plot area, as trace_panel lays it out
        self.assertIn('y1="%.1f"' % bottom, body)  # the 0% gridline
        self.assertIn('y1="%.1f"' % top, body)     # the 50% gridline
        # E08 starts at 26.0%, which must land between the two.
        start = top + (bottom - top) * (1 - 26.0 / 50)
        self.assertIn("%.1f" % start, body)


if __name__ == "__main__":
    unittest.main()
