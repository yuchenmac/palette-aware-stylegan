"""Tests for loading measured loss histories without merging overlapping runs."""

import csv
import tempfile
import unittest
from pathlib import Path

from plot_training import read_loss_records


class LossPlotInputTests(unittest.TestCase):
    def test_summary_and_step_records_are_identified_and_sorted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "steps.csv"
            fields = ["step", "kimg", "loss_history_start_step", "loss_ema_half_life_steps",
                      "d_total", "g_total", "loss_ema_d", "loss_ema_g"]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for step in (2, 1):
                    writer.writerow(dict(zip(fields, (step, step * .016, 1, 2000, 1.2, 1.3, 1.2, 1.3))))
            records, kind = read_loss_records([path])
            self.assertEqual(kind, "steps")
            self.assertEqual([r["step"] for r in records], [1, 2])
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                read_loss_records([path, path])


if __name__ == "__main__":
    unittest.main()
