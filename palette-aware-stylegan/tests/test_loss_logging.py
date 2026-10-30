"""Standard-library tests for complete loss intervals and checkpoint continuation."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from training.loss_logging import LossLogger


def loss_values(step: int) -> dict[str, float]:
    regularised = (step - 1) % 16 == 0
    r1 = 8.0 if regularised else 0.0
    path = 5.0 if regularised else 0.0
    return {
        "d_total": 1.0 + r1,
        "g_total": 2.0 + 3.0 + path,
        "d_adv": 1.0,
        "g_adv": 2.0,
        "r1": r1,
        "palette_weighted": 3.0,
        "path_weighted": path,
    }


class LossLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def logger(self, interval: int = 2000, directory: Path | None = None) -> LossLogger:
        return LossLogger(
            directory or self.directory,
            log_interval=interval,
            effective_batch_size=16,
        )

    @staticmethod
    def rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_interval_includes_lazy_steps_omitted_by_endpoint_sampling(self) -> None:
        logger = self.logger()
        summary = None
        for step in range(1, 2001):
            summary = logger.record(step, loss_values(step))
        self.assertIsNotNone(summary)
        self.assertEqual(summary["interval_steps"], 2000)
        self.assertEqual(summary["interval_start_step"], 1)
        self.assertFalse(summary["partial_interval"])
        self.assertAlmostEqual(summary["mean_r1"], 125 * 8.0 / 2000)
        self.assertAlmostEqual(summary["mean_path_weighted"], 125 * 5.0 / 2000)
        self.assertAlmostEqual(summary["mean_d_total"], 1.5)
        self.assertAlmostEqual(summary["mean_g_total"], 5.3125)
        rows = self.rows(logger.step_path)
        self.assertEqual(len(rows), 2000)
        self.assertEqual(float(rows[-1]["r1"]), 0.0)
        self.assertEqual(float(rows[-1]["path_weighted"]), 0.0)
        self.assertEqual(float(rows[-1]["kimg"]), 32.0)
        self.assertEqual(len(self.rows(logger.summary_path)), 1)

    def test_loss_ema_uses_all_steps_and_initialises_from_first_observation(self) -> None:
        logger = LossLogger(
            self.directory, log_interval=20, effective_batch_size=16,
            ema_half_life_steps=1,
        )
        first = loss_values(1)
        first.update(d_total=2.0, g_total=6.0)
        logger.record(1, first)
        second = loss_values(2)
        second.update(d_total=4.0, g_total=10.0)
        logger.record(2, second)
        self.assertEqual(logger.loss_ema_d, 3.0)
        self.assertEqual(logger.loss_ema_g, 8.0)
        self.assertEqual(logger.ema_half_life_steps, 1.0)

    def test_partial_checkpoint_resume_matches_uninterrupted_statistics(self) -> None:
        uninterrupted = self.logger(4, self.directory / "uninterrupted")
        expected = None
        for step in range(1, 5):
            expected = uninterrupted.record(step, loss_values(step))
        before = self.logger(4, self.directory / "resumed")
        for step in range(1, 4):
            before.record(step, loss_values(step))
        state = before.state_dict()
        restored = self.logger(4, self.directory / "resumed")
        restored.load_state_dict(state, completed_step=3)
        actual = restored.record(4, loss_values(4))
        self.assertEqual(actual, expected)
        self.assertEqual(restored.state_dict(), uninterrupted.state_dict())
        self.assertNotEqual(before.step_path, restored.step_path)
        self.assertEqual([int(r["step"]) for r in self.rows(before.step_path)], [1, 2, 3])
        self.assertEqual([int(r["step"]) for r in self.rows(restored.step_path)], [4])

    def test_old_checkpoint_starts_history_at_next_observation(self) -> None:
        logger = self.logger(4)
        logger.reset(completed_step=6)
        logger.record(7, loss_values(7))
        self.assertEqual(logger.loss_ema_d, loss_values(7)["d_total"])
        summary = logger.record(8, loss_values(8))
        self.assertEqual(summary["loss_history_start_step"], 7)
        self.assertEqual(summary["interval_start_step"], 7)
        self.assertEqual(summary["interval_steps"], 2)
        self.assertTrue(summary["partial_interval"])
        self.assertEqual([int(r["step"]) for r in self.rows(logger.step_path)], [7, 8])

    def test_final_flush_writes_one_labelled_partial_interval(self) -> None:
        logger = self.logger(4)
        for step in (1, 2, 3):
            self.assertIsNone(logger.record(step, loss_values(step)))
        summary = logger.flush()
        self.assertTrue(summary["partial_interval"])
        self.assertEqual(summary["interval_steps"], 3)
        self.assertIsNone(logger.flush())
        self.assertEqual(len(self.rows(logger.summary_path)), 1)

    def test_resume_rejects_incompatible_or_misaligned_logging_state(self) -> None:
        logger = self.logger(4)
        logger.record(1, loss_values(1))
        state = logger.state_dict()
        with self.assertRaisesRegex(ValueError, "log_interval"):
            self.logger(8).load_state_dict(state, completed_step=1)
        with self.assertRaisesRegex(ValueError, "training step"):
            self.logger(4).load_state_dict(state, completed_step=2)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            logger.record(3, loss_values(3))

    def test_unstarted_logger_does_not_create_files(self) -> None:
        logger = self.logger()
        self.assertEqual(list(self.directory.iterdir()), [])
        self.assertIsNone(logger.flush())
        self.assertEqual(logger.state_dict()["interval_steps"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
