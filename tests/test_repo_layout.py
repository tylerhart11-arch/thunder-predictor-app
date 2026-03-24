from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from app.config import build_paths, load_config
from dashboard.helpers import confidence_band, format_probability, is_pregame_status


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RepoLayoutTests(unittest.TestCase):
    def test_config_points_to_processed_data_layout(self) -> None:
        config = load_config("config/config.yaml")
        paths = build_paths(config)

        self.assertEqual(paths.cleaned_dir.relative_to(PROJECT_ROOT), Path("data/processed/cleaned"))
        self.assertEqual(paths.features_dir.relative_to(PROJECT_ROOT), Path("data/processed/features"))
        self.assertEqual(paths.predictions_dir.relative_to(PROJECT_ROOT), Path("data/processed/predictions"))
        self.assertEqual(paths.artifacts_dir.relative_to(PROJECT_ROOT), Path("data/processed/artifacts"))
        self.assertEqual(paths.diagnostics_dir.relative_to(PROJECT_ROOT), Path("reports/diagnostics"))

    def test_expected_project_directories_exist(self) -> None:
        required_dirs = [
            "app",
            "config",
            "data/raw",
            "data/processed",
            "docs",
            "models",
            "notebooks",
            "pipelines",
            "prompts",
            "reports",
            "scripts",
            "tests",
        ]

        for rel_path in required_dirs:
            with self.subTest(rel_path=rel_path):
                self.assertTrue((PROJECT_ROOT / rel_path).is_dir())

    def test_dashboard_helper_behaviors(self) -> None:
        self.assertEqual(format_probability(0.6123), "61.2%")
        self.assertEqual(confidence_band(0.83), "Strong edge")
        self.assertEqual(confidence_band(0.68), "Lean")
        self.assertEqual(confidence_band(0.51), "Toss-up")

        scoreboard = pd.DataFrame(
            {
                "GAME_STATUS_ID": [1, 2],
                "GAME_STATUS_TEXT": ["7:00 PM ET", "Final"],
            }
        )

        mask = is_pregame_status(scoreboard)
        self.assertEqual(mask.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
