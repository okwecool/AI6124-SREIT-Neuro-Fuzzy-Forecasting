"""Safety checks for the frozen final entry point; never run final fits here."""

import unittest

from src.project_eval import ROOT
from src.tools.run_final_holdout import CONFIG, preflight, run


class FinalRunnerSafetyTests(unittest.TestCase):
    def test_freeze_matches_data_and_source_files(self):
        frozen, data = preflight(CONFIG)
        self.assertEqual((len(data.ids("final_holdout", "train")),
                          len(data.ids("final_holdout", "test"))), (931, 251))
        self.assertEqual(frozen["final_model_fits"], 12)

    def test_existing_output_prevents_second_evaluation(self):
        with self.assertRaises(FileExistsError):
            run(CONFIG, ROOT / "output/development",
                ROOT / "output/runs/development", "cpu")


if __name__ == "__main__":
    unittest.main()
