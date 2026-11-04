"""Safety checks for the frozen final entry point; never run final fits here."""

import unittest

from src.project_eval import ROOT
from src.tools.run_final_holdout import CONFIG, preflight, run
from src.tools.verify_final_outputs import validate_completion_record


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

    def test_both_final_completion_paths_require_export_hashes(self):
        hashes = {name: "a" * 64 for name in (
            "prediction_file_sha256", "cases_file_sha256", "rules_file_sha256")}
        for status in ("final_execution_complete",
                       "final_analysis_complete_from_original_predictions"):
            validate_completion_record({"status": status, **hashes})
            with self.assertRaises(ValueError):
                validate_completion_record({"status": status})


if __name__ == "__main__":
    unittest.main()
