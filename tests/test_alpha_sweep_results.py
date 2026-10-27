"""Check sweep CSV output without models or a GPU.

Run: python3 -m unittest discover -s tests -p 'test_alpha_sweep_results.py' -v
"""

import csv
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]


class AlphaSweepResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workdir = Path(self.temp.name)
        self.results_csv = self.workdir / "results.csv"
        script = (REPOSITORY / "run_alpha_sweep.sbatch").read_text()
        # Exercise the actual per-run wrapper, replacing only the GPU workload.
        self.wrapper = re.findall(r"<<'PY'\n(.*?)\nPY", script, re.DOTALL)[-1]
        (self.workdir / "torch.py").write_text(
            "from types import SimpleNamespace\n"
            "cuda = SimpleNamespace(is_initialized=lambda: False)\n"
        )
        (self.workdir / "com.py").write_text(textwrap.dedent('''\
            import argparse
            import os

            def parse_args():
                parser = argparse.ArgumentParser()
                parser.add_argument("--test_task")
                parser.add_argument("--alpha", type=float)
                return parser.parse_args()

            def main(cfg):
                print("communication result: 1.0000, communication time: 3.48s")
                if os.environ.get("FAIL_EXPERIMENT"):
                    raise RuntimeError("experiment failed")
                # Log rounding must not lose precision in the saved result.
                print("communication result: 0.4321, communication time: 0.46s")
                return float(os.environ.get("FINAL_SCORE", "0.432123456789"))

            if __name__ == "__main__":
                main(parse_args())
        '''))

    def run_experiment(self, task="tipsheets", alpha="0.7", **overrides):
        return subprocess.run(
            [sys.executable, "-", "--test_task", task, "--alpha", alpha],
            input=self.wrapper,
            cwd=self.workdir,
            env=dict(os.environ, KVCOMM_RESULTS_CSV=str(self.results_csv), **overrides),
            text=True,
            capture_output=True,
        )

    def read_rows(self):
        with self.results_csv.open(newline="") as source:
            reader = csv.DictReader(source)
            self.assertEqual(reader.fieldnames,
                             ["test_task", "alpha", "communication_result"])
            return list(reader)

    def test_saves_final_score_instead_of_calibration_or_rounded_log(self):
        result = self.run_experiment()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.results_csv.exists(), "completed score was not saved")
        self.assertEqual(self.read_rows(), [
            {"test_task": "tipsheets", "alpha": "0.7",
             "communication_result": "0.432123456789"},
        ])

    def test_appends_across_processes_without_losing_rows_or_repeating_header(self):
        first = self.run_experiment()
        self.assertEqual(first.returncode, 0, first.stderr)
        second = self.run_experiment("countries", "0.0", FINAL_SCORE="0.0")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertTrue(self.results_csv.exists(), "completed scores were not saved")
        self.assertEqual(self.read_rows(), [
            {"test_task": "tipsheets", "alpha": "0.7",
             "communication_result": "0.432123456789"},
            {"test_task": "countries", "alpha": "0.0",
             "communication_result": "0.0"},
        ])

    def test_failed_experiment_does_not_append_a_calibration_score(self):
        first = self.run_experiment()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertTrue(self.results_csv.exists(), "completed score was not saved")
        original = self.results_csv.read_bytes()
        failed = self.run_experiment("qasper", "0.6", FAIL_EXPERIMENT="1")
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("experiment failed", failed.stderr)
        self.assertEqual(self.results_csv.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
