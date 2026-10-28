"""Check the sweep job's actual launch command without a GPU or cluster.

Run: python3 -m unittest discover -s tests -p 'test_alpha_sweep_results.py' -v
CSV scoring and recovery are covered in test_com_sweep.py.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]


class AlphaSweepJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workdir = Path(self.temp.name)
        (self.workdir / "logs").mkdir()
        self.capture = self.workdir / "launches.jsonl"
        script = (REPOSITORY / "run_alpha_sweep.sbatch").read_text()
        # Execute the real job body after its cluster/environment preflight.
        self.launch = "set -euo pipefail\n" + script.split(
            "# One CSV per job, appended immediately after each successful evaluation.", 1)[1]
        (self.workdir / "com_sweep.py").write_text(textwrap.dedent('''\
            import json
            import os
            import sys
            with open(os.environ["SWEEP_CAPTURE"], "a") as output:
                output.write(json.dumps(sys.argv[1:]) + "\\n")
            if os.environ.get("FAIL_SWEEP"):
                sys.exit(9)
        '''))

    def run_job(self, job_id="123", **overrides):
        return subprocess.run(
            ["bash"], input=self.launch, cwd=self.workdir,
            env=dict(os.environ, KVCOMM_PYTHON=sys.executable, SLURM_JOB_ID=job_id,
                     SWEEP_CAPTURE=str(self.capture), **overrides),
            text=True, capture_output=True,
        )

    def calls(self):
        return [json.loads(line) for line in self.capture.read_text().splitlines()]

    def argument(self, args, flag):
        index = args.index(flag) + 1
        end = next((i for i in range(index, len(args)) if args[i].startswith("--")), len(args))
        return args[index:end]

    def test_launches_one_runner_with_requested_77_cases(self):
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 1)
        args = self.calls()[0]
        tasks = self.argument(args, "--test_tasks")
        alphas = self.argument(args, "--alphas")
        self.assertEqual(tasks, ["countries", "tipsheets", "hotpotqa", "qasper",
                                 "musique", "multifieldqa_en", "twowikimqa"])
        self.assertEqual(alphas, ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5",
                                  "0.4", "0.3", "0.2", "0.1", "0.0"])
        self.assertEqual(len(tasks) * len(alphas), 77)
        expected = {"--model_A": "Qwen/Qwen3-4B", "--model_B": "Qwen/Qwen3-4B",
                    "--device": "cuda:0", "--top_layers": "0.7", "--calib_size": "1",
                    "--limit": "2", "--seed": "42", "--max_input_length": "8192"}
        for flag, value in expected.items():
            self.assertEqual(self.argument(args, flag), [value])
        self.assertNotIn("--do_test", args)

    def test_output_paths_identify_the_cluster_job(self):
        result = self.run_job(job_id="456")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.calls()[0]
        self.assertEqual(self.argument(args, "--results_csv"), ["logs/alpha-sweep-456-results.csv"])
        self.assertEqual(self.argument(args, "--run_name"), ["alpha-sweep-456"])

    def test_runner_failure_propagates_to_job(self):
        result = self.run_job(FAIL_SWEEP="1")
        self.assertEqual(result.returncode, 9, result.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertNotIn("Completed", result.stdout)


if __name__ == "__main__":
    unittest.main()
