import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibrate import Journal
import m1


class RunHandoffTests(unittest.TestCase):
    def test_complete_measurements_replay_without_backend_connection(self):
        checked = {"freeze_sha256": "freeze", "measurement": {}}
        project_id = "fresh-v2-project"
        identity = {"freeze_sha256": "freeze", "project_sha256": hashlib.sha256(project_id.encode()).hexdigest(),
                    "model": m1.MODEL, "lora_seed": m1.LORA_SEED}
        result = {"status": "M1_measurements_complete_pending_publication", "m1_complete": False,
                  "main_run_authorized": False, "launch_packet": {"publication_freeze_commit": None}}
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "runs/m1/journal.jsonl")
            journal.call("m1/identity", identity, lambda: identity)
            journal.call("m1/measurement-closeout", {}, lambda: result)
            with patch("m1.preflight", return_value=checked), patch("m1.Backend.connect") as connect:
                self.assertEqual(m1.run(directory, project_id=project_id, keychain_service="explicit"), result)
                connect.assert_not_called()
            self.assertTrue((Path(directory) / "runs/m1/launch_packet.json").is_file())

    def test_full_driver_reaches_closeout_only_after_every_calibration_gate(self):
        checked = {"freeze_sha256": "freeze", "test_commit": "a" * 40,
                   "paid_launch_allowed": True, "measurement": {"noise_repeats": 3}}
        selected = {"T1": {"candidate": "arithmetic_derivations"}}
        calls = []

        class API:
            tokenizer = object()

            def origin(self):
                calls.append("origin")
                return "client"

            def save(self, client, name, step):
                return {"state_path": "state", "sampler_path": "sampler"}

            def download_sampler(self, path, destination):
                calls.append("download-origin")
                return {"directory": str(destination), "adapter": {"rank": 32, "alpha": 32}}

        def stage(name, result):
            def call(*args):
                calls.append(name)
                return result
            return call

        for failed in (None, "screen", "persistence", "probes", "diversity"):
            calls.clear()
            results = {
                "screen": {"status": "task_screening_complete", "selected": selected},
                "persistence": {"status": "persistence_manipulation_complete", "events": []},
                "probes": {"status": "probe_calibration_complete"},
                "diversity": {"status": "diversity_calibration_complete"}}
            if failed:
                results[failed] = {"status": "m1_failed", "failure": failed, "m1_complete": False}
            final = {"status": "M1_measurements_complete_pending_publication", "m1_complete": False,
                     "main_run_authorized": False}
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as directory, \
                 patch("m1.preflight", return_value=checked), patch("m1.Backend.connect", return_value=API()), \
                 patch("m1.measure.evaluate_if", return_value={"if_score": 50}), \
                 patch("m1.data.load_rows", return_value=[{}] + [{"prompt": "prompt"}] * 2000), \
                 patch("m1.data.render_repair_prompt", return_value=[1]), \
                 patch("m1.screen_tasks", side_effect=stage("screen", results["screen"])), \
                 patch("m1.persistence", side_effect=stage("persistence", results["persistence"])), \
                 patch("m1.state_panel", return_value=[]), \
                 patch("m1.calibrate_probes", side_effect=stage("probes", results["probes"])), \
                 patch("m1.calibrate_diversity", side_effect=stage("diversity", results["diversity"])), \
                 patch("closeout.finish_m1", side_effect=stage("closeout", final)) as finish:
                result = m1.run(directory, project_id="fresh-v2-project", keychain_service="explicit")
                self.assertEqual(result, results[failed] if failed else final)
                if failed:
                    finish.assert_not_called()
                    self.assertEqual(calls[-1], failed)
                else:
                    self.assertEqual(calls, ["origin", "download-origin", "screen", "persistence", "probes", "diversity", "closeout"])
                    finish.assert_called_once()
                self.assertFalse(result["m1_complete"])


if __name__ == "__main__":
    unittest.main()
