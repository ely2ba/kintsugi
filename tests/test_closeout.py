from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from calibrate import Journal
import closeout


def checkpoint(name):
    return {"state_path": name + "-state", "sampler_path": name + "-sampler"}


def event(name, slot, cycle, arm, heldout, start):
    return {"branch": name, "slot": slot, "cycle": cycle, "arm": arm,
            "start_checkpoint": start, "A": checkpoint(name + "-A"), "B": checkpoint(name + "-B"),
            "learning": {"decision": {"checkpoint": {"heldout": heldout}}}}


def selection(slot, baseline, realizations=()):
    return {"candidate": slot, "selected": {"realizations": list(realizations)},
            "reference": {"references": {"heldout0": baseline}},
            "thresholds": {"minimum_movement": 0.1, "gate_competence": 0.6}}


class ContextTests(unittest.TestCase):
    def test_screening_is_disposable_and_persistence_histories_branch_at_common_b1(self):
        origin = checkpoint("origin")
        screens = [event("screen1", "T1", 1, "fixed", 0.6, origin),
                   event("screen2", "T1", 1, "fixed", 0.7, origin)]
        common = event("common", "T3", 1, "fixed", 0.7, origin)
        fixed2 = event("fixed2", "T2", 2, "fixed", 0.8, common["B"])
        fixed3 = event("fixed3", "T6", 3, "fixed", 0.9, fixed2["B"])
        rolling2 = event("rolling2", "T2", 2, "rolling", 0.85, common["B"])
        rolling3 = event("rolling3", "T6", 3, "rolling", 0.95, rolling2["B"])
        stage = {"screening": {"selected": {"T1": selection("T1", 0.1, screens)}},
                 "persistence": {"events": [common, fixed2, fixed3, rolling2, rolling3]}}
        histories = [row["acquired"] for row in closeout.contexts(stage)]
        self.assertEqual(histories, [{"T1": 0.6}, {"T1": 0.7}, {"T3": 0.7},
                                    {"T3": 0.7, "T2": 0.8}, {"T3": 0.7, "T2": 0.8, "T6": 0.9},
                                    {"T3": 0.7, "T2": 0.85}, {"T3": 0.7, "T2": 0.85, "T6": 0.95}])


class FakeBackend:
    def __init__(self):
        self.tokenizer = object()
        self.samples, self.downloads = [], []

    def sampler(self, path):
        return path

    def sample(self, sampler, prompts, **kwargs):
        self.samples.append((sampler, prompts, kwargs))
        return {"groups": [[{"tokens": [1], "logprobs": [-1.0], "text": "answer"}]], "accounting": {}}

    def download_sampler(self, path, destination):
        self.downloads.append(path)
        return {"directory": "/virtual/" + path, "accounting": {}}


class PhysicalMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeBackend()
        self.origin = {**checkpoint("origin"), "download": {"directory": "/virtual/origin"}}
        self.panel_manifest = {"path": "data/diversity/panel.jsonl", "sha256": "panel",
                               "sampling": {"samples": 8, "max_tokens": 512, "temperature": 1.0, "seed": 7}}
        self.geometry, self.metrics, self.kl, self.protected = [], [], [], []
        self.values = {}
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

        def read(path):
            if "diversity" in str(path):
                return self.panel_manifest
            slot = Path(path).stem
            return {"candidate": slot, "metric": "verifier_success",
                    "splits": {"heldout": {"path": f"data/{slot}/heldout.jsonl"}}}

        def rows(path):
            return [{"header": True}, {"prompt": "repair"}] if str(path).endswith("repair_pool.jsonl") else [{"prompt_tokens": [2]}]

        def geometry(current, *, previous_path):
            self.geometry.append((str(current), str(previous_path)))
            return [{"frobenius": 1.0, "effective_rank": 1.0, "stable_rank": 1.0, "cosine_to_previous": 0.5}]

        def task(api, client, sampler, examples, manifest, seed):
            self.metrics.append((sampler, manifest["candidate"]))
            return {"q": self.values[(sampler, manifest["candidate"])], "valid": True, "accounting": {}}

        def kl(api, sampler, prompts, groups):
            self.kl.append(sampler)
            return {"kl_to_cycle0": 0.1, "accounting": {}}

        def protected(api, sampler, split, seed):
            self.protected.append((sampler, split))
            return {"if_score": 25, "total": 30, "accounting": {}}

        for name, replacement in (("closeout._read", read), ("closeout.load_rows", rows),
                                  ("data.render_repair_prompt", lambda tokenizer, prompt: [1]),
                                  ("closeout.adapter_geometry", geometry), ("closeout.evaluate_task", task),
                                  ("closeout.forward_kl", kl), ("closeout.evaluate_if", protected),
                                  ("closeout.diversity_summary", lambda examples, groups: {"coverage_gap": 0.2})):
            patcher = patch(name, side_effect=replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def measure(self, stage):
        return closeout.measure_m1_checkpoints(
            self.api, self.root, Journal(self.root / "journal.jsonl"), stage, {"candidate": "graph_coloring"},
            {"kl": {"prompt_count": 1, "samples": 1, "max_tokens": 512, "temperature": 1.0, "seed": 1}})

    def test_noop_alias_and_durable_replay_measure_one_physical_checkpoint(self):
        only = event("only", "T1", 1, "fixed", 0.7, self.origin)
        only["B"] = {**only["A"], "alias_of": "A"}
        stage = {"origin": self.origin, "screening": {"selected": {"T1": selection("T1", 0.2, [only])}},
                 "persistence": {"events": []}}
        first = self.measure(stage)
        self.assertEqual(len(first["physical_checkpoints"]), 1)
        self.assertEqual(len(first["retention"]), 2)
        self.assertEqual(first["retention"][0]["physical_key"], first["retention"][1]["physical_key"])
        self.assertEqual([row["current"] for row in first["retention"]], [1.0, 1.0])
        self.assertEqual(self.metrics, [])  # Current-task A heldout is reused for its physical B alias.
        self.assertEqual(self.api.downloads, [only["A"]["sampler_path"]])
        self.assertEqual(len(self.geometry), 1)
        self.assertEqual(len(self.kl), 1)
        self.assertEqual(len(self.protected), 2)  # Origin plus the one A/B physical state.
        counts = (len(self.api.samples), len(self.api.downloads), len(self.geometry), len(self.kl), len(self.protected))
        self.assertEqual(self.measure(stage), first)
        self.assertEqual(counts, (len(self.api.samples), len(self.api.downloads), len(self.geometry), len(self.kl), len(self.protected)))

    def test_geometry_uses_actual_start_then_a_and_retention_keeps_acquisition_denominators(self):
        common = event("common", "T3", 1, "fixed", 0.7, self.origin)
        second = event("second", "T2", 2, "fixed", 0.8, common["B"])
        stage = {"origin": self.origin,
                 "screening": {"selected": {"T3": selection("T3", 0.2), "T2": selection("T2", 0.1)}},
                 "persistence": {"events": [common, second]}}
        self.values = {(common["B"]["sampler_path"], "T3"): 0.6,
                       (second["A"]["sampler_path"], "T3"): 0.55,
                       (second["B"]["sampler_path"], "T3"): 0.4,
                       (second["B"]["sampler_path"], "T2"): 0.65}
        result = self.measure(stage)
        suffix = "/adapter_model.safetensors"
        self.assertEqual(self.geometry, [("/virtual/common-A-sampler" + suffix, "/virtual/origin" + suffix),
                                         ("/virtual/common-B-sampler" + suffix, "/virtual/common-A-sampler" + suffix),
                                         ("/virtual/second-A-sampler" + suffix, "/virtual/common-B-sampler" + suffix),
                                         ("/virtual/second-B-sampler" + suffix, "/virtual/second-A-sampler" + suffix)])
        a, b = result["retention"][-2:]
        self.assertAlmostEqual(a["tasks"]["T3"]["denominator"], 0.5)
        self.assertAlmostEqual(a["tasks"]["T2"]["denominator"], 0.7)
        self.assertAlmostEqual(a["prior_mean"], 0.7)
        self.assertEqual(a["current"], 1.0)
        self.assertAlmostEqual(b["prior_mean"], 0.4)
        self.assertAlmostEqual(b["current"], 0.55 / 0.7)
        self.assertEqual(a["prior_coverage"], {"defined": 1, "total": 1, "fraction": 1.0})
        self.assertNotIn((second["A"]["sampler_path"], "T2"), self.metrics)

    def test_missing_start_checkpoint_is_not_silently_replaced_with_origin(self):
        only = event("only", "T1", 1, "fixed", 0.7, self.origin)
        del only["start_checkpoint"]
        stage = {"origin": self.origin, "screening": {"selected": {"T1": selection("T1", 0.2, [only])}},
                 "persistence": {"events": []}}
        with self.assertRaises(KeyError):
            self.measure(stage)


class CostUnitTests(unittest.TestCase):
    def test_only_selected_recipe_prefixes_and_first_competence_dose_are_charged(self):
        completed, events = {}, []
        amounts = {"learn/start": 10, "learn/update/001": 11, "learn/evaluate/001": 12,
                   "learn/update/002": 21, "learn/evaluate/002": 22, "learn/update/003": 31,
                   "learn/evaluate/003": 32, "learn/heldout/003": 5,
                   "repair/update/001": 7, "repair/criterion/005": 8}
        for realization in (1, 2):
            branch = f"screen/task/1e-05/{realization}"
            events.append({"branch": branch, "learning": {"start": {"gate": 0.2},
                           "points": [{"step": 1, "gate": 0.3}, {"step": 2, "gate": 0.7}, {"step": 3, "gate": 0.8}]}})
            for suffix, amount in amounts.items():
                completed[branch + "/" + suffix] = {"result": {"accounting": {"train_tokens": realization * amount,
                                                                                         "gradient_target_tokens": 999}}}
            completed[branch + "/learn/complete"] = {"result": {"nested": {"accounting": {"train_tokens": 99999}}}}
        completed["screen/task/1e-05/10/learn/start"] = {"result": {"accounting": {"train_tokens": 99999}}}
        completed["screen/task/3e-05/1/learn/start"] = {"result": {"accounting": {"train_tokens": 99999}}}
        stage = {"screening": {"selected": {"T1": selection("T1", 0.2, events)}}}
        journal = SimpleNamespace(completed=completed)
        prices = {"train": 1_000_000, "prefill": 0, "cached": 0, "sample": 0}
        result = closeout.measured_task_units(journal, stage, prices)
        self.assertEqual(result, {"T1": {"learn_to_competence": 121.5, "learn_to_damage": 216.0,
                                         "repair": 22.5, "native_heldout": 7.5}})


if __name__ == "__main__":
    unittest.main()
