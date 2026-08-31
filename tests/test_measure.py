import json
import math
import unittest

from backend import accounting, token_row
import if_suite
import measure
import probes
import tasks


def output(text, *, logprob=-2.0, length=2, truncated=False):
    return {"text": text, "tokens": [1] * length, "logprobs": [logprob] * length,
            "truncated": truncated, "stop_reason": "length" if truncated else "stop"}


class BackendDouble:
    def __init__(self):
        self.batches, self.sample_calls, self.score_calls, self.sampler_paths = [], [], [], []
        self.groups = []
        self.scored = []
        self.invalid_batch = None

    def evaluate_nll(self, client, rows):
        self.batches.append(rows)
        counts = [sum(token_row(row)[1]) for row in rows]
        total = sum(len(token_row(row)[0]) - 1 for row in rows)
        if len(self.batches) == self.invalid_batch:
            return {"valid": False, "nll": None, "target_tokens": sum(counts),
                    "failure": "nonfinite", "accounting": accounting(forward_tokens=total)}
        nll = sum(row["test_nll"] * count for row, count in zip(rows, counts)) / sum(counts)
        return {"valid": True, "nll": nll, "target_tokens": sum(counts),
                "accounting": accounting(forward_tokens=total)}

    def sampler(self, path):
        self.sampler_paths.append(path)
        return "sampler"

    def render_prompt(self, prompt):
        return [ord(value) for value in prompt]

    def sample(self, sampler, prompts, **kwargs):
        self.sample_calls.append((sampler, prompts, kwargs))
        return {"groups": self.groups, "accounting": accounting(sample_tokens=17)}

    def score(self, sampler, prompts, groups):
        self.score_calls.append((sampler, prompts, groups))
        return {"logprobs": self.scored,
                "accounting": accounting(scoring_prefill_tokens=11,
                                         scoring_discarded_sample_tokens_estimate=2)}


class MeasurementTests(unittest.TestCase):
    def test_nll_batches_16_and_weights_by_targets_not_batch_means(self):
        api = BackendDouble()
        rows = [{"prompt_tokens": [1, 2, 3], "completion_tokens": [4], "test_nll": 1.0}] * 16
        rows += [{"tokens": list(range(1, 18)), "test_nll": 3.0}]
        result = measure.evaluate_probe_loss(api, "client", rows)
        self.assertEqual([len(batch) for batch in api.batches], [16, 1])
        self.assertEqual((result["nll"], result["q"], result["target_tokens"]), (2.0, -2.0, 32))
        self.assertEqual(result["accounting"]["forward_tokens"], 64)
        self.assertEqual(result["accounting"]["gradient_target_tokens"], 0)

    def test_native_language_evaluation_does_not_sample(self):
        api = BackendDouble()
        result = measure.evaluate_task(api, "client", "unused", [{"tokens": [1, 2], "test_nll": 4.0}],
                                       {"metric": "negative_nll"}, seed=5)
        self.assertEqual(result["q"], -4.0)
        self.assertEqual(api.sample_calls, [])
        self.assertEqual(api.sampler_paths, [])

    def test_invalid_forward_stops_and_keeps_completed_token_accounting(self):
        api = BackendDouble()
        api.invalid_batch = 2
        result = measure.evaluate_probe_loss(api, "client", [{"tokens": [1, 2], "test_nll": 1.0}] * 40)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["q"])
        self.assertEqual(result["accounting"]["forward_tokens"], 32)
        self.assertEqual(len(api.batches), 2)

    def test_malformed_late_row_does_not_trigger_an_earlier_paid_batch(self):
        api = BackendDouble()
        rows = [{"tokens": [1, 2], "test_nll": 1.0}] * 16 + [{"tokens": [1]}]
        with self.assertRaises(ValueError):
            measure.evaluate_probe_loss(api, "client", rows)
        self.assertEqual(api.batches, [])

    def test_exact_metric_uses_native_checker_and_frozen_sampling(self):
        api = BackendDouble()
        rows = [{**tasks.make_example("json_extraction", index), "prompt_tokens": [1, 2]} for index in (1, 2)]
        api.groups = [[output(rows[0]["completion"])], [output("not json")]]
        manifest = {"metric": "verifier_success", "evaluation": {"max_tokens": 512, "samples": 1, "temperature": 0.0}}
        result = measure.evaluate_task(api, "client", "checkpoint", rows, manifest, 19)
        self.assertEqual((result["q"], result["passed"], result["total"]), (0.5, 1, 2))
        self.assertEqual(api.sample_calls[0][2], {"samples": 1, "max_tokens": 512, "temperature": 0.0, "seed": 19})
        self.assertEqual(result["accounting"]["sample_tokens"], 17)
        manifest["evaluation"]["temperature"] = 1.0
        with self.assertRaises(ValueError):
            measure.evaluate_task(api, "client", "checkpoint", rows, manifest, 19)
        self.assertEqual(len(api.sample_calls), 1)

    def test_if_separate_count_fraction_and_split(self):
        api = BackendDouble()
        selected = if_suite.items(split="heldout")
        api.groups = [[output("invalid output")]] * len(selected)
        expected = if_suite.evaluate({item["id"]: "invalid output" for item in selected}, split="heldout")
        result = measure.evaluate_if(api, "checkpoint", "heldout", seed=19)
        self.assertEqual(result["if_score"], expected["passed"])
        self.assertEqual(result["score"], expected["score"])
        self.assertEqual(result["total"], 30)
        self.assertEqual(api.sample_calls[0][2], {"samples": 1, "max_tokens": 96, "temperature": 0.0, "seed": 19})
        self.assertEqual(set(result["outputs"]), {item["id"] for item in selected})

    def test_forward_kl_fixed_cycle0_orientation_completion_only_no_resampling(self):
        api = BackendDouble()
        groups = [[{"tokens": [3, 4], "logprobs": [-1.0, -2.0]}],
                  [{"tokens": [8], "logprobs": [-0.5]}]]
        api.scored = [[[-2.0, -3.0]], [[-1.5]]]
        result = measure.forward_kl(api, "current", [[1, 2], [9]], groups)
        self.assertEqual(result["kl_to_cycle0"], 1.0)
        self.assertEqual(result["completion_tokens"], 3)
        self.assertEqual(api.score_calls[0][2], groups)
        self.assertEqual(api.sample_calls, [])
        self.assertEqual(result["accounting"]["scoring_prefill_tokens"], 11)
        api.scored = [[[-0.5, -1.5]], [[0.0]]]
        self.assertEqual(measure.forward_kl(api, "current", [[1, 2], [9]], groups)["kl_to_cycle0"], -0.5)

    def test_forward_kl_invalid_frozen_data_fails_before_paid_scoring(self):
        api = BackendDouble()
        with self.assertRaises(ValueError):
            measure.forward_kl(api, "checkpoint", [[1]], [[{"tokens": [2], "logprobs": [math.nan]}]])
        self.assertEqual(api.score_calls, [])

    def test_diversity_coverage_family_quotient_and_unavailable_concentration(self):
        example = probes.make_diversity("graph_coloring", 1)
        colors = json.loads(example["completion"])
        alternate = json.dumps([(value + 1) % 3 for value in colors])
        groups = [[output(example["completion"]), output(alternate)] + [output("invalid")] * 6,
                  [output("invalid", truncated=True)] * 8]
        result = measure.diversity_summary([example, example], groups)
        self.assertEqual((result["pass1"], result["pass8"], result["coverage_gap"]), (0.125, 0.5, 0.375))
        first, second = result["results"]
        self.assertEqual((first["unique_valid_outputs"], first["strategy_families"]), (2, 1))
        self.assertEqual(first["strategy_family_concentration"], 1.0)
        self.assertIsNone(second["strategy_family_concentration"])
        self.assertEqual((result["strategy_family_concentration"], result["concentration_items"]), (1.0, 1))
        self.assertEqual(result["sampled_token_surprisal"], 2.0)
        self.assertEqual(result["truncation_rate"], 0.5)

    def test_diversity_concentration_is_within_item_not_pooled_labels(self):
        example = probes.make_diversity("set_partition", 1)
        families = probes.enumerate_diversity_families(example)
        self.assertGreaterEqual(len(families), 4)
        result = measure.diversity_summary([example], [[output(text) for text in families[:4]] * 2])
        self.assertEqual(result["strategy_family_concentration"], 0.25)
        self.assertEqual(result["strategy_families"], 4)
        self.assertEqual(result["pass1"], 1.0)
        self.assertEqual(result["coverage_gap"], 0.0)

    def test_diversity_requires_exactly_eight_valid_logprob_sequences(self):
        example = probes.make_diversity("graph_coloring", 1)
        with self.assertRaises(ValueError):
            measure.diversity_summary([example], [[output(example["completion"])]] )
        with self.assertRaises(ValueError):
            measure.diversity_summary([example], [[output("invalid", logprob=math.inf)] * 8])

    def test_noise_empirical_only_and_single_measurement_sd_unavailable(self):
        result = measure.noise_summary([1.0, 2.0, 3.0], [(1.0, 1.2), (3.0, 2.7)])
        self.assertEqual((result["count"], result["mean"], result["sample_sd"]), (3, 2.0, 1.0))
        self.assertAlmostEqual(result["paired_absolute_differences"][0], 0.2)
        self.assertAlmostEqual(result["paired_absolute_differences"][1], 0.3)
        self.assertIsNone(measure.noise_summary([1.0])["sample_sd"])
        self.assertNotIn("confidence_interval", result)
        self.assertNotIn("noise_bound", result)
        with self.assertRaises(ValueError):
            measure.noise_summary([math.nan])


if __name__ == "__main__":
    unittest.main()
