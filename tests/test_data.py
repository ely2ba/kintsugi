import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import data
from protocol import ARMS, ORDERS, TASK_SLOTS, order_manifests, schedule_seed


class DataTests(unittest.TestCase):
    def test_exact_orders_and_common_task_schedule(self):
        manifests = order_manifests()
        expected = {"O1": "1234567", "O2": "2416375", "O3": "5361742", "O4": "7625431"}
        edges = []
        for order, manifest in manifests.items():
            self.assertEqual("".join(slot[1:] for slot in manifest["slots"]), expected[order])
            self.assertEqual({row["arm"] for row in manifest["lineages"]}, set(ARMS))
            edges.extend(zip(manifest["slots"], manifest["slots"][1:]))
            for row in manifest["cycles"]:
                self.assertEqual(row["task_schedule_key"], row["repair_schedule_key"])
        self.assertEqual(len(edges), len(set(edges)))
        for task in TASK_SLOTS:
            self.assertEqual(len({sequence.index(task) for sequence in ORDERS.values()}), 4)
            self.assertEqual(schedule_seed(task, "repair-rollouts", 3), schedule_seed(task, "repair-rollouts", 3))
        with self.assertRaises(ValueError):
            schedule_seed("fixed-O1", "repair-rollouts", 3)

    def test_preoutcome_batch_rule(self):
        for prompt, completion, expected in ((100, 20, 64), (500, 200, 32), (1000, 300, 16)):
            rows = [{"prompt_tokens": [1] * prompt, "completion_tokens": [2] * completion}] * 128
            self.assertEqual(data.fixed_synthetic_batch(rows)[0], expected)

    def test_document_level_splits_before_chunking_and_duplicate_removal(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(range(100))
        documents = ((str(i), f"unique text document {i}", "pinned-file") for i in range(10000))
        with patch.object(data, "LM_BATCH", 1):
            splits, files = data.natural_splits("legal_text", "T7", Tokenizer(), documents)
        seen = set()
        for split, rows in splits.items():
            ids = {row["document_id"] for row in rows}
            self.assertFalse(seen & ids)
            seen.update(ids)
            self.assertEqual(len(rows), 128 if split in ("gate", "heldout") else 120)
        self.assertEqual(files, ["pinned-file"])

    def test_insufficient_source_cannot_silently_repeat_or_reduce_split(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(range(100))
        duplicate_documents = [(str(i), "same text", "file") for i in range(10000)]
        with self.assertRaises(RuntimeError):
            data.natural_splits("legal_text", "T7", Tokenizer(), duplicate_documents)

    def test_artifacts_are_write_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            data.write_once(path, {"value": 1})
            data.write_once(path, {"value": 1})
            with self.assertRaises(RuntimeError):
                data.write_once(path, {"value": 2})


if __name__ == "__main__":
    unittest.main()
