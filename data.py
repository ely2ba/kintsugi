"""Freeze concrete corpora, native checkers, and task-keyed example schedules.

No model outcomes enter data preparation. Downloaded source text and token rows
stay local; public manifests contain source revisions, splits, counts and hashes.
"""
import argparse
import hashlib
import json
import lzma
import random
import statistics
import unicodedata
from pathlib import Path

from calibrate import canonical
from protocol import (IF_HASHES, MODEL, REPAIR_POOL_HASH, TASK_SLOTS,
                      TOKENIZER_REVISION, LANGUAGE_PROBES, order_manifests, schedule_seed)

WIKIPEDIA_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKIPEDIA_SHARDS = {"is": 1, "eu": 2, "vi": 4, "id": 3, "fi": 3}
SOURCES = {
    f"wikipedia_{language}": {
        "dataset": "wikimedia/wikipedia", "revision": WIKIPEDIA_REVISION,
        "config": f"20231101.{language}", "split": "train",
        "files": [f"20231101.{language}/train-{i:05d}-of-{count:05d}.parquet" for i in range(count)],
        "license": "CC-BY-SA-3.0 / GFDL as declared by dataset card",
        "text_field": "text", "document_id": "id",
    }
    for language, count in WIKIPEDIA_SHARDS.items()
}
SOURCES.update({
    "legal_text": {
        "dataset": "pile-of-law/pile-of-law", "revision": "2e96169e7e4b43f8ea36230515ebb44b27423b94",
        "config": "scotus_oral_arguments", "split": "train",
        "files": ["data/train.scotus_oral.jsonl.xz"],
        "license": "CC-BY-NC-SA-4.0 collection; U.S. Supreme Court oral-argument transcripts",
        "text_field": "text", "document_id": "normalized_text_sha256 (one transcript per row; source URL is collection-wide)",
    },
    "biomedical_abstracts": {
        "dataset": "qiaojin/PubMedQA", "revision": "9001f2853fb87cab8d220904e0de81ac6973b318",
        "config": "pqa_unlabeled", "split": "train",
        "files": ["pqa_unlabeled/train-00000-of-00001.parquet"],
        "license": "MIT dataset; underlying abstract rights remain with their publishers",
        "text_field": "context.contexts + long_answer (abstract conclusion); no questions or labels",
        "document_id": "pubid",
    },
})
TRAIN_SPLITS = ("reference", "screen1", "screen2", "main", "persistence")
EVAL_COUNTS = {"gate": 128, "heldout": 128}
LM_BATCH, LM_SEQUENCE_TOKENS, LM_MIN_TOKENS = 16, 513, 65
MAX_CHUNKS_PER_DOCUMENT = 8


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path, value):
    path = Path(path)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"frozen artifact differs: {path}; do not overwrite a freeze")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def write_rows(path, rows):
    path = Path(path)
    payload = b"".join((canonical(row) + "\n").encode() for row in rows)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"frozen rows differ: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def load_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def render_prompt(tokenizer, prompt):
    # Exact v1 non-thinking prefix; acquisition prompts are never truncated.
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return tokenizer.encode(text, add_special_tokens=False)


def encode_example(tokenizer, example):
    return {**example, "prompt_tokens": render_prompt(tokenizer, example["prompt"]),
            "completion_tokens": tokenizer.encode(example["completion"] + "<|im_end|>", add_special_tokens=False)}


def fixed_synthetic_batch(specimens):
    """Pre-outcome workload rule, applied once to 128 unscored specimens."""
    total = sorted(len(row["prompt_tokens"]) + len(row["completion_tokens"]) for row in specimens)
    mean_target = statistics.mean(len(row["completion_tokens"]) for row in specimens)
    p95 = total[(95 * len(total) + 99) // 100 - 1]
    size = 64 if p95 <= 512 and mean_target <= 128 else 32 if p95 <= 1024 and mean_target <= 256 else 16
    return size, {"specimens": len(specimens), "p95_total_tokens": p95, "mean_target_tokens": mean_target,
                  "rule": "64 if p95_total<=512 and mean_target<=128; else 32 if <=1024 and <=256; else 16"}


def synthetic_splits(candidate, slot, tokenizer):
    from tasks import INSTANCE_LIMITS, make_example, verify
    specimens = [encode_example(tokenizer, make_example(candidate, i)) for i in range(128)]
    batch_size, workload = fixed_synthetic_batch(specimens)
    counts = {**{split: 120 * batch_size for split in TRAIN_SPLITS}, **EVAL_COUNTS}
    start, splits, keys = 1024, {}, set()
    if start + sum(counts.values()) > INSTANCE_LIMITS[candidate]:
        raise RuntimeError("generator space is insufficient for disjoint frozen splits")
    for split, count in counts.items():
        rows = []
        for index in range(start, start + count):
            example = make_example(candidate, index)
            if example["semantic_key"] in keys or not verify(example, example["completion"]):
                raise RuntimeError(f"semantic collision or invalid gold: {candidate}/{split}/{index}")
            keys.add(example["semantic_key"])
            rows.append(encode_example(tokenizer, example))
        random.Random(schedule_seed(slot, "example-order:" + split)).shuffle(rows)
        splits[split] = rows
        start += count
    return batch_size, workload, splits


def source_documents(candidate):
    """Read pinned public files only, never a mutable dataset branch or script."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as parquet
    source = SOURCES[candidate]
    for relative in source["files"]:
        path = hf_hub_download(source["dataset"], relative, repo_type="dataset", revision=source["revision"])
        if relative.endswith(".xz"):
            with lzma.open(path, "rt", encoding="utf-8") as handle:
                rows = (json.loads(line) for line in handle)
                for row in rows:
                    text = row["text"]
                    normalized = " ".join(unicodedata.normalize("NFC", text).split())
                    doc_id = hashlib.sha256(normalized.encode()).hexdigest()
                    yield doc_id, text, relative
        else:
            for batch in parquet.ParquetFile(path).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    text = ("\n".join(row["context"]["contexts"] + [row["long_answer"]])
                            if candidate == "biomedical_abstracts" else row["text"])
                    yield str(row[source["document_id"]]), text, relative


def natural_splits(candidate, slot, tokenizer, documents=None, probe=False):
    counts = ({"train": 490 * 8, "val": 128} if probe else
              {**{split: 120 * LM_BATCH for split in TRAIN_SPLITS}, **EVAL_COUNTS})
    names = tuple(counts)
    splits = {name: [] for name in names}
    seen_ids, seen_texts = set(), set()
    used_files = set()
    for doc_id, text, relative in documents if documents is not None else source_documents(candidate):
        normalized = " ".join(unicodedata.normalize("NFC", text).split())
        text_hash = hashlib.sha256(normalized.encode()).hexdigest()
        if not normalized or doc_id in seen_ids or text_hash in seen_texts:
            continue
        seen_ids.add(doc_id)
        seen_texts.add(text_hash)
        digest = hashlib.sha256(f"{candidate}\0{doc_id}".encode()).digest()
        # Document assignment precedes chunking. No document crosses a split.
        bucket = int.from_bytes(digest[:8], "big") % sum(counts.values())
        bound = 0
        for split in names:
            bound += counts[split]
            if bucket < bound:
                break
        if len(splits[split]) >= counts[split]:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        for chunk, start in enumerate(range(0, len(tokens), LM_SEQUENCE_TOKENS)):
            if chunk == MAX_CHUNKS_PER_DOCUMENT or len(splits[split]) == counts[split]:
                break
            values = tokens[start:start + LM_SEQUENCE_TOKENS]
            if len(values) < LM_MIN_TOKENS:
                continue
            splits[split].append({"candidate": candidate, "document_id": doc_id,
                                   "document_text_sha256": text_hash, "source_file": relative,
                                   "chunk": chunk, "tokens": values})
            used_files.add(relative)
        if all(len(splits[name]) == count for name, count in counts.items()):
            break
    if any(len(splits[name]) != count for name, count in counts.items()):
        raise RuntimeError(f"pinned source exhausted before split quotas: {candidate}: "
                           + str({name: len(rows) for name, rows in splits.items()}))
    for split, rows in splits.items():
        random.Random(schedule_seed(slot, "example-order:" + split)).shuffle(rows)
    return splits, sorted(used_files)


def freeze_candidate(root, candidate, tokenizer):
    matches = [slot for slot, candidates in TASK_SLOTS.items() if candidate in candidates]
    if len(matches) != 1:
        raise ValueError("not an acquisition candidate")
    slot = matches[0]
    if candidate in SOURCES:
        batch_size, workload = LM_BATCH, {"max_sequence_tokens": LM_SEQUENCE_TOKENS,
                                          "minimum_sequence_tokens": LM_MIN_TOKENS,
                                          "max_chunks_per_document": MAX_CHUNKS_PER_DOCUMENT}
        splits, used_files = natural_splits(candidate, slot, tokenizer)
        source = {**SOURCES[candidate], "used_files": used_files}
    else:
        batch_size, workload, splits = synthetic_splits(candidate, slot, tokenizer)
        source = {"generator": "tasks.py", "generator_sha256": sha256_file(Path(__file__).with_name("tasks.py")),
                  "instance_indices": "unscored specimens 0..127; disjoint data starts at 1024"}
    manifest = {"candidate": candidate, "slot": slot, "metric": "negative_nll" if candidate in SOURCES else "verifier_success",
                "batch_size": batch_size, "workload": workload, "source": source,
                "tokenizer": MODEL, "tokenizer_revision": TOKENIZER_REVISION,
                "evaluation": {"temperature": 0.0, "samples": 1, "max_tokens": 512}, "splits": {}}
    for split, rows in splits.items():
        relative = f"data/{candidate}/{split}.jsonl"
        digest = write_rows(root / relative, rows)
        manifest["splits"][split] = {"path": relative, "sha256": digest, "examples": len(rows),
                                       "gradient_target_tokens": sum(len(row.get("completion_tokens", []))
                                           if "completion_tokens" in row else len(row["tokens"]) - 1 for row in rows)}
    write_once(root / f"manifests/tasks/{candidate}.json", manifest)
    return manifest


def freeze_shared(root, v1_pool):
    from if_suite import ITEMS, assert_prompt_hash_disjoint, manifest_hashes, prompt_hash
    if manifest_hashes() != IF_HASHES:
        raise RuntimeError("v1 IF suite changed")
    pool = load_rows(v1_pool)
    if len(pool) != 2001 or pool[0]["pool_sha256"] != REPAIR_POOL_HASH:
        raise RuntimeError("wrong v1 repair pool")
    body_hash = hashlib.sha256(b"".join((canonical(row) + "\n").encode() for row in pool[1:])).hexdigest()
    if body_hash != REPAIR_POOL_HASH or any(prompt_hash(row["prompt"]) != row["prompt_sha256"] for row in pool[1:]):
        raise RuntimeError("repair pool content mismatch")
    assert_prompt_hash_disjoint(pool[1:])
    write_rows(root / "data/repair_pool.jsonl", pool)
    write_once(root / "manifests/shared.json", {"if_hashes": IF_HASHES, "repair_pool": pool[0],
                                               "repair_pool_file_sha256": sha256_file(root / "data/repair_pool.jsonl")})
    for order, manifest in order_manifests().items():
        write_once(root / f"manifests/orders/{order}.json", manifest)
    write_once(root / "manifests/sources.json", SOURCES)


def freeze_probe(root, candidate, tokenizer):
    from probes import PROBE_CANDIDATES, make_probe, verify_probe
    if candidate in LANGUAGE_PROBES:
        splits, files = natural_splits(candidate, candidate, tokenizer, probe=True)
        probe_class, batch_size = "language", 8
        source = {**SOURCES[candidate], "used_files": files}
    elif candidate in PROBE_CANDIDATES:
        splits, offset, keys = {}, 1024, set()
        for split, count in (("train", 64 * 32), ("val", 128)):
            rows = []
            for index in range(offset, offset + count):
                example = make_probe(candidate, index)
                if example["semantic_key"] in keys or not verify_probe(example, example["completion"]):
                    raise RuntimeError("probe semantic collision or invalid gold")
                keys.add(example["semantic_key"])
                rows.append(encode_example(tokenizer, example))
            random.Random(schedule_seed(candidate, "example-order:" + split)).shuffle(rows)
            splits[split] = rows
            offset += count
        probe_class, batch_size = "structured", 32
        source = {"generator": "probes.py", "generator_sha256": sha256_file(Path(__file__).with_name("probes.py"))}
    else:
        raise ValueError("not a registered probe candidate")
    manifest = {"candidate": candidate, "class": probe_class, "batch_size": batch_size,
                "source": source, "tokenizer": MODEL, "tokenizer_revision": TOKENIZER_REVISION, "splits": {}}
    for split, rows in splits.items():
        relative = f"data/probes/{candidate}/{split}.jsonl"
        manifest["splits"][split] = {"path": relative, "sha256": write_rows(root / relative, rows), "examples": len(rows)}
    write_once(root / f"manifests/probes/{candidate}.json", manifest)
    return manifest


def freeze_diversity(root, candidate, tokenizer):
    from probes import DIVERSITY_CANDIDATES, make_diversity, verify_diversity
    if candidate not in DIVERSITY_CANDIDATES:
        raise ValueError("not a declared diversity candidate")
    rows = [encode_example(tokenizer, make_diversity(candidate, i)) for i in range(100)]
    if len({row["semantic_key"] for row in rows}) != len(rows):
        raise RuntimeError("diversity panel repeats an underlying construction")
    if any(verify_diversity(row, row["completion"]) is None for row in rows):
        raise RuntimeError("invalid diversity specimen")
    relative = f"data/diversity/{candidate}.jsonl"
    manifest = {"candidate": candidate, "examples": 100, "path": relative,
                "sha256": write_rows(root / relative, rows), "generator": "probes.py",
                "generator_sha256": sha256_file(Path(__file__).with_name("probes.py")),
                "sampling": {"samples": 8, "temperature": 1.0, "max_tokens": 512, "seed": 20260831},
                "safe_length_rule": {"max_truncation_rate": 0.0, "p95_tokens_at_most": 384},
                "family_coverage": sum(row["family_count"] >= 4 for row in rows) / len(rows),
                "family_definition": "unlabeled vertex partitions / unordered equal-sum partitions; not latent reasoning",
                "selection": "first qualifying candidate in graph_coloring, set_partition order; no post-outcome tuning"}
    write_once(root / f"manifests/diversity/{candidate}.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(candidate for pair in TASK_SLOTS.values() for candidate in pair))
    parser.add_argument("--v1-pool", type=Path)
    parser.add_argument("--probe", choices=("graph_path", "calendar_arithmetic", "unit_conversion", *LANGUAGE_PROBES))
    parser.add_argument("--diversity", choices=("graph_coloring", "set_partition"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.v1_pool:
        freeze_shared(root, args.v1_pool)
    if args.candidate or args.probe or args.diversity:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION, local_files_only=True)
        if args.candidate:
            manifest = freeze_candidate(root, args.candidate, tokenizer)
        elif args.probe:
            manifest = freeze_probe(root, args.probe, tokenizer)
        else:
            manifest = freeze_diversity(root, args.diversity, tokenizer)
        print(canonical({"candidate": manifest["candidate"], "batch_size": manifest.get("batch_size"),
                         "splits": {name: item["examples"] for name, item in manifest.get("splits", {}).items()}}))
    if not any((args.candidate, args.v1_pool, args.probe, args.diversity)):
        parser.error("choose --candidate or --v1-pool; neither makes model calls")


if __name__ == "__main__":
    main()
