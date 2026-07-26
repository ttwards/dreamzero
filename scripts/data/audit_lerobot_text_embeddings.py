#!/usr/bin/env python3
"""Audit LeRobot v3 task texts against per-text embedding cache files."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


def embedding_hash(task: str) -> str:
    """Match the cache key used by ``LeRobotSingleDataset``."""
    return hashlib.sha1(task.encode("utf-8")).hexdigest()[:16]


def audit_split(dataset_root: Path, split: str) -> dict:
    split_root = dataset_root / split
    tasks_path = split_root / "meta" / "tasks.parquet"
    embedding_dir = split_root / "text_embs"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot v3 tasks metadata: {tasks_path}")
    if not embedding_dir.is_dir():
        raise FileNotFoundError(f"Missing text embedding directory: {embedding_dir}")

    tasks = pq.read_table(tasks_path, columns=["task_index", "task"]).to_pydict()
    grouped: dict[str, dict] = {}
    for task_index, raw_task in zip(tasks["task_index"], tasks["task"], strict=True):
        task = str(raw_task)
        task_hash = embedding_hash(task)
        entry = grouped.setdefault(
            task,
            {
                "task": task,
                "sha1_16": task_hash,
                "task_indices": [],
                "expected_embedding": f"text_embs/{task_hash}.pt",
            },
        )
        entry["task_indices"].append(int(task_index))

    missing = [
        entry
        for entry in grouped.values()
        if not (embedding_dir / f"{entry['sha1_16']}.pt").is_file()
    ]
    missing.sort(key=lambda entry: (entry["task_indices"][0], entry["task"]))

    return {
        "dataset": split,
        "tasks_file": str(tasks_path.relative_to(dataset_root)),
        "embedding_dir": str(embedding_dir.relative_to(dataset_root)),
        "task_rows": len(tasks["task"]),
        "unique_task_texts": len(grouped),
        "present_unique_embeddings": len(grouped) - len(missing),
        "missing_unique_embeddings": len(missing),
        "affected_task_rows": sum(len(entry["task_indices"]) for entry in missing),
        "missing": missing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dagger", "multitask", "singletask", "val"],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    datasets = [audit_split(dataset_root, split) for split in args.splits]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "cache_key": "sha1(raw_task_text)[:16]",
        "summary": {
            "dataset_count": len(datasets),
            "task_rows": sum(item["task_rows"] for item in datasets),
            "unique_task_texts_per_dataset": sum(
                item["unique_task_texts"] for item in datasets
            ),
            "present_unique_embeddings": sum(
                item["present_unique_embeddings"] for item in datasets
            ),
            "missing_unique_embeddings": sum(
                item["missing_unique_embeddings"] for item in datasets
            ),
            "affected_task_rows": sum(item["affected_task_rows"] for item in datasets),
        },
        "datasets": datasets,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(
        f"wrote {args.output}: "
        f"{summary['missing_unique_embeddings']} missing unique embeddings "
        f"across {summary['dataset_count']} datasets"
    )


if __name__ == "__main__":
    main()
