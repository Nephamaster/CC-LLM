from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from scripts.data_factory.config import load_config
from scripts.data_factory.io_utils import iter_jsonl
from scripts.data_factory.prepare import prepare_sources


class PrepareAndDedupTests(unittest.TestCase):
    def _config(self, root: Path, sources: dict) -> object:
        raw = json.loads(Path("scripts/data_factory/phase1_config.json").read_text(encoding="utf-8"))
        raw.update(
            {
                "repo_root": str(root),
                "phase_root": "phase1",
                "tokenizer_path": "model",
                "prepare_workers": 2,
                "dedup_workers": 2,
                "sources": sources,
            }
        )
        raw["quality"]["min_chars"] = 1
        raw["dedup"].update(
            {
                "near_min_chars": 100_000,
                "decontamination_paths": [],
                "batch_size": 2,
                "batch_chars": 10_000,
                "sqlite_cache_mb": 64,
                "export_registry_parquet": False,
            }
        )
        path = root / "config.json"
        path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return load_config(path)

    def test_prepare_splits_source_files_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_root = root / "cci"
            source_root.mkdir()
            for file_index in range(2):
                rows = [
                    {
                        "id": f"{file_index}-{row_index}",
                        "text": f"第{file_index}个文件的第{row_index}条有效中文文本。",
                        "score": 1.0,
                    }
                    for row_index in range(2)
                ]
                (source_root / f"part_{file_index:06d}.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )
            config = self._config(
                root,
                {
                    "cci3_hq": {
                        "enabled": True,
                        "paths": [str(source_root / "part_*.jsonl")],
                        "license": "Apache-2.0",
                    },
                    "external": [],
                },
            )

            config.normalized_dir.mkdir(parents=True)
            (config.normalized_dir / "clue-00000.jsonl").write_text("{}\n", encoding="utf-8")

            report = prepare_sources(config, overwrite=True, workers=2)
            self.assertEqual(report["sources"]["cci3_hq"]["kept"], 4)
            self.assertEqual(len(list(config.normalized_dir.glob("cci3_hq-t*.jsonl"))), 2)
            self.assertFalse((config.normalized_dir / "clue-00000.jsonl").exists())

            resumed = prepare_sources(config, resume=True, workers=1)
            self.assertEqual(resumed["sources"]["cci3_hq"]["kept"], 4)

            unchanged_output = config.normalized_dir / "cci3_hq-t00000-00000.jsonl"
            unchanged_mtime = unchanged_output.stat().st_mtime_ns
            changed_input = source_root / "part_000001.jsonl"
            with changed_input.open("at", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {"id": "1-2", "text": "重新下载文件中新增的有效中文文本。", "score": 1.0},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            resumed = prepare_sources(config, resume=True, workers=1)
            self.assertEqual(resumed["sources"]["cci3_hq"]["kept"], 5)
            self.assertEqual(unchanged_output.stat().st_mtime_ns, unchanged_mtime)
            changed_output = config.normalized_dir / "cci3_hq-t00001-00000.jsonl"
            self.assertEqual(len(list(iter_jsonl([changed_output]))), 3)

    def test_dedup_is_global_across_shards_and_resumes(self) -> None:
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy is not installed in the local interpreter")
        from scripts.data_factory.dedup import deduplicate

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = self._config(root, {"external": []})
            config.normalized_dir.mkdir(parents=True)
            shared = "跨分片重复文本。" * 8
            shards = [
                [
                    {"doc_id": "a", "text": shared, "source": "s1", "category": "chinese_natural"},
                    {"doc_id": "b", "text": "唯一文本甲。" * 8, "source": "s1", "category": "chinese_natural"},
                ],
                [
                    {"doc_id": "c", "text": shared, "source": "s2", "category": "chinese_natural"},
                    {"doc_id": "d", "text": "唯一文本乙。" * 8, "source": "s2", "category": "chinese_natural"},
                ],
            ]
            for shard_index, rows in enumerate(shards):
                (config.normalized_dir / f"source-{shard_index:05d}.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )

            report = deduplicate(config, overwrite=True, workers=2)
            self.assertEqual(report["input_records"], 4)
            self.assertEqual(report["kept_records"], 3)
            self.assertEqual(report["removed_by_reason"], {"exact": 1})
            kept = list(iter_jsonl(sorted(config.deduplicated_dir.glob("part-*.jsonl"))))
            self.assertEqual([row["doc_id"] for row in kept], ["a", "b", "d"])

            resumed = deduplicate(config, resume=True, workers=1)
            self.assertEqual(resumed["kept_records"], 3)
            self.assertEqual(resumed["removed_by_reason"], {"exact": 1})


if __name__ == "__main__":
    unittest.main()