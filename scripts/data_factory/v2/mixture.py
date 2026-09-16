"""New-character coverage selection and pre-tokenization final mixture."""

from __future__ import annotations

import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator

from scripts.data_factory.v2.config import DataFactoryConfig
from scripts.data_factory.v2.quality import MixtureMetrics, sub_bucket
from scripts.data_factory.v2.sampling import eligible_buckets, load_new_characters, stable_key


def _input_files(config: DataFactoryConfig, plan: dict[str, Any]) -> list[Path]:
    root = config.run_root / "decontaminated" / str(plan["plan_sha256"])[:16]
    files = sorted(path for path in root.rglob("*.parquet") if path.is_file()) if root.is_dir() else []
    if not files:
        raise FileNotFoundError(f"decontaminated candidate files are missing: {root}")
    return files


def _iter_rows(files: list[Path]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=4096):
            yield from batch.to_pylist()


def _input_schema(files: list[Path]):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(files[0]).schema_arrow
    if "sample_key" not in schema.names:
        raise ValueError("mixture input schema is missing sample_key")
    if not pa.types.is_uint64(schema.field("sample_key").type):
        raise TypeError("mixture input sample_key must be uint64")
    return schema


def _feature_map(config: DataFactoryConfig, new_characters: frozenset[str]) -> dict[str, frozenset[str]]:
    path = config.enhancement.token_ids_path.parent / "features" / "char_feature_index.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"character feature index is missing: {path}")
    result: dict[str, frozenset[str]] = {}
    with path.open("rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            char = row.get("char")
            if char not in new_characters:
                continue
            features: set[str] = set()
            masks = row.get("pinyin_mask", [])
            for index, active in enumerate(masks):
                if not active:
                    continue
                for field in ("pinyin_ids", "shengmu_ids", "yunmu_ids", "tone_ids"):
                    values = row.get(field, [])
                    if index < len(values):
                        if int(values[index]) > 1:
                            features.add(f"{field}:{values[index]}")
            for field in ("stroke_count_id", "radical_stroke_id", "structure_id"):
                value = row.get(field)
                if value is not None and int(value) > 1:
                    features.add(f"{field}:{value}")
            result[char] = frozenset(features)
    missing = new_characters - result.keys()
    if missing:
        raise RuntimeError(f"feature index misses {len(missing)} new Hanzi")
    return result


def _new_chars(text: str, target: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted(set(text) & target, key=ord))


def _write_frequency_reports(
    config: DataFactoryConfig,
    new_characters: frozenset[str],
    tf: Counter[str],
    df: Counter[str],
    source_sets: dict[str, set[str]],
    feature_df: Counter[str],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    report_dir = config.run_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "char": char,
            "tf": tf[char],
            "df": df[char],
            "source_df": len(source_sets.get(char, set())),
        }
        for char in sorted(new_characters, key=ord)
    ]
    pq.write_table(pa.Table.from_pylist(rows), report_dir / "new_hanzi_frequency.parquet", compression="zstd")
    feature_rows = [
        {"feature": feature, "df": count}
        for feature, count in sorted(feature_df.items())
    ]
    pq.write_table(pa.Table.from_pylist(feature_rows), report_dir / "feature_frequency.parquet", compression="zstd")


class ShardedParquetWriter:
    def __init__(self, root: Path, schema: Any, max_rows: int = 100_000) -> None:
        self.root = root
        self.schema = schema
        self.max_rows = max_rows
        self.buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.indices: Counter[str] = Counter()
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, bucket: str, row: dict[str, Any]) -> None:
        self.buffers[bucket].append(row)
        if len(self.buffers[bucket]) >= self.max_rows:
            self._flush(bucket)

    def _flush(self, bucket: str) -> None:
        rows = self.buffers[bucket]
        if not rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        output = self.root / bucket
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"part-{self.indices[bucket]:05d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=self.schema),
            path,
            compression="zstd",
        )
        self.indices[bucket] += 1
        rows.clear()

    def close(self) -> None:
        for bucket in list(self.buffers):
            self._flush(bucket)


def _push_bounded(heap: list[tuple], item: tuple, limit: int) -> None:
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def build_mixture(
    config: DataFactoryConfig,
    plan: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    files = _input_files(config, plan)
    schema = _input_schema(files)
    output_root = config.run_root / "selected" / str(plan["plan_sha256"])[:16]
    if output_root.exists() and any(output_root.rglob("*.parquet")) and not overwrite:
        raise FileExistsError(f"selected mixture already exists: {output_root}")
    if overwrite and output_root.exists():
        import shutil

        shutil.rmtree(output_root)

    new_characters = load_new_characters(config.enhancement.token_ids_path)
    feature_map = _feature_map(config, new_characters)
    enhancement_bucket = config.enhancement.bucket
    enhancement_weights = next(b.source_weights for b in config.buckets if b.name == enhancement_bucket)

    def is_enhancement(row):
        return enhancement_bucket in eligible_buckets(config, str(row["text"]), row, new_characters)
    selection_targets = {
        bucket.name: config.bucket_tokens[bucket.name]
        + int(round(config.validation_tokens * bucket.fraction))
        for bucket in config.buckets
    }
    available_tokens: dict[str, Counter[str]] = defaultdict(Counter)
    tf: Counter[str] = Counter()
    df: Counter[str] = Counter()
    source_sets: dict[str, set[str]] = defaultdict(set)
    feature_df: Counter[str] = Counter()
    enhancement_documents = 0
    enhancement_tokens = 0

    for row in _iter_rows(files):
        source = str(row["source"])
        tokens = int(row["estimated_tokens"])
        if not is_enhancement(row):
            continue
        chars = _new_chars(str(row["text"]), new_characters)
        if not chars:
            continue
        enhancement_documents += 1
        enhancement_tokens += tokens
        tf.update(char for char in row["text"] if char in new_characters)
        df.update(chars)
        features = {feature for char in chars for feature in feature_map[char]}
        feature_df.update(features)
        for char in chars:
            source_sets[char].add(source)

    _write_frequency_reports(config, new_characters, tf, df, source_sets, feature_df)
    if enhancement_documents == 0:
        raise RuntimeError("new-character candidate bucket is empty")

    target_tokens = selection_targets[enhancement_bucket]
    average_tokens = enhancement_tokens / enhancement_documents
    fill_limit = min(4_000_000, max(10_000, math.ceil(target_tokens / max(1.0, average_tokens) * 1.5)))
    per_char_limit = max(config.enhancement.coverage_targets, default=100)
    fill_heaps: dict[str, list[tuple]] = defaultdict(list)
    character_heaps: dict[str, list[tuple]] = defaultdict(list)

    for row in _iter_rows(files):
        if not is_enhancement(row):
            continue
        chars = _new_chars(str(row["text"]), new_characters)
        if not chars:
            continue
        features = {feature for char in chars for feature in feature_map[char]}
        score_new = sum(1.0 / math.sqrt(df[char] + 1) for char in chars)
        score_feature = sum(1.0 / math.sqrt(feature_df[feature] + 1) for feature in features)
        score = score_new + config.enhancement.feature_score_weight * score_feature
        tie = stable_key(config.seed, "enhancement", str(row["id"]))
        item = (
            score,
            tie,
            str(row["id"]),
            int(row["estimated_tokens"]),
            str(row["source"]),
            tuple(row.get("tags") or []),
            chars,
            str(row.get("parent_doc_id") or row["id"]),
        )
        _push_bounded(fill_heaps[str(row["source"])], item, fill_limit)
        for char in chars:
            _push_bounded(character_heaps[char], item, per_char_limit)

    selected: dict[str, tuple] = {}
    selected_parents: set[str] = set()
    selected_tokens = 0
    source_tokens: Counter[str] = Counter()
    tag_tokens: Counter[str] = Counter()
    selected_char_df: Counter[str] = Counter()
    selected_feature_df: Counter[str] = Counter()
    coverage_achieved: Counter[int] = Counter()
    coverage_goals = tuple(sorted(config.enhancement.coverage_targets))
    constraints = config.enhancement.constraints

    def can_select(item: tuple) -> bool:
        nonlocal selected_tokens
        _score, _tie, doc_id, tokens, source, tags, _chars, parent = item
        if parent in selected_parents or selected_tokens + tokens > target_tokens * 1.01:
            return False
        source_limit = target_tokens * min(
            constraints.get("single_source_max_fraction", 1.0), enhancement_weights[source]
        )
        if source_tokens[source] + tokens > source_limit:
            return False
        if "classical" in tags or "classical_candidate" in tags:
            classical_limit = target_tokens * constraints.get("classical_max_fraction", 1.0)
            if tag_tokens["classical"] + tokens > classical_limit:
                return False
        return True

    def add(item: tuple) -> bool:
        nonlocal selected_tokens
        if not can_select(item):
            return False
        _score, _tie, doc_id, tokens, source, tags, chars, parent = item
        selected[doc_id] = item
        selected_parents.add(parent)
        selected_tokens += tokens
        source_tokens[source] += tokens
        if "traditional" in tags:
            tag_tokens["traditional"] += tokens
        if "classical" in tags or "classical_candidate" in tags:
            tag_tokens["classical"] += tokens
        else:
            tag_tokens["modern"] += tokens
        for char in chars:
            before = selected_char_df[char]
            selected_char_df[char] += 1
            for goal in coverage_goals:
                if before < goal <= selected_char_df[char]:
                    coverage_achieved[goal] += 1
        selected_feature_df.update(
            {feature for char in chars for feature in feature_map[char]}
        )
        return True

    observed = sorted(df, key=lambda char: (df[char], ord(char)))
    for goal in sorted(config.enhancement.coverage_targets):
        eligible = [char for char in observed if df[char] >= goal]
        required = math.ceil(len(new_characters) * config.enhancement.coverage_targets[goal])
        for char in eligible:
            if coverage_achieved[goal] >= required:
                break
            for item in sorted(character_heaps[char], reverse=True):
                if selected_char_df[char] >= goal or selected_tokens >= target_tokens:
                    break
                add(item)

    ranked_fill = sorted((item for heap in fill_heaps.values() for item in heap), reverse=True)

    def fill_constraint(tag: str, fraction: float, predicate: Callable[[tuple], bool]) -> None:
        target = int(target_tokens * fraction)
        for item in ranked_fill:
            if tag_tokens[tag] >= target or selected_tokens >= target_tokens:
                break
            if predicate(item):
                add(item)

    fill_constraint("modern", constraints.get("modern_min_fraction", 0.0), lambda item: "classical" not in item[5] and "classical_candidate" not in item[5])
    fill_constraint("traditional", constraints.get("traditional_min_fraction", 0.0), lambda item: "traditional" in item[5])
    for item in ranked_fill:
        if selected_tokens >= target_tokens:
            break
        add(item)

    # Recount residual ordinary capacity after ownership is assigned to enhancement.
    available_tokens.clear()
    for row in _iter_rows(files):
        if str(row.get("parent_doc_id") or row["id"]) not in selected_parents and row["candidate_bucket"] != enhancement_bucket:
            available_tokens[str(row["candidate_bucket"])][str(row["source"])] += int(row["estimated_tokens"])

    import pyarrow as pa
    import pyarrow.parquet as pq

    report_dir = config.run_root / "reports"
    selected_frequency_rows = [
        {"char": char, "selected_df": selected_char_df[char]}
        for char in sorted(new_characters, key=ord)
    ]
    pq.write_table(
        pa.Table.from_pylist(selected_frequency_rows),
        report_dir / "new_hanzi_selected_frequency.parquet",
        compression="zstd",
    )
    selected_feature_rows = [
        {"feature": feature, "selected_df": selected_feature_df[feature]}
        for feature in sorted(feature_df)
    ]
    pq.write_table(
        pa.Table.from_pylist(selected_feature_rows),
        report_dir / "feature_coverage.parquet",
        compression="zstd",
    )

    bucket_actual: Counter[str] = Counter()
    source_actual: dict[str, Counter[str]] = defaultdict(Counter)
    metrics = MixtureMetrics(config)
    written_parents: set[str] = set()
    writer = ShardedParquetWriter(output_root, schema=schema)

    def ordered_rows():
        for row in _iter_rows(files):
            if str(row["id"]) in selected:
                yield row
        # Reserve scarce horizontal attributes before ordinary quota filling.
        for name, limits in config.attributes.items():
            minimum = sum(selection_targets.values()) * limits.get("min_fraction", 0)
            for row in _iter_rows(files):
                if metrics.attributes[name] >= minimum:
                    break
                tags = set(row.get("tags") or [])
                matches = "long_doc" in tags if name == "long_document" else bool(tags & {"classical", "classical_candidate"})
                if matches:
                    yield row
        yield from _iter_rows(files)

    for row in ordered_rows():
        bucket = str(row["candidate_bucket"])
        source = str(row["source"])
        tokens = int(row["estimated_tokens"])
        parent = str(row.get("parent_doc_id") or row["id"])
        if parent in written_parents:
            continue
        if str(row["id"]) in selected:
            bucket = enhancement_bucket
            row = {**row, "candidate_bucket": bucket}
            keep = True
        elif bucket == enhancement_bucket or parent in selected_parents:
            keep = False
        else:
            bucket_spec = next(value for value in config.buckets if value.name == bucket)
            target = int(round(selection_targets[bucket] * bucket_spec.source_weights[source]))
            keep = source_actual[bucket][source] < target
            if bucket_spec.sub_buckets:
                name = sub_bucket(row)
                keep = keep and metrics.subs[bucket][name] < selection_targets[bucket] * bucket_spec.sub_buckets.get(name, 0)
        if keep:
            tags = set(row.get("tags") or [])
            classical_max = config.attributes.get("classical_chinese", {}).get("max_fraction", 1)
            if tags & {"classical", "classical_candidate"} and metrics.attributes["classical_chinese"] + tokens > sum(selection_targets.values()) * classical_max:
                continue
            writer.write(bucket, row)
            bucket_actual[bucket] += tokens
            source_actual[bucket][source] += tokens
            written_parents.add(parent)
            metrics.add(row, tokens)
    writer.close()

    coverage = {
        str(goal): {
            "required_fraction": fraction,
            "actual_characters": sum(selected_char_df[char] >= goal for char in new_characters),
            "actual_fraction": sum(selected_char_df[char] >= goal for char in new_characters) / len(new_characters),
        }
        for goal, fraction in sorted(config.enhancement.coverage_targets.items())
    }
    coverage_passed = all(
        value["actual_fraction"] >= value["required_fraction"]
        for value in coverage.values()
    )
    source_cap = constraints.get("single_source_max_fraction", 1.0)
    constraint_checks = {
        "single_source": max(source_tokens.values(), default=0) <= max(1, selected_tokens) * source_cap * 1.01,
        "modern": tag_tokens["modern"] >= target_tokens * constraints.get("modern_min_fraction", 0.0) * 0.95,
        "traditional": tag_tokens["traditional"] >= target_tokens * constraints.get("traditional_min_fraction", 0.0) * 0.95,
        "classical": tag_tokens["classical"] <= target_tokens * constraints.get("classical_max_fraction", 1.0) * 1.01,
    }
    checks = {
        bucket.name: {
            "target_tokens": selection_targets[bucket.name],
            "estimated_tokens": bucket_actual[bucket.name],
            "passed": bucket_actual[bucket.name] >= selection_targets[bucket.name] * 0.99,
        }
        for bucket in config.buckets
    }
    shortfalls = {
        bucket.name: max(0, selection_targets[bucket.name] - bucket_actual[bucket.name])
        for bucket in config.buckets
        if bucket_actual[bucket.name] < selection_targets[bucket.name]
    }
    report = {
        "stage": "mixture",
        "phase": config.phase,
        "run_id": config.run_id,
        "plan_sha256": plan["plan_sha256"],
        "passed": (
            all(value["passed"] for value in checks.values())
            and all(constraint_checks.values())
            and metrics.report()["passed"]
        ),
        "bucket_checks": checks,
        "distribution": metrics.report(),
        "estimated_shortfalls": shortfalls,
        "ordinary_available_after_enhancement": {b: dict(values) for b, values in available_tokens.items()},
        "source_shortfalls": {
            b.name: {
                source: max(0, int(round(selection_targets[b.name] * weight)) - source_actual[b.name][source])
                for source, weight in b.source_weights.items()
            }
            for b in config.buckets
        },
        "source_tokens": {bucket: dict(values) for bucket, values in source_actual.items()},
        "enhancement": {
            "selected_documents": len(selected),
            "estimated_tokens": selected_tokens,
            "source_tokens": dict(source_tokens),
            "tag_tokens": dict(tag_tokens),
            "coverage": coverage,
            "coverage_passed": coverage_passed,
            "coverage_is_diagnostic": True,
            "candidate_coverage": {
                str(goal): sum(df[char] >= goal for char in new_characters)
                for goal in config.enhancement.coverage_targets
            },
            "constraint_checks": constraint_checks,
        },
        "output": str(output_root),
    }
    report_path = config.run_root / "reports" / "mixture_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
