"""Streaming file helpers used by the data pipeline."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wt", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def iter_jsonl(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open("rt", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                yield row


def expand_paths(patterns: str | Iterable[str], root: Path) -> list[Path]:
    values = [patterns] if isinstance(patterns, str) else list(patterns)
    paths: set[Path] = set()
    for value in values:
        candidate = Path(value).expanduser()
        pattern = str(candidate if candidate.is_absolute() else root / candidate)
        paths.update(Path(match).resolve() for match in glob.glob(pattern, recursive=True) if Path(match).is_file())
    return sorted(paths)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class JsonlShardWriter:
    def __init__(self, directory: Path, prefix: str, max_records: int) -> None:
        self.directory = directory
        self.prefix = prefix
        self.max_records = max_records
        self.directory.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._path: Path | None = None
        self._digest = None
        self._shard_index = 0
        self._records_in_shard = 0
        self.total_records = 0
        self.files: list[dict[str, Any]] = []

    def _open_next(self) -> None:
        self.close_shard()
        self._path = self.directory / f"{self.prefix}-{self._shard_index:05d}.jsonl"
        self._file = self._path.open("wb")
        self._digest = hashlib.sha256()
        self._shard_index += 1
        self._records_in_shard = 0

    def write(self, row: dict[str, Any]) -> None:
        if self._file is None or self._records_in_shard >= self.max_records:
            self._open_next()
        assert self._file is not None
        assert self._digest is not None
        payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        self._file.write(payload)
        self._digest.update(payload)
        self._records_in_shard += 1
        self.total_records += 1

    def close_shard(self) -> None:
        if self._file is None or self._path is None:
            return
        assert self._digest is not None
        self._file.close()
        self.files.append(
            {
                "path": str(self._path),
                "records": self._records_in_shard,
                "sha256": self._digest.hexdigest(),
            }
        )
        self._file = None
        self._path = None
        self._digest = None

    def close(self) -> None:
        self.close_shard()

    def __enter__(self) -> "JsonlShardWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class TokenJsonlShardWriter(JsonlShardWriter):
    def __init__(self, directory: Path, prefix: str, max_tokens: int) -> None:
        super().__init__(directory, prefix, max_records=2**63 - 1)
        self.max_tokens = max_tokens
        self._tokens_in_shard = 0
        self.total_tokens = 0

    def _open_next(self) -> None:
        super()._open_next()
        self._tokens_in_shard = 0

    def write(self, row: dict[str, Any]) -> None:
        token_count = int(row["token_count"])
        if self._file is None or (self._tokens_in_shard and self._tokens_in_shard + token_count > self.max_tokens):
            self._open_next()
        super().write(row)
        self._tokens_in_shard += token_count
        self.total_tokens += token_count

    def close_shard(self) -> None:
        if self._file is not None:
            token_count = self._tokens_in_shard
            super().close_shard()
            self.files[-1]["tokens"] = token_count

