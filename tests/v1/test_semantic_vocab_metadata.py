from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.vocab.semantic_vocab import (
    SemanticVocabBuildConfig,
    SemanticVocabBuilder,
    SemanticVocabResult,
    add_single_hanzi_bpe_paths,
)


class FakeTokenizer:
    def encode(self, _text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [10, 11]


class FakeBpeState:
    @staticmethod
    def encode_text_piece(_text: str) -> str:
        return "abc"


class SemanticVocabMetadataTest(unittest.TestCase):
    def test_new_hanzi_ids_exclude_intermediate_bpe_nodes(self) -> None:
        new_vocab = {"a": 0, "b": 1, "c": 2}
        merges, init_ids, new_hanzi_ids, _added, _promoted = add_single_hanzi_bpe_paths(
            tokenizer=FakeTokenizer(),
            state=FakeBpeState(),
            old_vocab={"a": 0, "b": 1, "c": 2},
            new_vocab=new_vocab,
            new2old_token_id={0: 0, 1: 1, 2: 2},
            merges=[],
            hanzi_set={"中"},
        )

        self.assertEqual(merges, [("a", "b"), ("ab", "c")])
        self.assertEqual(set(init_ids), {3, 4})
        self.assertEqual(new_hanzi_ids, {4: "中"})

    def test_alignment_metadata_files_include_matching_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            builder = SemanticVocabBuilder(SemanticVocabBuildConfig(output_dir=output_dir))
            result = SemanticVocabResult(
                vocab={},
                merges=[],
                new2old_token_id={},
                new_token_init_token_ids={4: [10, 11]},
                removed_multi_hanzi_tokens=[{"token": "中国", "old_token_id": 7}],
                new_hanzi_token_ids={4: "中"},
                manifest={
                    "removed_multi_hanzi_token_count": 1,
                    "new_hanzi_token_count": 1,
                },
            )

            builder._write_alignment_metadata(result)

            removed_path = output_dir / "removed_multi_hanzi_tokens.json"
            new_hanzi_path = output_dir / "new_hanzi_token_ids.json"
            self.assertEqual(json.loads(removed_path.read_text(encoding="utf-8"))[0]["token"], "中国")
            self.assertEqual(json.loads(new_hanzi_path.read_text(encoding="utf-8")), {"4": "中"})
            self.assertEqual(
                result.manifest["removed_multi_hanzi_tokens_sha256"],
                hashlib.sha256(removed_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result.manifest["new_hanzi_token_ids_sha256"],
                hashlib.sha256(new_hanzi_path.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()