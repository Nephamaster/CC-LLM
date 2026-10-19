"""Environment upgrades must not block encoding via Git or source fingerprints."""

import importlib.metadata
import sys
import types
import unittest
from unittest.mock import Mock, patch

from scripts.data_factory.ancient_sft.encoding import SwiftEncoder


class EncoderEnvironmentTest(unittest.TestCase):
    def encoder(self, metadata_missing=False):
        swift = types.ModuleType("swift")
        swift.__version__ = "new-installed-version"
        swift.__file__ = "/environment/site-packages/swift/__init__.py"
        model = types.ModuleType("swift.model")
        template_module = types.ModuleType("swift.template")
        model.get_model_processor = Mock(return_value=(None, object()))
        template = Mock()
        template.encode.return_value = {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}
        template_module.get_template = Mock(return_value=template)
        version = Mock(return_value="current-version")
        if metadata_missing:
            version.side_effect = importlib.metadata.PackageNotFoundError
        with patch.dict(sys.modules, {"swift": swift, "swift.model": model, "swift.template": template_module}), \
                patch("importlib.metadata.version", version), \
                patch("subprocess.run", side_effect=AssertionError("Git must not be invoked")), \
                patch("pathlib.Path.rglob", side_effect=AssertionError("Sources must not be fingerprinted")):
            encoder = SwiftEncoder({"tokenizer_path": "/local/model", "max_length": 2048,
                                    "swift_commit": "obsolete-pin"})
        return encoder, model, template_module, template

    def test_installed_or_modified_swift_is_used_without_version_gate(self):
        encoder, model, templates, template = self.encoder()
        self.assertEqual(encoder.info["package_version"], "new-installed-version")
        self.assertNotIn("template_source_sha256", encoder.info)
        self.assertFalse(model.get_model_processor.call_args.kwargs["load_model"])
        self.assertFalse(templates.get_template.call_args.kwargs["enable_thinking"])
        template.set_mode.assert_called_once_with("train")
        self.assertEqual(encoder([{"role": "user", "content": "test"}])["assistant_loss_tokens"], 2)

    def test_missing_package_metadata_does_not_block_encoding(self):
        encoder, _, _, _ = self.encoder(metadata_missing=True)
        self.assertEqual(encoder.info["dependencies"], {"transformers": None, "tokenizers": None, "torch": None})

    def test_actual_label_errors_are_still_rejected(self):
        encoder, _, _, template = self.encoder()
        template.encode.return_value = {"input_ids": [1, 2], "labels": [-100, -100]}
        with self.assertRaisesRegex(RuntimeError, "missing answer loss"):
            encoder([])


if __name__ == "__main__":
    unittest.main()
