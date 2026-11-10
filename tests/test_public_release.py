import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import audit_outputs

class PublicReleaseTests(unittest.TestCase):
    def test_audit_show_tolerates_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = dict(model="fixture", task="mmlu", lang="hi", script="native",
                       gold="A", pred="A", correct=True, gen="Answer: A")
            (Path(tmp) / "fixture__mmlu.jsonl").write_text(json.dumps(row) + '\n{"unfinished"')
            out = io.StringIO()
            with patch.object(audit_outputs, "EX", Path(tmp)), patch.object(sys, "argv", ["audit", "--show", "fixture"]), contextlib.redirect_stdout(out):
                audit_outputs.main()
            self.assertIn("Answer: A", out.getvalue())

    def test_model_root_override(self):
        env = dict(os.environ, MSB_MODELS_ROOT="/tmp/example-models", PYTHONPATH=str(ROOT / "src"))
        result = subprocess.run([sys.executable, "-c", "import common; print(common.MODELS['Llama-3.1-8B-Instruct']['path'])"],env=env,text=True,capture_output=True,check=True)
        self.assertEqual(result.stdout.strip(), "/tmp/example-models/Llama-3.1-8B-Instruct")

    def test_script_classifier_ignores_urls(self):
        import common
        self.assertEqual(common.classify_script("नमस्ते दुनिया https://example.org", "Devanagari"), common.classify_script("नमस्ते दुनिया", "Devanagari"))
