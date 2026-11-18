"""Check likelihood target accounting and download-free asset behavior."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from prismserve.pretrained_assets import prepare_model, sha256_file
from prismserve.pretrained_experiments import evaluate_nll


class FixedLogits:
    context_limit = 16
    device = torch.device("cpu")

    def __call__(self, batch):
        logits = torch.arange(64, dtype=torch.float32) * .07
        return logits.expand(batch.shape[0], batch.shape[1], -1), None


class PretrainedExperimentTests(unittest.TestCase):
    def test_sliding_nll_scores_each_target_once(self):
        ids = list(range(53))
        result = evaluate_nll(FixedLogits(), ids, context=8, stride=3)
        expected = F.cross_entropy((torch.arange(64) * .07).expand(52, -1), torch.tensor(ids[1:]))
        self.assertEqual(result["scored_tokens"], 52)
        self.assertAlmostEqual(result["nll"], float(expected), places=5)
        self.assertAlmostEqual(result["perplexity"], math.exp(result["nll"]), places=5)

    def test_invalid_window_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_nll(FixedLogits(), [1, 2], context=8, stride=8)

    def test_missing_model_never_downloads_implicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with patch.dict("os.environ", {"HF_HUB_CACHE": str(path / "empty-cache")}), patch("prismserve.pretrained_assets.urlopen") as network:
                with self.assertRaises(FileNotFoundError):
                    prepare_model(path / "model", download=False)
                network.assert_not_called()

    def test_provenance_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "content"
            path.write_bytes(b"abc")
            self.assertEqual(sha256_file(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
