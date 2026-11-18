"""Pinned, explicit asset preparation; inference never downloads models implicitly."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.request import urlopen

GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MODEL_FILES = ("config.json", "generation_config.json", "merges.txt", "model.safetensors",
               "tokenizer.json", "tokenizer_config.json", "vocab.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_model(output: Path = Path("models/gpt2"), download: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cache_root = Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")))
    candidates = [cache_root / repo / "snapshots" / GPT2_REVISION
                  for repo in ("models--gpt2", "models--openai-community--gpt2")]
    source = next((p for p in candidates if all((p / name).is_file() for name in MODEL_FILES)), None)
    for name in MODEL_FILES:
        destination = output / name
        if destination.is_file():
            continue
        if source is not None:
            shutil.copyfile(source / name, destination)
        elif download:
            url = f"https://huggingface.co/gpt2/resolve/{GPT2_REVISION}/{name}"
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, 1024 * 1024)
            temporary.replace(destination)
        else:
            raise FileNotFoundError("GPT-2 weights are not local. Run pretrained-prepare --download to fetch the pinned public model.")
    result = {"model_id": "gpt2", "canonical_model_id": "openai-community/gpt2", "revision": GPT2_REVISION,
              "source": "existing Hugging Face cache or verified local files" if source else "explicit pinned public download or existing local files",
              "model_card": "https://huggingface.co/openai-community/gpt2", "license": "MIT (model card)",
              "files": {name: {"bytes": (output / name).stat().st_size,
                                "sha256": sha256_file(output / name)} for name in MODEL_FILES}}
    (output / "prismserve_source.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"output": str(output.resolve()), "model_id": "gpt2", "revision": GPT2_REVISION,
            "bytes": sum(row["bytes"] for row in result["files"].values())}


def prepare_wikitext(output: Path = Path("data/wikitext2")) -> dict:
    """Fetch a bounded official dataset-viewer prefix for validation and test.

    The authors' legacy S3 archive is unavailable at its old URL. The viewer
    supplies JSON without requiring a large Parquet dependency. Raw payloads,
    row selection, transformations and hashes are recorded for reproducibility.
    """
    output.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in ("validation", "test"):
        url = ("https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext"
               f"&config=wikitext-2-raw-v1&split={split}&offset=0&length=100")
        payload_path = output / f"{split}.rows.json"
        if not payload_path.exists():
            with urlopen(url, timeout=30) as response:
                payload = response.read(4 * 1024 * 1024 + 1)
            if len(payload) > 4 * 1024 * 1024:
                raise ValueError("unexpected dataset response size")
            parsed = json.loads(payload)
            if "rows" not in parsed or not parsed["rows"]:
                raise ValueError("dataset viewer response has no rows")
            payload_path.write_bytes(payload)
        parsed = json.loads(payload_path.read_bytes())
        texts = [row["row"]["text"] for row in parsed["rows"]]
        target = output / f"wiki.{split}.prefix.txt"
        target.write_text("\n\n".join(texts), encoding="utf-8", newline="")
        splits[split] = {"file": target.name, "rows": len(texts), "offset": 0, "url": url,
                         "text_sha256": sha256_file(target), "payload_sha256": sha256_file(payload_path),
                         "bytes": target.stat().st_size}
    result = {"dataset": "WikiText-2", "configuration": "wikitext-2-raw-v1",
              "selection": "First 100 rows of each split, joined with two newlines; benchmark further limits tokens.",
              "dataset_card": "https://huggingface.co/datasets/Salesforce/wikitext",
              "license": "CC BY-SA 3.0 / GFDL; Wikipedia contributors; see dataset card",
              "source_revision": "Dataset viewer current revision; original response payloads retained and hashed.",
              "splits": splits}
    (output / "source.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "ATTRIBUTION.md").write_text(
        "# WikiText-2 attribution\n\nSource: https://huggingface.co/datasets/Salesforce/wikitext\n\n"
        "Dataset by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher, derived from Wikipedia contributors.\n\n"
        "Dataset card: https://huggingface.co/datasets/Salesforce/wikitext\n\n"
        "Licenses: https://creativecommons.org/licenses/by-sa/3.0/ and https://www.gnu.org/licenses/fdl-1.3.html\n\n"
        "First 100 rows of validation/test joined with two newlines. Original response JSON retained. "
        "Validation is used for controller calibration; test only for evaluation. No model training is performed.\n",
        encoding="utf-8")
    return result
