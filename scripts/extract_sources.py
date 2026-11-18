"""Optional PDF provenance extraction; original PDFs are never changed."""
from pathlib import Path
import hashlib
import json

import pymupdf


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "source_text"
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in root.glob("*.pdf"):
        with pymupdf.open(path) as doc:
            text = "\n".join(f"\n--- PAGE {i+1} ---\n" + page.get_text() for i, page in enumerate(doc))
            (output / (path.stem + ".txt")).write_text(text, encoding="utf-8")
            manifest.append({"file": path.name, "pages": len(doc), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
