"""Package the completed project and evidence without altering source documents."""
from pathlib import Path
import hashlib
import json
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "deliverables"
    output.mkdir(exist_ok=True)
    selected = []
    for directory in ["prismserve", "configs", "tests", "scripts", "docs", "data", "results", "artifacts/source_text", "deliverables"]:
        for path in (root / directory).rglob("*"):
            if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
                continue
            if (path.suffix == ".pyc" or (path.suffix == ".zip" and "deliverables" in path.parts)
                    or path.name in {"package_manifest.json", "demo_stdout.json"}):
                continue
            selected.append(path)
    selected.extend(root / name for name in ["README.md", "pyproject.toml", "references.bib", ".gitignore"])
    model_manifest = root / "models/gpt2/prismserve_source.json"
    if model_manifest.exists():
        selected.append(model_manifest)
    selected = sorted(set(selected))
    manifest = {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in selected}
    manifest_path = output / "package_manifest.json"
    manifest_path.write_text(json.dumps({"files": manifest, "original_pdfs": "Provided separately by the user; not duplicated in this archive."}, indent=2), encoding="utf-8")
    target = output / "PrismServe_Project.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in selected + [manifest_path]:
            archive.write(path, "PrismServe/" + str(path.relative_to(root)).replace("\\", "/"))
    print(f"Packaged {len(selected)+1} files: {target} ({target.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
