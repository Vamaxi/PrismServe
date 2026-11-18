"""Run the complete suite and save a machine-readable validation record."""
from pathlib import Path
import io
import json
import platform
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    (output / "validation.txt").write_text(stream.getvalue(), encoding="utf-8")
    record = {"successful": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "elapsed_s": time.perf_counter() - started,
              "python": sys.version, "platform": platform.platform()}
    (output / "validation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
