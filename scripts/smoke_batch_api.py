"""Verify overlapping real HTTP requests reach a shared actual model batch."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import argparse
import os
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / "results/controller")
    parser.add_argument("--calibration", default="results/live/calibration.json")
    args = parser.parse_args()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    target = args.output
    target.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", "prismserve", "live-serve", "--port", str(port),
               "--device-budget-mib", "1400", "--calibration", args.calibration]
    extra = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    log = (target / "http_smoke_server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, **extra)
    base = f"http://127.0.0.1:{port}"
    def get(path):
        with urlopen(base + path, timeout=10) as response:
            return json.load(response)
    try:
        deadline = time.monotonic() + 180
        while True:
            try:
                health = get("/health")
                break
            except (URLError, TimeoutError, ConnectionError):
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"batch server failed to start; see {target / 'http_smoke_server.log'}")
                time.sleep(.2)
        gate = threading.Barrier(4)
        prompts = ["Artificial intelligence helps scientists", "The computer processes information by",
                   "The village was quiet until", "Modern operating systems provide"]
        def complete(prompt):
            gate.wait(timeout=10)
            request = Request(base + "/v1/completions", data=json.dumps({"model": "prismserve-gpt2-batched", "prompt": prompt,
                              "max_tokens": 12}).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(complete, prompts))
        metrics = get("/metrics")
        assert all(row["usage"]["completion_tokens"] > 0 for row in responses)
        assert metrics["completed_requests"] == 4
        assert metrics["max_execution_batch_size"] > 1
        result = {"passed": True, "real_concurrent_http_clients": 4, "health": health,
                  "metrics": metrics, "responses": responses}
        (target / "http_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"passed": True, "completed": metrics["completed_requests"],
                          "max_actual_execution_batch_size": metrics["max_execution_batch_size"]}, indent=2))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log.close()


if __name__ == "__main__":
    main()
