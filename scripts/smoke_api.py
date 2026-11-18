"""Launch a hidden local server, call its real HTTP API, and terminate our process."""
from pathlib import Path
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=root / "results/demo/tiny_checkpoint.pt")
    parser.add_argument("--pretrained", action="store_true", help="Test the actual local GPT-2 HTTP server")
    parser.add_argument("--model-path", type=Path, default=root / "models/gpt2")
    args = parser.parse_args()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    output = root / "results"
    output.mkdir(exist_ok=True)
    label = "pretrained_api_smoke" if args.pretrained else "api_smoke"
    log = (output / f"{label}_server.log").open("w", encoding="utf-8")
    command = ([sys.executable, "-m", "prismserve", "pretrained-serve", "--port", str(port), "--model-path", str(args.model_path)]
               if args.pretrained else
               [sys.executable, "-m", "prismserve", "serve", "--port", str(port), "--checkpoint", str(args.checkpoint)])
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, **kwargs)
    base = f"http://127.0.0.1:{port}"
    def get(path):
        with urlopen(base + path, timeout=5) as response:
            return json.load(response)
    try:
        deadline = time.monotonic() + 90
        while True:
            try:
                health = get("/health")
                break
            except (URLError, TimeoutError, ConnectionError):
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"server did not start; see results/{label}_server.log")
                time.sleep(.2)
        payload = {"model": "prismserve-gpt2" if args.pretrained else "prismserve-tiny",
                   "prompt": "Artificial intelligence can help scientists" if args.pretrained else "The serving system", "max_tokens": 8,
                   "quantized_layers": [0], "cpu_attention": True}
        request = Request(base + "/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            completion = json.load(response)
        metrics = get("/metrics")
        assert completion["usage"]["completion_tokens"] == 8
        assert metrics["completed_requests"] == 1
        result = {"successful": True, "health": health, "completion": completion, "metrics": metrics}
        (output / f"{label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
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
