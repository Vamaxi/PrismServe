"""Submit four simultaneous completion requests to an already-running live API."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    prompts = ["Artificial intelligence helps scientists", "The computer processes information by",
               "The village was quiet until", "Modern operating systems provide"]
    def complete(prompt):
        payload = {"model": "prismserve-gpt2-batched", "prompt": prompt, "max_tokens": 24}
        request = Request(args.url.rstrip("/") + "/v1/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        return {"prompt": prompt, "completion": result["choices"][0]["text"],
                "ttft_ms": round(result["prismserve"]["ttft_ms"], 2)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(complete, prompts):
            print(json.dumps(result))
    with urlopen(args.url.rstrip("/") + "/metrics", timeout=10) as response:
        metrics = json.load(response)
    print(json.dumps({key: metrics[key] for key in ("completed_requests", "max_execution_batch_size", "max_active")}, indent=2))


if __name__ == "__main__":
    main()
