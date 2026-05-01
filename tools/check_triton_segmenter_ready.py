from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _fetch_json(url: str) -> dict:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Triton readiness for the ShadowGen segmenter model.")
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8010")
    parser.add_argument("model_name", nargs="?", default="shadowgen_segmenter")
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    args = parser.parse_args(argv or sys.argv[1:])

    base_url = args.base_url.rstrip("/")
    model_name = args.model_name
    ready_url = f"{base_url}/v2/models/{model_name}/ready"
    config_url = f"{base_url}/v2/models/{model_name}/config"

    deadline = time.monotonic() + max(args.wait_seconds, 0.0)
    last_error = ""
    while True:
        try:
            with urlopen(Request(ready_url), timeout=10) as response:
                if response.status == 200:
                    break
                last_error = f"segmenter model is not ready: HTTP {response.status}"
        except HTTPError as exc:
            last_error = f"segmenter model readiness check failed: HTTP {exc.code}"
        except (ConnectionAbortedError, ConnectionResetError, TimeoutError, URLError) as exc:
            last_error = f"failed to reach Triton at {base_url}: {exc}"
        if time.monotonic() >= deadline:
            print(last_error, file=sys.stderr)
            return 1
        time.sleep(max(args.interval_seconds, 0.1))

    try:
        config = _fetch_json(config_url)
    except Exception as exc:  # noqa: BLE001
        print(f"segmenter model config fetch failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"ready": True, "model_name": model_name, "config": config.get("config", config)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
