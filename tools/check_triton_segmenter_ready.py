from __future__ import annotations

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _fetch_json(url: str) -> dict:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    base_url = argv[0] if argv else "http://127.0.0.1:8010"
    base_url = base_url.rstrip("/")
    model_name = argv[1] if len(argv) > 1 else "shadowgen_segmenter"
    ready_url = f"{base_url}/v2/models/{model_name}/ready"
    config_url = f"{base_url}/v2/models/{model_name}/config"

    try:
        with urlopen(Request(ready_url), timeout=10) as response:
            if response.status != 200:
                print(f"segmenter model is not ready: HTTP {response.status}", file=sys.stderr)
                return 1
    except HTTPError as exc:
        print(f"segmenter model readiness check failed: HTTP {exc.code}", file=sys.stderr)
        return 1
    except (ConnectionAbortedError, ConnectionResetError, TimeoutError, URLError) as exc:
        print(f"failed to reach Triton at {base_url}: {exc}", file=sys.stderr)
        return 1

    try:
        config = _fetch_json(config_url)
    except Exception as exc:  # noqa: BLE001
        print(f"segmenter model config fetch failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"ready": True, "model_name": model_name, "config": config.get("config", config)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
