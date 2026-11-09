"""Two small helpers for calling the OpenAI API in class.

ask()      works exactly like client.responses.create(...), but saves every
           answer on disk: running a cell again costs nothing.
ask_many() calls ask() for many inputs at the same time.

All of us share one API key. The OpenAI client waits and retries
automatically when the key hits a rate limit (HTTP 429).
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import Response
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache" / "llm"

# Outside Codespaces, the key can come from a local .env file
load_dotenv(ROOT / ".env", override=False)

# One client for all calls, created on the first call.
# It retries up to 8 times on rate limits.
_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(max_retries=8)
    return _client


def ask(run: int = 0, use_cache: bool = True, **kwargs) -> Response:
    """Call client.responses.create(**kwargs) and save the answer on disk.

    run:       use run=1, 2, ... to get a new answer to the same request
               (e.g. to measure consistency). The same run returns the saved answer.
    use_cache: False always calls the API (e.g. to measure speed).
    """
    key = json.dumps([kwargs, run], sort_keys=True, ensure_ascii=False)
    path = CACHE_DIR / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json")

    if use_cache and path.exists():
        return Response.model_validate_json(path.read_text(encoding="utf-8"))

    response = get_client().responses.create(**kwargs)

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        # by_alias: keep the field names of the API (e.g. 'schema', not 'schema_')
        path.write_text(response.model_dump_json(by_alias=True), encoding="utf-8")
    return response


def ask_many(inputs: list, max_workers: int = 8, **kwargs) -> list[Response]:
    """ask(input=x, **kwargs) for every x in inputs, 8 calls at a time.

    The answers come back in the same order as the inputs.
    """
    get_client()  # create the client once, before the threads start
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(ask, input=x, **kwargs) for x in inputs]
        return [f.result() for f in tqdm(futures)]
