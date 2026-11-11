(
    (
        "Text-only OpenAI helpers with local "
        "de-identification and a disk "
        "cache.\n"
        "\n"
        "Identical sanitized requests and run "
        "numbers reuse completed answers. New "
        "prompts,\n"
        "new runs and use_cache=False can incur "
        "charges. Never send raw data via a "
        "separate\n"
        "OpenAI client: use these helpers, "
        "including ask_decision() for fixed "
        "choices.\n"
    )
)

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.decision import Decision
from openai.types.responses import Response
from tqdm.auto import tqdm

from .privacy import sanitize_request

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE_DIR = ROOT / ".cache" / "llm-v2"
load_dotenv(ROOT / ".env", override=False)
_CLIENT = None
_client_lock = Lock()


def get_client() -> OpenAI:
    """Create or reuse the shared API client under a lock."""
    # A shared lazy client avoids constructing clients on cache hits.
    global _CLIENT  # pylint: disable=global-statement
    with _client_lock:
        if _CLIENT is None:
            _CLIENT = OpenAI(max_retries=8)
        return _CLIENT


def _complete(response):
    if isinstance(response, Response):
        return response.status == "completed" and not any(
            getattr(part, "type", "") == "refusal"
            for item in response.output
            for part in getattr(item, "content", [])
        )
    return not any(answer.type == "refusal" for answer in response.answers)


def _request(endpoint, response_type, run, use_cache, kwargs):
    kwargs = sanitize_request(
        kwargs
    )  # Must finish before cache or client access.
    if endpoint == "responses":
        kwargs["store"] = False
    key = json.dumps(
        [endpoint, kwargs, run], sort_keys=True, ensure_ascii=False
    )
    path = CACHE_DIR / (
        hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json"
    )
    if not use_cache:
        return getattr(get_client(), endpoint).create(**kwargs)

    # flock coordinates identical requests across notebook kernels as well as
    # threads. Lock files contain no request text and are released on process
    # exit.
    # Import only on calls using the Unix cache lock.
    import fcntl  # pylint: disable=import-outside-toplevel

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            try:
                response = response_type.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                if _complete(response):
                    return response
            except (ValueError, OSError):
                warnings.warn(
                    (
                        (
                            "Invalid cached answer; requesting a "
                            "replacement (may incur "
                            "charges)."
                        )
                    ),
                    stacklevel=2,
                )
        response = getattr(get_client(), endpoint).create(**kwargs)
        if _complete(response):
            # Readers see either the old complete file or the new complete file.
            fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as target:
                    target.write(response.model_dump_json(by_alias=True))
                os.replace(name, path)
            finally:
                Path(name).unlink(missing_ok=True)
        return response


def ask(run: int = 0, use_cache: bool = True, **kwargs) -> Response:
    """Redact locally, call Responses, cache completed non-refused answers.

    run=1, 2, ... requests independent answers; reusing a run reuses its cache.
    use_cache=False always sends a new request. Linux/macOS file locks required.
    """
    return _request("responses", Response, run, use_cache, kwargs)


def ask_decision(run: int = 0, use_cache: bool = True, **kwargs) -> Decision:
    """Fixed choices with the same local redaction and cache as ask()."""
    return _request("decisions", Decision, run, use_cache, kwargs)


def ask_many(inputs: list, max_workers: int = 8, **kwargs) -> list[Response]:
    (
        (
            "Call ask concurrently, preserving input "
            "order; cache hits need no API "
            "key."
        )
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(
            tqdm(
                pool.map(lambda x: ask(input=x, **kwargs), inputs),
                total=len(inputs),
            )
        )
