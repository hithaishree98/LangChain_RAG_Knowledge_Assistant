"""Shared API lifecycle helpers used by all experiment scripts."""
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]


def _wait_for_api(port: int = 8000, timeout: int = 420) -> bool:
    """Poll /health until the API reports ready.

    Readiness includes pre-loading the embedding model (OpenAI text-embedding-3-small
    or MiniLM-L6-v2) and the cross-encoder reranker during lifespan startup. On a
    cold disk cache that can take 3-5 minutes; 7 min gives generous headroom.
    Previously the model loaded lazily on first upload, hiding the cold-start cost
    inside request timeouts.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_api(env_overrides: dict, port: int = 8000) -> subprocess.Popen:
    print(f"  [api] starting with {env_overrides}")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "0.0.0.0", "--port", str(port)],
        cwd=str(REPO_ROOT / "api"),
        env={**os.environ, **env_overrides},
    )
    if not _wait_for_api(port):
        proc.terminate()
        raise RuntimeError("API did not become healthy within 420s (model warmup)")
    print("  [api] ready")
    return proc


def stop_api(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("  [api] stopped")


def assert_workspace_ready(user_id: str, expected_min_docs: int = 6,
                           port: int = 8000) -> None:
    """Fail fast if a prior-experiment workspace is missing or under-populated.

    Used by exp2/exp3 which reuse eval_full instead of re-uploading. If exp1
    crashed mid-upload, the index could be partial and silently skew results.
    """
    api_key = os.getenv("API_KEY", "")
    token = os.getenv("EVAL_TOKEN", "")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        r = requests.get(f"http://localhost:{port}/documents",
                         headers=headers, timeout=10)
    except Exception as e:
        raise RuntimeError(f"workspace check failed for {user_id}: {e}")
    if r.status_code != 200:
        raise RuntimeError(f"/documents {r.status_code} for {user_id}: {r.text[:200]}")
    docs = r.json() if r.text else []
    if len(docs) < expected_min_docs:
        raise RuntimeError(
            f"workspace {user_id} has only {len(docs)} docs "
            f"(expected >= {expected_min_docs}). Re-run exp1 before this experiment."
        )
    print(f"  [workspace] {user_id}: {len(docs)} docs OK")
