"""
utils/cache_utils.py — In-process brief cache with customer-scoped invalidation.

Uses a simple dict cache with TTL. In production, swap _CACHE for Redis
by implementing the same interface against redis-py. The API surface is
intentionally minimal so the swap is a one-file change.
"""
import os
import time
import logging
import threading
from typing import Any, Dict, Optional

_log = logging.getLogger(__name__)

_CACHE: Dict[str, tuple] = {}   # key → (data, timestamp)
_CACHE_LOCK = threading.Lock()
_TTL_SECONDS = int(os.getenv("BRIEF_CACHE_TTL_MINUTES", "30")) * 60


def _key(customer_id: str, as_of_date: str, brief_type: str) -> str:
    return f"{customer_id}::{as_of_date}::{brief_type}"


def get_cached(customer_id: str, as_of_date: str,
               brief_type: str = "pre_meeting") -> Optional[Dict[str, Any]]:
    """Return cached brief dict if still within TTL, else None."""
    k = _key(customer_id, as_of_date, brief_type)
    with _CACHE_LOCK:
        entry = _CACHE.get(k)
        if entry is None:
            return None
        data, ts = entry
        if time.time() - ts > _TTL_SECONDS:
            del _CACHE[k]
            _log.debug("cache_expired key=%s", k)
            return None
    _log.debug("cache_hit key=%s age_s=%.0f", k, time.time() - ts)
    return data


def set_cached(customer_id: str, as_of_date: str,
               brief_type: str, data: Dict[str, Any]) -> None:
    """Store a brief in the cache."""
    k = _key(customer_id, as_of_date, brief_type)
    with _CACHE_LOCK:
        _CACHE[k] = (data, time.time())
    _log.debug("cache_set key=%s ttl_s=%d", k, _TTL_SECONDS)


def invalidate_customer(customer_id: str) -> int:
    """
    Remove all cached entries for a customer.
    Called when a new document is uploaded for that customer.
    Returns the number of entries removed.
    """
    prefix = f"{customer_id}::"
    with _CACHE_LOCK:
        keys_to_remove = [k for k in list(_CACHE) if k.startswith(prefix)]
        for k in keys_to_remove:
            del _CACHE[k]
    if keys_to_remove:
        _log.info("cache_invalidated customer=%s entries=%d",
                  customer_id, len(keys_to_remove))
    return len(keys_to_remove)


def clear_all() -> int:
    """Clear entire cache. Used in tests."""
    with _CACHE_LOCK:
        count = len(_CACHE)
        _CACHE.clear()
    return count


def get_stats() -> Dict[str, Any]:
    """Return cache statistics for /health endpoint."""
    now = time.time()
    with _CACHE_LOCK:
        snapshot = list(_CACHE.values())
    alive = sum(1 for (_, ts) in snapshot if now - ts <= _TTL_SECONDS)
    expired = len(snapshot) - alive
    return {
        "total_entries": len(snapshot),
        "alive": alive,
        "expired": expired,
        "ttl_seconds": _TTL_SECONDS,
    }
