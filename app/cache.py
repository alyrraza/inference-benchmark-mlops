"""
Redis-backed response cache for /predict.

WHY THIS EXISTS:
Inference is deterministic - the same image bytes run through the same
backend always produce the same prediction. Re-running the full batching
+ model forward pass for a request this service has already answered
wastes CPU for zero benefit. This module checks Redis before a request
ever reaches the queue, and stores the result after a real inference run,
so an identical future request gets served straight out of memory instead
of waiting on a model.

Caching is a performance optimization, not a correctness requirement: if
Redis is unreachable, every function here degrades to "treat this as a
cache miss" rather than raising and taking the whole service down with it.
"""

import hashlib
import json

import redis

from app import config

_client: redis.Redis | None = None


def get_client() -> redis.Redis:
    """
    Returns a shared, lazily-connected Redis client. redis-py doesn't
    actually open a socket here - the real connection attempt happens on
    the first command (get/set/ping), which is why every caller below
    wraps its Redis call in a try/except rather than checking a
    connection state up front.
    """
    global _client
    if _client is None:
        _client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True,  # get back str instead of bytes - one less .decode() everywhere
            socket_connect_timeout=config.REDIS_CONNECT_TIMEOUT_SECONDS,
            socket_timeout=config.REDIS_CONNECT_TIMEOUT_SECONDS,
            # Recent redis-py versions default to negotiating RESP3 with a
            # HELLO command on connect. This dev machine's local Redis
            # instance is a very old build (3.0.504, from 2016, the only
            # one readily available for Windows without admin rights - see
            # config.py's REDIS_PORT comment) that predates RESP3 entirely
            # and errors with "unknown command 'HELLO'" on every connection
            # attempt. Forcing protocol=2 (RESP2) skips that handshake and
            # works against both old and modern Redis servers - CI's
            # redis:7-alpine service container doesn't need this, but it's
            # harmless there too.
            protocol=2,
        )
    return _client


def is_redis_available() -> bool:
    try:
        return get_client().ping()
    except redis.exceptions.RedisError:
        return False


def build_cache_key(image_bytes: bytes, backend: str) -> str:
    """
    One cache key = one exact image, run through one specific backend.

    Hashing the raw uploaded bytes (not the preprocessed tensor) means two
    requests only share a cache entry if the file uploaded was genuinely
    byte-for-byte identical - this is deliberately strict "identical
    request" matching, not a fuzzy "looks like the same image" match.

    The backend name is part of the key, not an afterthought: Phase 1's
    accuracy validation showed different backends can produce tiny
    numerical differences (TensorRT's FP16 rounding had a small nonzero
    mean absolute difference from PyTorch, even though on this project's
    CPU backends the match was numerically exact). A pytorch result must
    never be served back for an onnx request, even if today, on this
    model, the two would agree - the cache key shouldn't rely on that
    staying true.
    """
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    return f"inferbench:predict:{backend}:{image_hash}"


def get_cached_prediction(image_bytes: bytes, backend: str) -> dict | None:
    """Returns the cached {predicted_class_id, predicted_label} dict, or None on a miss (or if Redis is down)."""
    key = build_cache_key(image_bytes, backend)
    try:
        cached = get_client().get(key)
    except redis.exceptions.RedisError as exc:
        print(f"[cache] Redis unavailable on read, treating as cache miss: {exc}")
        return None
    if cached is None:
        return None
    return json.loads(cached)


def store_prediction(image_bytes: bytes, backend: str, result: dict) -> None:
    """Stores a fresh prediction result with the configured TTL. Silently no-ops if Redis is unreachable."""
    key = build_cache_key(image_bytes, backend)
    try:
        get_client().set(key, json.dumps(result), ex=config.CACHE_TTL_SECONDS)
    except redis.exceptions.RedisError as exc:
        print(f"[cache] Redis unavailable on write, skipping cache store: {exc}")
