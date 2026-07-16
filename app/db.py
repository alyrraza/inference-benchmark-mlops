"""
PostgreSQL metadata store: logs every /predict request's outcome (backend,
batch size, cache hit/miss, latency, timestamp) for later analysis.

This is the "API -> DB : log latency, batch_size, model_type" step in
docs/sequence_diagram.puml, drawn after both the cache-hit and cache-miss
branches - every request gets logged here, regardless of which path
answered it.

Like app/cache.py's Redis calls, a database write failure here is caught
and logged, never allowed to turn into a failed /predict response - the
metadata store exists to observe the system, not to gate whether it
works. Unlike Redis (which is read from on the critical path and changes
what gets returned), Postgres here is purely a write-after-the-fact
side effect, so this degrades even more simply: log an error and move on.
"""

import asyncpg

from app import config

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS request_log (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    backend TEXT NOT NULL,
    cache_hit BOOLEAN NOT NULL,
    batch_size INTEGER,
    predicted_class_id INTEGER NOT NULL,
    total_latency_ms DOUBLE PRECISION NOT NULL
);
"""

# Every query this service runs filters or sorts by created_at (the
# verification script and any future Grafana dashboard both want "recent
# requests" or "requests in the last N minutes") - an index here is what
# keeps that fast as the table grows, instead of a full table scan on
# every query.
CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_request_log_created_at ON request_log (created_at DESC);
"""

INSERT_SQL = """
INSERT INTO request_log (backend, cache_hit, batch_size, predicted_class_id, total_latency_ms)
VALUES ($1, $2, $3, $4, $5);
"""


async def create_pool() -> asyncpg.Pool:
    """
    Creates the connection pool once, at app startup. See
    docs/concepts/04_postgres_metadata_store.md for what a connection
    pool is and why every production database client uses one instead of
    opening a fresh connection per request.
    """
    pool = await asyncpg.create_pool(
        host=config.DB_HOST,
        port=config.DB_PORT,
        database=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        min_size=config.DB_POOL_MIN_SIZE,
        max_size=config.DB_POOL_MAX_SIZE,
    )
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLE_SQL)
        await conn.execute(CREATE_INDEX_SQL)
    return pool


async def log_request(
    pool: asyncpg.Pool,
    backend: str,
    cache_hit: bool,
    batch_size: int | None,
    predicted_class_id: int,
    total_latency_ms: float,
) -> None:
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                INSERT_SQL, backend, cache_hit, batch_size, predicted_class_id, total_latency_ms
            )
    except Exception as exc:
        # A metadata-logging failure must never fail the actual prediction
        # the user is waiting on - print and move on, same philosophy as
        # app/cache.py's Redis error handling.
        print(f"[db] failed to log request, continuing: {exc}")
