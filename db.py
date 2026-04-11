import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    duration_minutes REAL NOT NULL DEFAULT 0,
    transcript TEXT NOT NULL DEFAULT '',
    insights JSONB NOT NULL DEFAULT '[]',
    method TEXT NOT NULL DEFAULT '',
    video_date TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def _get_conn():
    return psycopg2.connect(DATABASE_URL)


@contextmanager
def _cursor():
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    if not DATABASE_URL:
        logger.warning("No DATABASE_URL set — feed/history disabled")
        return
    try:
        with _cursor() as cur:
            cur.execute(SCHEMA)
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database init failed: {e}")


def save_episode(result: dict, insights: list) -> int | None:
    if not DATABASE_URL:
        return None
    try:
        with _cursor() as cur:
            cur.execute(
                """
                INSERT INTO episodes (url, title, source, duration_minutes, transcript, insights, method, video_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET
                    title = EXCLUDED.title,
                    source = EXCLUDED.source,
                    duration_minutes = EXCLUDED.duration_minutes,
                    transcript = EXCLUDED.transcript,
                    insights = EXCLUDED.insights,
                    method = EXCLUDED.method,
                    video_date = EXCLUDED.video_date
                RETURNING id
                """,
                (
                    result["url"],
                    result["title"],
                    result.get("source", ""),
                    result.get("duration_minutes", 0),
                    result.get("transcript", ""),
                    json.dumps(insights),
                    result.get("method", ""),
                    result.get("date", ""),
                ),
            )
            row = cur.fetchone()
            return row["id"] if row else None
    except Exception as e:
        logger.error(f"Failed to save episode: {e}")
        return None


def get_all_episodes() -> list[dict]:
    if not DATABASE_URL:
        return []
    try:
        with _cursor() as cur:
            cur.execute(
                """
                SELECT id, url, title, source, duration_minutes, insights, method, video_date, created_at
                FROM episodes
                ORDER BY created_at DESC
                """
            )
            rows = cur.fetchall()
            for row in rows:
                if isinstance(row["insights"], str):
                    row["insights"] = json.loads(row["insights"])
                row["created_at"] = row["created_at"].isoformat() if row["created_at"] else ""
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to fetch episodes: {e}")
        return []


def get_episode(episode_id: int) -> dict | None:
    if not DATABASE_URL:
        return None
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT * FROM episodes WHERE id = %s",
                (episode_id,),
            )
            row = cur.fetchone()
            if row:
                if isinstance(row["insights"], str):
                    row["insights"] = json.loads(row["insights"])
                row["created_at"] = row["created_at"].isoformat() if row["created_at"] else ""
                return dict(row)
            return None
    except Exception as e:
        logger.error(f"Failed to fetch episode {episode_id}: {e}")
        return None
