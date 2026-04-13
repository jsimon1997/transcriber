"""
Storage backend using a JSON file in the GitHub repo.
No database needed — free forever.
"""

import base64
import json
import logging
import os
import threading
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "jsimon1997/transcriber")
FEED_FILE = "feed.json"

_lock = threading.Lock()


def _headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }


def _api_url():
    return f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FEED_FILE}"


def _read_feed() -> tuple[list[dict], str]:
    """Read feed.json from GitHub. Returns (episodes, sha)."""
    if not GITHUB_TOKEN:
        logger.warning("No GITHUB_TOKEN — feed disabled")
        return [], ""
    try:
        resp = requests.get(_api_url(), headers=_headers(), timeout=15)
        if resp.status_code == 404:
            return [], ""
        resp.raise_for_status()
        data = resp.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        episodes = json.loads(content) if content.strip() else []
        return episodes, data["sha"]
    except Exception as e:
        logger.error(f"Failed to read feed from GitHub: {e}")
        return [], ""


def _write_feed(episodes: list[dict], sha: str) -> bool:
    """Write feed.json to GitHub."""
    if not GITHUB_TOKEN:
        return False
    try:
        content = json.dumps(episodes, indent=2, ensure_ascii=False)
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        payload = {
            "message": "Update feed",
            "content": encoded,
        }
        if sha:
            payload["sha"] = sha
        resp = requests.put(_api_url(), headers=_headers(), json=payload, timeout=15)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to write feed to GitHub: {e}")
        return False


def init_db():
    """Ensure feed.json exists in the repo."""
    if not GITHUB_TOKEN:
        logger.warning("No GITHUB_TOKEN set — feed/history disabled")
        return
    episodes, sha = _read_feed()
    if not sha:
        # Create the file
        logger.info("Creating feed.json in GitHub repo")
        _write_feed([], "")
    else:
        logger.info(f"Feed has {len(episodes)} episodes")


def save_episode(result: dict, insights: list) -> int | None:
    if not GITHUB_TOKEN:
        return None
    with _lock:
        episodes, sha = _read_feed()

        # Check if URL already exists — update it
        existing_idx = None
        for i, ep in enumerate(episodes):
            if ep.get("url") == result["url"]:
                existing_idx = i
                break

        episode = {
            "id": existing_idx + 1 if existing_idx is not None else len(episodes) + 1,
            "url": result["url"],
            "title": result["title"],
            "source": result.get("source", ""),
            "duration_minutes": result.get("duration_minutes", 0),
            "transcript": result.get("transcript", ""),
            "insights": insights,
            "method": result.get("method", ""),
            "video_date": result.get("date", ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if existing_idx is not None:
            episode["created_at"] = episodes[existing_idx].get("created_at", episode["created_at"])
            episodes[existing_idx] = episode
        else:
            episodes.append(episode)

        if _write_feed(episodes, sha):
            logger.info(f"Saved episode: {result['title']}")
            return episode["id"]
        return None


def get_all_episodes() -> list[dict]:
    if not GITHUB_TOKEN:
        return []
    episodes, _ = _read_feed()
    # Return without transcripts (too large for feed listing)
    feed = []
    for ep in episodes:
        item = {k: v for k, v in ep.items() if k != "transcript"}
        feed.append(item)
    # Sort by video release date descending (latest first), fall back to created_at
    feed.sort(key=lambda x: x.get("video_date", "") or x.get("created_at", ""), reverse=True)
    return feed


def get_episode(episode_id: int) -> dict | None:
    if not GITHUB_TOKEN:
        return None
    episodes, _ = _read_feed()
    for ep in episodes:
        if ep.get("id") == episode_id:
            return ep
    return None


def delete_episode(episode_id: int) -> bool:
    if not GITHUB_TOKEN:
        return False
    with _lock:
        episodes, sha = _read_feed()
        new_eps = [ep for ep in episodes if ep.get("id") != episode_id]
        if len(new_eps) == len(episodes):
            return False
        _write_feed(new_eps, sha)
        return True
