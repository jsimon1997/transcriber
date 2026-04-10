import json
import logging
import os
import re
from typing import Optional
from dataclasses import dataclass

import requests
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

logger = logging.getLogger(__name__)

SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY", "")
_PROXIES = (
    {"http": f"http://scraperapi:{SCRAPER_API_KEY}@proxy-server.scraperapi.com:8001",
     "https": f"http://scraperapi:{SCRAPER_API_KEY}@proxy-server.scraperapi.com:8001"}
    if SCRAPER_API_KEY else None
)


@dataclass
class TranscriptResult:
    title: str
    source: str
    date: str
    duration_minutes: float
    url: str
    segments: list
    method: str


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_transcript(result: TranscriptResult) -> str:
    lines = [
        f"Title: {result.title}",
        f"Source: {result.source}",
        f"Date: {result.date}",
        f"Duration: {result.duration_minutes:.0f} minutes",
        f"URL: {result.url}",
        "",
        "---",
        "",
    ]
    for seg in result.segments:
        ts = _fmt_ts(seg["start"])
        lines.append(f"[{ts}] {seg['text'].strip()}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------

def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def is_spotify_url(url: str) -> bool:
    return "open.spotify.com/episode" in url


def extract_youtube_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
        r"youtube\.com/shorts/([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

def _get_metadata(video_id: str) -> dict:
    """Fetch title and channel via YouTube oEmbed — no API key, no bot detection."""
    try:
        resp = requests.get(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return {"title": data.get("title", "Unknown Title"), "channel": data.get("author_name", "Unknown Channel")}
    except Exception as e:
        logger.warning(f"oEmbed fetch failed: {e}")
        return {"title": "Unknown Title", "channel": "Unknown Channel"}


def transcribe_youtube(url: str) -> TranscriptResult:
    video_id = extract_youtube_id(url)
    if not video_id:
        raise ValueError(f"Could not extract a YouTube video ID from: {url}")

    logger.info(f"YouTube video ID: {video_id}")

    logger.info("Fetching video metadata...")
    meta = _get_metadata(video_id)
    title = meta["title"]
    channel = meta["channel"]
    date = ""
    duration_min = 0

    logger.info("Fetching captions...")
    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id, proxies=_PROXIES)
        logger.info("Captions found.")
        segments = [{"start": e["start"], "text": e["text"]} for e in entries]
        return TranscriptResult(
            title=title, source=channel, date=date,
            duration_minutes=duration_min, url=url,
            segments=segments, method="captions",
        )
    except (TranscriptsDisabled, NoTranscriptFound):
        raise RuntimeError(
            "No captions are available for this video. "
            "Only videos with existing captions (auto-generated or manual) are supported."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to fetch captions: {e}")


# ---------------------------------------------------------------------------
# Spotify
# ---------------------------------------------------------------------------

def _fetch_spotify_meta(url: str) -> dict:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        html = resp.text

        def og(prop):
            m = re.search(rf'<meta property="og:{prop}" content="([^"]+)"', html)
            return m.group(1) if m else ""

        title = og("title")
        show_name = og("site_name") or "Spotify Podcast"

        ld_match = re.search(
            r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL
        )
        if ld_match:
            try:
                data = json.loads(ld_match.group(1))
                title = data.get("name", title)
                if "partOfSeries" in data:
                    show_name = data["partOfSeries"].get("name", show_name)
            except Exception:
                pass

        return {"title": title, "show_name": show_name}
    except Exception as e:
        logger.warning(f"Spotify page fetch failed: {e}")
        return {"title": "", "show_name": ""}


def _search_youtube(query: str) -> Optional[str]:
    """Search YouTube via scraping — no API key needed."""
    logger.info(f"Searching YouTube: {query!r}")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        resp = requests.get(
            "https://www.youtube.com/results",
            params={"search_query": query},
            headers=headers,
            timeout=10,
        )
        # Extract video IDs from the response
        ids = re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', resp.text)
        if ids:
            return f"https://www.youtube.com/watch?v={ids[0]}"
    except Exception as e:
        logger.warning(f"YouTube search error: {e}")
    return None


def transcribe_spotify(url: str) -> TranscriptResult:
    logger.info(f"Processing Spotify URL: {url}")

    logger.info("Fetching Spotify episode metadata...")
    meta = _fetch_spotify_meta(url)
    title = meta["title"]
    show_name = meta["show_name"]

    if not title:
        raise ValueError(
            "Could not read episode info from Spotify. "
            "The page may require a logged-in session."
        )

    logger.info(f"Episode: '{title}' | Show: '{show_name}'")

    yt_url = _search_youtube(f"{show_name} {title}")
    if not yt_url:
        raise RuntimeError(
            f"Could not find this episode on YouTube.\n\n"
            f"Show:    {show_name}\n"
            f"Episode: {title}\n\n"
            f"Only episodes that are also published on YouTube (with captions) are supported."
        )

    logger.info(f"Found on YouTube: {yt_url}")
    result = transcribe_youtube(yt_url)
    result.url = url  # surface original Spotify URL
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def transcribe_url(url: str) -> dict:
    url = url.strip()

    if is_youtube_url(url):
        result = transcribe_youtube(url)
    elif is_spotify_url(url):
        result = transcribe_spotify(url)
    else:
        raise ValueError(
            "Unsupported URL. Please paste a YouTube video URL "
            "(youtube.com or youtu.be) or a Spotify episode URL "
            "(open.spotify.com/episode/...)."
        )

    return {
        "title": result.title,
        "source": result.source,
        "date": result.date,
        "duration_minutes": round(result.duration_minutes, 1),
        "url": result.url,
        "transcript": format_transcript(result),
        "method": result.method,
    }
