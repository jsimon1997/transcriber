import json
import logging
import os
import re
from typing import Optional
from dataclasses import dataclass
from html import unescape
from xml.etree import ElementTree

import requests
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

logger = logging.getLogger(__name__)

SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY", "")


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
# ScraperAPI helper
# ---------------------------------------------------------------------------

def _scraper_fetch(url: str) -> str:
    """Fetch a URL via ScraperAPI's direct API — no proxy, no SSL issues."""
    resp = requests.get(
        "https://api.scraperapi.com/",
        params={"api_key": SCRAPER_API_KEY, "url": url},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# YouTube captions via ScraperAPI (server-side)
# ---------------------------------------------------------------------------

def _fetch_captions_via_scraper(video_id: str) -> list:
    """Fetch YouTube captions by scraping the watch page via ScraperAPI."""
    logger.info("Fetching captions via ScraperAPI...")

    # Step 1: Fetch the YouTube watch page
    html = _scraper_fetch(f"https://www.youtube.com/watch?v={video_id}")

    # Step 2: Extract captionTracks from the page's JavaScript
    match = re.search(r'"captionTracks"\s*:\s*(\[.*?\])', html)
    if not match:
        raise RuntimeError(
            "No captions are available for this video. "
            "Only videos with existing captions (auto-generated or manual) are supported."
        )

    try:
        tracks = json.loads(match.group(1))
    except json.JSONDecodeError:
        raise RuntimeError("Failed to parse caption track data from YouTube.")

    if not tracks:
        raise RuntimeError("No captions are available for this video.")

    # Prefer English, fall back to first available
    caption_url = None
    for track in tracks:
        if track.get("languageCode", "").startswith("en"):
            caption_url = track.get("baseUrl")
            break
    if not caption_url:
        caption_url = tracks[0].get("baseUrl")

    if not caption_url:
        raise RuntimeError("Caption URL not found in track data.")

    # Step 3: Fetch the captions XML via ScraperAPI
    logger.info("Fetching caption XML...")
    xml_text = _scraper_fetch(caption_url)

    # Step 4: Parse XML into segments
    root = ElementTree.fromstring(xml_text)
    segments = []
    for elem in root.findall(".//text"):
        start = float(elem.get("start", 0))
        text = unescape(elem.text or "")
        if text.strip():
            segments.append({"start": start, "text": text})

    if not segments:
        raise RuntimeError("Captions were found but contained no text.")

    logger.info(f"Parsed {len(segments)} caption segments.")
    return segments


# ---------------------------------------------------------------------------
# YouTube metadata
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


# ---------------------------------------------------------------------------
# YouTube pipeline
# ---------------------------------------------------------------------------

def transcribe_youtube(url: str) -> TranscriptResult:
    video_id = extract_youtube_id(url)
    if not video_id:
        raise ValueError(f"Could not extract a YouTube video ID from: {url}")

    logger.info(f"YouTube video ID: {video_id}")

    logger.info("Fetching video metadata...")
    meta = _get_metadata(video_id)
    title = meta["title"]
    channel = meta["channel"]

    logger.info("Fetching captions...")

    if SCRAPER_API_KEY:
        # Server path: use ScraperAPI to bypass YouTube IP blocking
        segments = _fetch_captions_via_scraper(video_id)
    else:
        # Local path: use youtube-transcript-api directly
        try:
            entries = YouTubeTranscriptApi.get_transcript(video_id)
            segments = [{"start": e["start"], "text": e["text"]} for e in entries]
        except (TranscriptsDisabled, NoTranscriptFound):
            raise RuntimeError(
                "No captions are available for this video. "
                "Only videos with existing captions (auto-generated or manual) are supported."
            )

    logger.info("Captions found.")
    return TranscriptResult(
        title=title, source=channel, date="",
        duration_minutes=0, url=url,
        segments=segments, method="captions",
    )


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
        if SCRAPER_API_KEY:
            html = _scraper_fetch(f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}")
        else:
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            resp = requests.get(
                "https://www.youtube.com/results",
                params={"search_query": query},
                headers=headers,
                timeout=10,
            )
            html = resp.text
        ids = re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', html)
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
    result.url = url
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
