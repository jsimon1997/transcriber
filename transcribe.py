import json
import logging
import os
import random
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
# ScraperAPI helper (with session support)
# ---------------------------------------------------------------------------

def _scraper_fetch(url: str, session_number: Optional[int] = None) -> str:
    """Fetch a URL via ScraperAPI direct API.

    Use session_number to pin requests to the same proxy IP — critical
    for YouTube caption URLs which are IP-bound.
    """
    params = {"api_key": SCRAPER_API_KEY, "url": url}
    if session_number is not None:
        params["session_number"] = str(session_number)

    resp = requests.get(
        "https://api.scraperapi.com/",
        params=params,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.text


def _clean_caption_url(url: str) -> str:
    """Ensure all unicode escapes are decoded in caption URLs."""
    # json.loads handles \u0026 → & but some edge cases leave them.
    # Belt-and-suspenders cleanup:
    url = url.replace("\\u0026", "&")
    url = url.replace("\\u003d", "=")
    url = url.replace("\\u003f", "?")
    return url


# ---------------------------------------------------------------------------
# Parse caption segments from JSON or XML
# ---------------------------------------------------------------------------

def _parse_json_captions(content: str) -> list:
    """Parse YouTube JSON3 caption format."""
    data = json.loads(content)
    segments = []
    for event in data.get("events", []):
        start_ms = event.get("tStartMs", 0)
        segs = event.get("segs", [])
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if text and text != "\n":
            segments.append({"start": start_ms / 1000.0, "text": text})
    return segments


def _parse_xml_captions(content: str) -> list:
    """Parse YouTube XML caption format."""
    root = ElementTree.fromstring(content)
    segments = []
    for elem in root.findall(".//text"):
        start = float(elem.get("start", 0))
        text = unescape(elem.text or "").strip()
        if text:
            segments.append({"start": start, "text": text})
    return segments


# ---------------------------------------------------------------------------
# YouTube captions via ScraperAPI (server-side)
# ---------------------------------------------------------------------------

def _fetch_captions_via_scraper(video_id: str) -> list:
    """Fetch YouTube captions by scraping the watch page via ScraperAPI.

    Uses a consistent ScraperAPI session so the page fetch and caption
    fetch come from the same proxy IP (caption URLs are IP-bound).
    """
    # Use a random session number so concurrent requests don't collide
    session = random.randint(1, 999999)

    logger.info(f"Fetching YouTube page via ScraperAPI (session={session})...")

    # Step 1: Fetch the YouTube watch page
    html = _scraper_fetch(
        f"https://www.youtube.com/watch?v={video_id}",
        session_number=session,
    )
    logger.info(f"Page fetched: {len(html)} chars")

    # Step 2: Extract captionTracks from the page's embedded JavaScript
    match = re.search(r'"captionTracks"\s*:\s*(\[.*?\])', html)
    if not match:
        raise RuntimeError(
            "No captions are available for this video. "
            "Only videos with existing captions (auto-generated or manual) are supported."
        )

    raw_json = match.group(1)
    logger.info(f"captionTracks JSON found ({len(raw_json)} chars)")

    try:
        tracks = json.loads(raw_json)
    except json.JSONDecodeError:
        # Sometimes the JSON has issues; try cleaning unicode escapes first
        cleaned = raw_json.replace("\\u0026", "&")
        try:
            tracks = json.loads(cleaned)
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse caption track data from YouTube.")

    if not tracks:
        raise RuntimeError("No captions are available for this video.")

    logger.info(f"Found {len(tracks)} caption track(s): {[t.get('languageCode','?') for t in tracks]}")

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

    # Clean any remaining unicode escapes
    caption_url = _clean_caption_url(caption_url)
    logger.info(f"Caption URL (cleaned): {caption_url[:100]}...")

    # Step 3: Fetch captions using the SAME ScraperAPI session (same proxy IP)
    json_url = caption_url + ("&" if "?" in caption_url else "?") + "fmt=json3"

    segments = []

    # Try JSON format first, then XML — all via ScraperAPI with same session
    for fetch_url, fmt, parser in [
        (json_url, "json3", _parse_json_captions),
        (caption_url, "xml", _parse_xml_captions),
    ]:
        try:
            logger.info(f"Fetching captions ({fmt}) via ScraperAPI (session={session})...")
            content = _scraper_fetch(fetch_url, session_number=session).strip()
            logger.info(f"Caption response ({fmt}): {len(content)} chars")

            if not content:
                logger.warning(f"Empty response for {fmt} format")
                continue

            segments = parser(content)
            if segments:
                logger.info(f"Parsed {len(segments)} segments from {fmt} format")
                break
            else:
                logger.warning(f"No segments parsed from {fmt} format")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"ScraperAPI caption fetch ({fmt}) HTTP error: {e}")
        except Exception as e:
            logger.warning(f"ScraperAPI caption fetch ({fmt}) failed: {e}")

    # Fallback: try direct fetch (works for some videos from some IPs)
    if not segments:
        logger.info("ScraperAPI session fetch failed, trying direct fetch as fallback...")
        for fetch_url, fmt, parser in [
            (json_url, "json3", _parse_json_captions),
            (caption_url, "xml", _parse_xml_captions),
        ]:
            try:
                resp = requests.get(fetch_url, timeout=15)
                resp.raise_for_status()
                content = resp.text.strip()
                logger.info(f"Direct fetch ({fmt}): {len(content)} chars")
                if not content:
                    continue
                segments = parser(content)
                if segments:
                    logger.info(f"Direct fetch worked: {len(segments)} segments from {fmt}")
                    break
            except Exception as e:
                logger.warning(f"Direct caption fetch ({fmt}) failed: {e}")

    if not segments:
        raise RuntimeError(
            "Captions were found but could not be downloaded. "
            "The caption URLs may have expired. Please try again."
        )

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
        return {
            "title": data.get("title", "Unknown Title"),
            "channel": data.get("author_name", "Unknown Channel"),
        }
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

    # Calculate duration from last segment
    duration_min = 0.0
    if segments:
        duration_min = segments[-1]["start"] / 60.0

    logger.info(f"Transcription complete: {len(segments)} segments, ~{duration_min:.0f} min")
    return TranscriptResult(
        title=title, source=channel, date="",
        duration_minutes=duration_min, url=url,
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
            html = _scraper_fetch(
                f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}"
            )
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
