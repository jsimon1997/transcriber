import json
import logging
import os
import re
from typing import Optional
from dataclasses import dataclass
from html import unescape
from xml.etree import ElementTree

import requests

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
# Caption parsing
# ---------------------------------------------------------------------------

def _parse_json3_captions(content: str) -> list:
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
# Core: Fetch YouTube captions with proper session/cookies
# ---------------------------------------------------------------------------

def _fetch_captions_with_session(video_id: str) -> list:
    """Fetch captions using a proper HTTP session that maintains cookies.

    YouTube requires cookies (especially CONSENT) to serve caption content.
    By using a session, we get cookies from the watch page and reuse them
    for the caption URL fetch.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
    # Pre-set CONSENT cookie to bypass EU consent wall
    session.cookies.set("CONSENT", "YES+cb", domain=".youtube.com")

    # Step 1: Fetch the watch page to get cookies + captionTracks
    logger.info(f"Fetching YouTube page with session cookies...")
    resp = session.get(
        f"https://www.youtube.com/watch?v={video_id}",
        timeout=30,
    )
    resp.raise_for_status()
    html = resp.text
    logger.info(f"Page fetched: {len(html)} chars, cookies: {list(session.cookies.keys())}")

    # Step 2: Extract captionTracks
    match = re.search(r'"captionTracks"\s*:\s*(\[.*?\])', html)
    if not match:
        raise RuntimeError(
            "No captions are available for this video. "
            "Only videos with existing captions (auto-generated or manual) are supported."
        )

    tracks = json.loads(match.group(1))
    if not tracks:
        raise RuntimeError("No caption tracks found.")

    logger.info(f"Found {len(tracks)} caption tracks: {[t.get('languageCode','?') for t in tracks]}")

    # Prefer English
    caption_url = None
    for track in tracks:
        if track.get("languageCode", "").startswith("en"):
            caption_url = track.get("baseUrl", "").replace("\\u0026", "&")
            break
    if not caption_url:
        caption_url = tracks[0].get("baseUrl", "").replace("\\u0026", "&")

    if not caption_url:
        raise RuntimeError("No caption URL found in tracks.")

    # Step 3: Fetch captions using the SAME session (with cookies)
    segments = []
    json_url = caption_url + ("&" if "?" in caption_url else "?") + "fmt=json3"

    for fetch_url, fmt, parser in [
        (json_url, "json3", _parse_json3_captions),
        (caption_url, "xml", _parse_xml_captions),
    ]:
        try:
            logger.info(f"Fetching captions ({fmt}) with session cookies...")
            cap_resp = session.get(fetch_url, timeout=15)
            content = cap_resp.text.strip()
            logger.info(f"Caption response ({fmt}): {len(content)} chars, status: {cap_resp.status_code}")

            if not content:
                logger.warning(f"Empty response for {fmt}")
                continue

            segments = parser(content)
            if segments:
                logger.info(f"Parsed {len(segments)} segments from {fmt}")
                break
        except Exception as e:
            logger.warning(f"Caption fetch ({fmt}) failed: {e}")

    return segments


def _fetch_captions_via_scraper_session(video_id: str) -> list:
    """Fetch captions via ScraperAPI, replicating cookie-based session.

    Uses ScraperAPI to fetch the watch page, then tries to fetch caption
    URLs with cookies extracted from the page.
    """
    logger.info("Fetching YouTube page via ScraperAPI...")
    resp = requests.get(
        "https://api.scraperapi.com/",
        params={
            "api_key": SCRAPER_API_KEY,
            "url": f"https://www.youtube.com/watch?v={video_id}",
        },
        timeout=60,
    )
    resp.raise_for_status()
    html = resp.text
    logger.info(f"ScraperAPI page: {len(html)} chars")

    # Extract captionTracks
    match = re.search(r'"captionTracks"\s*:\s*(\[.*?\])', html)
    if not match:
        raise RuntimeError("No captions found via ScraperAPI.")

    tracks = json.loads(match.group(1))
    if not tracks:
        raise RuntimeError("Empty caption tracks.")

    caption_url = None
    for track in tracks:
        if track.get("languageCode", "").startswith("en"):
            caption_url = track.get("baseUrl", "").replace("\\u0026", "&")
            break
    if not caption_url:
        caption_url = tracks[0].get("baseUrl", "").replace("\\u0026", "&")

    # Try caption URLs with a session + cookies
    session = requests.Session()
    session.cookies.set("CONSENT", "YES+cb", domain=".youtube.com")
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    })

    segments = []
    json_url = caption_url + "&fmt=json3"

    for fetch_url, fmt, parser in [
        (json_url, "json3", _parse_json3_captions),
        (caption_url, "xml", _parse_xml_captions),
    ]:
        try:
            cap_resp = session.get(fetch_url, timeout=15)
            content = cap_resp.text.strip()
            logger.info(f"Direct caption ({fmt}): {len(content)} chars")
            if content:
                segments = parser(content)
                if segments:
                    break
        except Exception as e:
            logger.warning(f"Direct caption ({fmt}): {e}")

    # If direct still fails, try via ScraperAPI with cookies parameter
    if not segments:
        for fetch_url, fmt, parser in [
            (json_url, "json3", _parse_json3_captions),
            (caption_url, "xml", _parse_xml_captions),
        ]:
            try:
                r = requests.get("https://api.scraperapi.com/", params={
                    "api_key": SCRAPER_API_KEY,
                    "url": fetch_url,
                    "keep_headers": "true",
                }, headers={
                    "Cookie": "CONSENT=YES+cb",
                }, timeout=60)
                content = r.text.strip()
                logger.info(f"ScraperAPI caption ({fmt}): {len(content)} chars")
                if content:
                    segments = parser(content)
                    if segments:
                        break
            except Exception as e:
                logger.warning(f"ScraperAPI caption ({fmt}): {e}")

    return segments


# ---------------------------------------------------------------------------
# YouTube metadata
# ---------------------------------------------------------------------------

def _get_metadata(video_id: str) -> dict:
    """Fetch title and channel via YouTube oEmbed."""
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
        logger.warning(f"oEmbed failed: {e}")
        return {"title": "Unknown Title", "channel": "Unknown Channel"}


# ---------------------------------------------------------------------------
# YouTube pipeline
# ---------------------------------------------------------------------------

def transcribe_youtube(url: str) -> TranscriptResult:
    video_id = extract_youtube_id(url)
    if not video_id:
        raise ValueError(f"Could not extract a YouTube video ID from: {url}")

    logger.info(f"YouTube video ID: {video_id}")
    meta = _get_metadata(video_id)
    title = meta["title"]
    channel = meta["channel"]

    segments = []
    method = "unknown"
    errors = []

    # Strategy 1: Direct session with cookies (fastest, works from most IPs)
    try:
        logger.info("Strategy 1: Direct session with cookies...")
        segments = _fetch_captions_with_session(video_id)
        if segments:
            method = "session"
            logger.info(f"Session method: {len(segments)} segments")
    except Exception as e:
        logger.warning(f"Session method failed: {e}")
        errors.append(f"session: {e}")

    # Strategy 2: youtube-transcript-api library
    if not segments:
        try:
            from youtube_transcript_api import (
                YouTubeTranscriptApi,
                TranscriptsDisabled,
                NoTranscriptFound,
            )
            logger.info("Strategy 2: youtube-transcript-api...")
            entries = YouTubeTranscriptApi.get_transcript(video_id)
            segments = [{"start": e["start"], "text": e["text"]} for e in entries]
            if segments:
                method = "yt-api"
                logger.info(f"yt-api method: {len(segments)} segments")
        except Exception as e:
            logger.warning(f"yt-api failed: {e}")
            errors.append(f"yt-api: {e}")

    # Strategy 3: ScraperAPI-based fetch (when direct fails due to IP blocking)
    if not segments and SCRAPER_API_KEY:
        try:
            logger.info("Strategy 3: ScraperAPI scraper...")
            segments = _fetch_captions_via_scraper_session(video_id)
            if segments:
                method = "scraper"
                logger.info(f"Scraper method: {len(segments)} segments")
        except Exception as e:
            logger.warning(f"Scraper method failed: {e}")
            errors.append(f"scraper: {e}")

    if not segments:
        error_detail = "; ".join(errors) if errors else "unknown"
        raise RuntimeError(
            f"Could not fetch captions for this video. "
            f"Details: {error_detail}"
        )

    duration_min = segments[-1]["start"] / 60.0 if segments else 0.0
    logger.info(f"Done: {len(segments)} segments, ~{duration_min:.0f} min, method={method}")

    return TranscriptResult(
        title=title, source=channel, date="",
        duration_minutes=duration_min, url=url,
        segments=segments, method=method,
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
    """Search YouTube via scraping."""
    logger.info(f"Searching YouTube: {query!r}")
    try:
        if SCRAPER_API_KEY:
            resp = requests.get("https://api.scraperapi.com/", params={
                "api_key": SCRAPER_API_KEY,
                "url": f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}",
            }, timeout=60)
            html = resp.text
        else:
            resp = requests.get(
                "https://www.youtube.com/results",
                params={"search_query": query},
                headers={"User-Agent": "Mozilla/5.0"},
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
    meta = _fetch_spotify_meta(url)
    title = meta["title"]
    show_name = meta["show_name"]

    if not title:
        raise ValueError("Could not read episode info from Spotify.")

    logger.info(f"Episode: '{title}' | Show: '{show_name}'")
    yt_url = _search_youtube(f"{show_name} {title}")
    if not yt_url:
        raise RuntimeError(
            f"Could not find this episode on YouTube.\n"
            f"Show: {show_name}\nEpisode: {title}"
        )

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
            "Unsupported URL. Please paste a YouTube or Spotify episode URL."
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
