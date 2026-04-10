import base64
import json
import logging
import os
import re
from typing import Optional
from dataclasses import dataclass
from html import unescape

import requests
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

logger = logging.getLogger(__name__)

SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY", "")

# YouTube's public innertube API key (embedded in every YouTube page)
INNERTUBE_API_KEY = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"


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

def _scraper_fetch(url: str, session_number: Optional[int] = None) -> str:
    """Fetch a URL via ScraperAPI direct API."""
    params = {"api_key": SCRAPER_API_KEY, "url": url}
    if session_number is not None:
        params["session_number"] = str(session_number)
    resp = requests.get("https://api.scraperapi.com/", params=params, timeout=60)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# YouTube innertube transcript API
# ---------------------------------------------------------------------------

def _build_transcript_params(video_id: str) -> str:
    """Build the base64-encoded protobuf params for get_transcript API."""
    # Protobuf encoding for GetTranscriptParams { VideoInfo { video_id } }
    vid_bytes = video_id.encode("utf-8")
    inner = b"\x0a" + bytes([len(vid_bytes)]) + vid_bytes  # field 1 = video_id
    outer = b"\x0a" + bytes([len(inner)]) + inner           # field 1 = VideoInfo
    return base64.b64encode(outer).decode("ascii")


def _fetch_transcript_innertube(video_id: str, via_scraper: bool = False) -> list:
    """Fetch transcript using YouTube's innertube get_transcript API.

    This is the same API the YouTube website uses for "Show transcript".
    It doesn't rely on caption URLs (which are IP/session locked).
    """
    params_b64 = _build_transcript_params(video_id)

    payload = {
        "context": {
            "client": {
                "clientName": "WEB",
                "clientVersion": "2.20241120.01.00",
                "hl": "en",
            }
        },
        "params": params_b64,
    }

    api_url = f"https://www.youtube.com/youtubei/v1/get_transcript?key={INNERTUBE_API_KEY}"

    if via_scraper and SCRAPER_API_KEY:
        logger.info("Fetching transcript via innertube API (through ScraperAPI)...")
        resp = requests.get(
            "https://api.scraperapi.com/",
            params={
                "api_key": SCRAPER_API_KEY,
                "url": api_url,
                # ScraperAPI doesn't support POST passthrough easily,
                # so we'll try direct first
            },
            timeout=60,
        )
        # ScraperAPI GET won't work for POST endpoint — fall back to direct
        logger.warning("ScraperAPI GET for innertube not applicable, using direct POST")
        via_scraper = False

    logger.info("Fetching transcript via innertube API (direct POST)...")
    resp = requests.post(
        api_url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    # Parse transcript from innertube response
    segments = []

    # Navigate the response structure
    actions = data.get("actions", [])
    for action in actions:
        panel = action.get("updateEngagementPanelAction", {}).get("content", {})
        transcript_renderer = panel.get("transcriptRenderer", {})
        body = transcript_renderer.get("body", {})
        search_panel = body.get("transcriptSearchPanelRenderer", {})
        segment_list = search_panel.get("body", {}).get("transcriptSegmentListRenderer", {})

        initial_segments = segment_list.get("initialSegments", [])
        for seg in initial_segments:
            segment_renderer = seg.get("transcriptSegmentRenderer", {})
            start_ms = int(segment_renderer.get("startMs", "0"))
            end_ms = int(segment_renderer.get("endMs", "0"))
            snippet = segment_renderer.get("snippet", {})

            # Extract text from snippet runs
            text_parts = []
            for run in snippet.get("runs", []):
                text_parts.append(run.get("text", ""))
            text = "".join(text_parts).strip()

            if text:
                segments.append({
                    "start": start_ms / 1000.0,
                    "text": text,
                })

    return segments


def _fetch_captions_via_scraper(video_id: str) -> list:
    """Fetch captions by scraping the YouTube page and extracting caption tracks.

    Uses ScraperAPI to fetch the watch page, finds captionTracks,
    then tries to fetch caption content.
    """
    logger.info("Fetching YouTube page via ScraperAPI for caption tracks...")
    html = _scraper_fetch(f"https://www.youtube.com/watch?v={video_id}")
    logger.info(f"Page fetched: {len(html)} chars")

    # Extract captionTracks
    match = re.search(r'"captionTracks"\s*:\s*(\[.*?\])', html)
    if not match:
        raise RuntimeError("No captions found on this video page.")

    try:
        tracks = json.loads(match.group(1))
    except json.JSONDecodeError:
        raise RuntimeError("Failed to parse caption track data.")

    if not tracks:
        raise RuntimeError("No caption tracks available.")

    # Prefer English
    caption_url = None
    for track in tracks:
        if track.get("languageCode", "").startswith("en"):
            caption_url = track.get("baseUrl", "").replace("\\u0026", "&")
            break
    if not caption_url:
        caption_url = tracks[0].get("baseUrl", "").replace("\\u0026", "&")

    if not caption_url:
        raise RuntimeError("Caption URL not found.")

    logger.info(f"Caption URL found, trying to fetch content...")

    # Try direct fetch (sometimes works)
    segments = []
    for fmt_param, fmt_name in [("&fmt=json3", "json"), ("", "xml")]:
        try:
            fetch_url = caption_url + fmt_param
            resp = requests.get(fetch_url, timeout=15)
            content = resp.text.strip()
            if not content:
                continue

            if fmt_name == "json":
                data = json.loads(content)
                for event in data.get("events", []):
                    start_ms = event.get("tStartMs", 0)
                    segs = event.get("segs", [])
                    text = "".join(s.get("utf8", "") for s in segs).strip()
                    if text and text != "\n":
                        segments.append({"start": start_ms / 1000.0, "text": text})
            else:
                from xml.etree import ElementTree
                root = ElementTree.fromstring(content)
                for elem in root.findall(".//text"):
                    start = float(elem.get("start", 0))
                    text = unescape(elem.text or "").strip()
                    if text:
                        segments.append({"start": start, "text": text})

            if segments:
                return segments
        except Exception as e:
            logger.warning(f"Caption URL fetch ({fmt_name}) failed: {e}")

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

    meta = _get_metadata(video_id)
    title = meta["title"]
    channel = meta["channel"]

    segments = []
    method = "captions"
    errors = []

    # Strategy 1: Innertube get_transcript API (works from any IP)
    try:
        logger.info("Strategy 1: Innertube get_transcript API...")
        segments = _fetch_transcript_innertube(video_id)
        if segments:
            logger.info(f"Innertube API returned {len(segments)} segments")
            method = "innertube"
    except Exception as e:
        logger.warning(f"Innertube API failed: {e}")
        errors.append(f"innertube: {e}")

    # Strategy 2: youtube-transcript-api library (works from non-blocked IPs)
    if not segments:
        try:
            logger.info("Strategy 2: youtube-transcript-api library...")
            entries = YouTubeTranscriptApi.get_transcript(video_id)
            segments = [{"start": e["start"], "text": e["text"]} for e in entries]
            if segments:
                logger.info(f"youtube-transcript-api returned {len(segments)} segments")
                method = "yt-transcript-api"
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            logger.warning(f"youtube-transcript-api: no captions available: {e}")
            errors.append(f"yt-api: {e}")
        except Exception as e:
            logger.warning(f"youtube-transcript-api failed: {e}")
            errors.append(f"yt-api: {e}")

    # Strategy 3: ScraperAPI page scraping + caption URL fetch
    if not segments and SCRAPER_API_KEY:
        try:
            logger.info("Strategy 3: ScraperAPI page scraping...")
            segments = _fetch_captions_via_scraper(video_id)
            if segments:
                logger.info(f"ScraperAPI scraping returned {len(segments)} segments")
                method = "scraper"
        except Exception as e:
            logger.warning(f"ScraperAPI scraping failed: {e}")
            errors.append(f"scraper: {e}")

    if not segments:
        error_detail = "; ".join(errors) if errors else "unknown"
        raise RuntimeError(
            f"Could not fetch captions for this video. "
            f"The video may not have captions, or YouTube may be blocking access. "
            f"Details: {error_detail}"
        )

    # Calculate duration from last segment
    duration_min = segments[-1]["start"] / 60.0 if segments else 0.0

    logger.info(f"Transcription complete: {len(segments)} segments, ~{duration_min:.0f} min, method={method}")
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
