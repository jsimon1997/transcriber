import json
import logging
import os
import re
from typing import Optional
from dataclasses import dataclass

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
# URL helpers
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
# yt-dlp subtitle extraction (primary method)
# ---------------------------------------------------------------------------

def _fetch_captions_ytdlp(video_id: str) -> tuple:
    """Extract subtitles using yt-dlp. Returns (segments, title, channel, duration).

    yt-dlp handles all YouTube anti-bot measures internally, including
    using mobile/TV innertube clients that work from cloud IPs.
    """
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "json3",
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title", "Unknown Title")
    channel = info.get("uploader", info.get("channel", "Unknown Channel"))
    duration = info.get("duration", 0)

    # Try manual subtitles first, then automatic
    segments = []
    for sub_type in ["subtitles", "automatic_captions"]:
        subs = info.get(sub_type, {})
        if not subs:
            continue

        # Find English subtitles
        for lang_code in ["en", "en-US", "en-GB"]:
            if lang_code not in subs:
                continue

            formats = subs[lang_code]
            # Prefer json3 format
            sub_url = None
            for fmt in formats:
                if fmt.get("ext") == "json3":
                    sub_url = fmt.get("url")
                    break
            if not sub_url:
                # Fall back to first available format
                for fmt in formats:
                    if fmt.get("url"):
                        sub_url = fmt.get("url")
                        break

            if sub_url:
                logger.info(f"Fetching subtitles ({sub_type}/{lang_code})...")
                try:
                    resp = requests.get(sub_url, timeout=30)
                    resp.raise_for_status()
                    content = resp.text.strip()

                    if not content:
                        continue

                    # Try JSON3 format
                    try:
                        data = json.loads(content)
                        for event in data.get("events", []):
                            start_ms = event.get("tStartMs", 0)
                            segs = event.get("segs", [])
                            text = "".join(s.get("utf8", "") for s in segs).strip()
                            if text and text != "\n":
                                segments.append({"start": start_ms / 1000.0, "text": text})
                    except json.JSONDecodeError:
                        # Try XML format
                        from html import unescape
                        from xml.etree import ElementTree
                        root = ElementTree.fromstring(content)
                        for elem in root.findall(".//text"):
                            start = float(elem.get("start", 0))
                            text = unescape(elem.text or "").strip()
                            if text:
                                segments.append({"start": start, "text": text})

                    if segments:
                        logger.info(f"Got {len(segments)} segments from {sub_type}/{lang_code}")
                        return segments, title, channel, duration
                except Exception as e:
                    logger.warning(f"Failed to fetch subtitle URL: {e}")

        # If we found subtitles dict but URL fetch failed, try getting data directly
        for lang_code in ["en", "en-US", "en-GB"]:
            if lang_code not in subs:
                continue
            formats = subs[lang_code]
            for fmt in formats:
                # Some yt-dlp versions include subtitle data directly
                if "data" in fmt:
                    try:
                        data = json.loads(fmt["data"]) if isinstance(fmt["data"], str) else fmt["data"]
                        for event in data.get("events", []):
                            start_ms = event.get("tStartMs", 0)
                            segs_data = event.get("segs", [])
                            text = "".join(s.get("utf8", "") for s in segs_data).strip()
                            if text and text != "\n":
                                segments.append({"start": start_ms / 1000.0, "text": text})
                        if segments:
                            return segments, title, channel, duration
                    except Exception:
                        pass

    return segments, title, channel, duration


# ---------------------------------------------------------------------------
# youtube-transcript-api fallback
# ---------------------------------------------------------------------------

def _fetch_captions_ytapi(video_id: str) -> list:
    """Fallback: use youtube-transcript-api library."""
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
    )
    entries = YouTubeTranscriptApi.get_transcript(video_id)
    return [{"start": e["start"], "text": e["text"]} for e in entries]


# ---------------------------------------------------------------------------
# YouTube metadata (via oEmbed, as backup)
# ---------------------------------------------------------------------------

def _get_metadata(video_id: str) -> dict:
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
    except Exception:
        return {"title": "Unknown Title", "channel": "Unknown Channel"}


# ---------------------------------------------------------------------------
# YouTube pipeline
# ---------------------------------------------------------------------------

def transcribe_youtube(url: str) -> TranscriptResult:
    video_id = extract_youtube_id(url)
    if not video_id:
        raise ValueError(f"Could not extract a YouTube video ID from: {url}")

    logger.info(f"YouTube video ID: {video_id}")

    segments = []
    title = "Unknown Title"
    channel = "Unknown Channel"
    duration = 0
    method = "unknown"
    errors = []

    # Strategy 1: yt-dlp (handles anti-bot, works from cloud IPs)
    try:
        logger.info("Strategy 1: yt-dlp subtitle extraction...")
        segments, title, channel, duration = _fetch_captions_ytdlp(video_id)
        if segments:
            method = "yt-dlp"
            logger.info(f"yt-dlp: {len(segments)} segments, title='{title}'")
    except Exception as e:
        logger.warning(f"yt-dlp failed: {e}")
        errors.append(f"yt-dlp: {e}")

    # Strategy 2: youtube-transcript-api (works from residential IPs)
    if not segments:
        try:
            logger.info("Strategy 2: youtube-transcript-api...")
            segments = _fetch_captions_ytapi(video_id)
            if segments:
                method = "yt-api"
                # Get metadata separately since library doesn't provide it
                meta = _get_metadata(video_id)
                title = meta["title"]
                channel = meta["channel"]
        except Exception as e:
            logger.warning(f"youtube-transcript-api failed: {e}")
            errors.append(f"yt-api: {e}")

    # Get metadata via oEmbed if yt-dlp didn't provide it
    if title == "Unknown Title":
        meta = _get_metadata(video_id)
        title = meta["title"]
        channel = meta["channel"]

    if not segments:
        error_detail = "; ".join(errors) if errors else "unknown"
        raise RuntimeError(
            f"Could not fetch captions for this video. "
            f"The video may not have captions enabled. "
            f"Details: {error_detail}"
        )

    duration_min = duration / 60.0 if duration else (segments[-1]["start"] / 60.0 if segments else 0)

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
    """Search YouTube for a video matching the query."""
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
