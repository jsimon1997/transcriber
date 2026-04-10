import glob
import json
import logging
import os
import re
import tempfile
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
# Parse subtitle files
# ---------------------------------------------------------------------------

def _parse_json3(content: str) -> list:
    data = json.loads(content)
    segments = []
    for event in data.get("events", []):
        start_ms = event.get("tStartMs", 0)
        segs = event.get("segs", [])
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if text and text != "\n":
            segments.append({"start": start_ms / 1000.0, "text": text})
    return segments


def _parse_vtt(content: str) -> list:
    """Parse WebVTT subtitle format."""
    segments = []
    # Match timestamp lines like: 00:00:01.360 --> 00:00:04.500
    pattern = r"(\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}"
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        m = re.match(pattern, lines[i].strip())
        if m:
            ts = m.group(1)
            parts = ts.split(":")
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            # Collect text lines until empty line
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip():
                # Remove VTT formatting tags like <c>, </c>, etc.
                line = re.sub(r"<[^>]+>", "", lines[i].strip())
                if line:
                    text_lines.append(line)
                i += 1
            text = " ".join(text_lines).strip()
            if text:
                segments.append({"start": seconds, "text": text})
        i += 1
    return segments


def _parse_srv3(content: str) -> list:
    """Parse YouTube srv3 (XML-based) subtitle format."""
    try:
        root = ElementTree.fromstring(content)
        segments = []
        for p in root.findall(".//p"):
            start_ms = int(p.get("t", "0"))
            text_parts = []
            # Get direct text
            if p.text:
                text_parts.append(p.text)
            # Get text from child <s> elements
            for s in p.findall("s"):
                if s.text:
                    text_parts.append(s.text)
                if s.tail:
                    text_parts.append(s.tail)
            text = " ".join(text_parts).strip()
            text = unescape(text)
            if text:
                segments.append({"start": start_ms / 1000.0, "text": text})
        return segments
    except Exception:
        return []


def _parse_xml_captions(content: str) -> list:
    """Parse standard YouTube XML caption format."""
    root = ElementTree.fromstring(content)
    segments = []
    for elem in root.findall(".//text"):
        start = float(elem.get("start", 0))
        text = unescape(elem.text or "").strip()
        if text:
            segments.append({"start": start, "text": text})
    return segments


def _parse_subtitle_file(filepath: str) -> list:
    """Parse a subtitle file, auto-detecting format."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        return []

    # Try JSON3 first
    try:
        segments = _parse_json3(content)
        if segments:
            return segments
    except (json.JSONDecodeError, Exception):
        pass

    # Try VTT
    if "WEBVTT" in content[:50] or "-->" in content[:500]:
        segments = _parse_vtt(content)
        if segments:
            return segments

    # Try srv3 / XML
    if content.strip().startswith("<"):
        segments = _parse_srv3(content)
        if segments:
            return segments
        segments = _parse_xml_captions(content)
        if segments:
            return segments

    return []


# ---------------------------------------------------------------------------
# yt-dlp subtitle extraction (primary method)
# ---------------------------------------------------------------------------

def _fetch_captions_ytdlp(video_id: str) -> tuple:
    """Extract subtitles using yt-dlp.

    Uses yt-dlp to actually DOWNLOAD subtitles (not just get URLs),
    so yt-dlp's anti-bot handling (including mobile innertube clients)
    is used for the actual content download.
    """
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")

        ydl_opts = {
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-GB", "en.*"],
            "subtitlesformat": "json3/srv3/vtt/best",
            "skip_download": True,
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            # Use player clients that bypass YouTube bot detection
            "extractor_args": {
                "youtube": {
                    "player_client": ["mediaconnect", "android", "web"],
                }
            },
        }

        # If ScraperAPI is available, use it as proxy to bypass IP blocking
        if SCRAPER_API_KEY:
            ydl_opts["proxy"] = f"http://scraperapi:{SCRAPER_API_KEY}@proxy-server.scraperapi.com:8001"
            ydl_opts["nocheckcertificate"] = True
            logger.info("Using ScraperAPI proxy for yt-dlp")

        logger.info(f"Running yt-dlp for {video_id}...")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        title = info.get("title", "Unknown Title")
        channel = info.get("uploader", info.get("channel", "Unknown Channel"))
        duration = info.get("duration", 0)

        # Find downloaded subtitle files
        sub_files = glob.glob(os.path.join(tmpdir, f"{video_id}.*"))
        # Filter to subtitle files only (not video/audio/info)
        sub_extensions = {".json3", ".vtt", ".srv3", ".srt", ".ttml", ".ass"}
        sub_files = [f for f in sub_files if any(f.endswith(ext) for ext in sub_extensions)
                     or ".en." in f or ".en-" in f]

        logger.info(f"yt-dlp files in tmpdir: {[os.path.basename(f) for f in glob.glob(os.path.join(tmpdir, '*'))]}")
        logger.info(f"Subtitle files found: {[os.path.basename(f) for f in sub_files]}")

        segments = []
        for sub_file in sub_files:
            logger.info(f"Parsing subtitle file: {os.path.basename(sub_file)}")
            segments = _parse_subtitle_file(sub_file)
            if segments:
                logger.info(f"Parsed {len(segments)} segments from {os.path.basename(sub_file)}")
                break

        # If no subtitle files were written, try the info dict for subtitle URLs
        # and fetch them using yt-dlp's URL opener (which handles cookies/auth)
        if not segments:
            logger.info("No subtitle files found, trying info dict...")
            for sub_type in ["subtitles", "automatic_captions"]:
                subs = info.get(sub_type, {})
                for lang in ["en", "en-US", "en-GB"]:
                    if lang not in subs:
                        continue
                    for fmt in subs[lang]:
                        sub_url = fmt.get("url")
                        if not sub_url:
                            continue
                        try:
                            # Use yt-dlp's URL opener which has proper cookies
                            with yt_dlp.YoutubeDL({"quiet": True}) as ydl2:
                                resp_data = ydl2.urlopen(sub_url).read().decode("utf-8")
                            if resp_data.strip():
                                try:
                                    segments = _parse_json3(resp_data)
                                except Exception:
                                    pass
                                if not segments and ("WEBVTT" in resp_data[:50] or "-->" in resp_data[:500]):
                                    segments = _parse_vtt(resp_data)
                                if not segments and resp_data.strip().startswith("<"):
                                    segments = _parse_srv3(resp_data) or _parse_xml_captions(resp_data)
                                if segments:
                                    logger.info(f"Got {len(segments)} segments from URL ({sub_type}/{lang})")
                                    return segments, title, channel, duration
                        except Exception as e:
                            logger.warning(f"yt-dlp URL fetch failed: {e}")

    return segments, title, channel, duration


# ---------------------------------------------------------------------------
# youtube-transcript-api fallback
# ---------------------------------------------------------------------------

def _fetch_captions_ytapi(video_id: str) -> list:
    from youtube_transcript_api import YouTubeTranscriptApi
    entries = YouTubeTranscriptApi.get_transcript(video_id)
    return [{"start": e["start"], "text": e["text"]} for e in entries]


# ---------------------------------------------------------------------------
# YouTube metadata
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

    # Strategy 1: yt-dlp (handles anti-bot, downloads subs properly)
    try:
        logger.info("Strategy 1: yt-dlp subtitle download...")
        segments, title, channel, duration = _fetch_captions_ytdlp(video_id)
        if segments:
            method = "yt-dlp"
    except Exception as e:
        logger.warning(f"yt-dlp failed: {e}")
        errors.append(f"yt-dlp: {e}")

    # Strategy 2: youtube-transcript-api
    if not segments:
        try:
            logger.info("Strategy 2: youtube-transcript-api...")
            segments = _fetch_captions_ytapi(video_id)
            if segments:
                method = "yt-api"
                meta = _get_metadata(video_id)
                title = meta["title"]
                channel = meta["channel"]
        except Exception as e:
            logger.warning(f"yt-api failed: {e}")
            errors.append(f"yt-api: {e}")

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
    """Fetch Spotify episode metadata.

    Uses ScraperAPI if available (Spotify blocks direct access from cloud IPs),
    falls back to direct fetch.
    """
    def _extract_meta(html: str) -> dict:
        def og(prop):
            m = re.search(rf'<meta property="og:{prop}" content="([^"]+)"', html)
            return m.group(1) if m else ""

        title = og("title")
        show_name = og("site_name") or "Spotify Podcast"

        # Also try ld+json for richer metadata
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

    # Try ScraperAPI first (Spotify blocks cloud IPs)
    if SCRAPER_API_KEY:
        try:
            logger.info("Fetching Spotify page via ScraperAPI...")
            resp = requests.get(
                "https://api.scraperapi.com/",
                params={"api_key": SCRAPER_API_KEY, "url": url},
                timeout=30,
            )
            resp.raise_for_status()
            meta = _extract_meta(resp.text)
            if meta["title"]:
                return meta
            logger.warning("ScraperAPI returned empty Spotify metadata")
        except Exception as e:
            logger.warning(f"ScraperAPI Spotify fetch failed: {e}")

    # Fallback: direct fetch
    try:
        resp = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        }, timeout=15)
        resp.raise_for_status()
        return _extract_meta(resp.text)
    except Exception as e:
        logger.warning(f"Direct Spotify fetch failed: {e}")
        return {"title": "", "show_name": ""}


def _search_youtube(query: str) -> Optional[str]:
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
        raise ValueError(
            "Could not read episode info from Spotify. "
            "Please check the URL is correct, or try pasting the YouTube URL directly "
            "if this podcast is also on YouTube."
        )

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
