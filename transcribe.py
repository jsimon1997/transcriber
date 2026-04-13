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
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")


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
    return "open.spotify.com/episode" in url or "open.spotify.com/show" in url


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
        upload_date = info.get("upload_date", "")  # YYYYMMDD format

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
                                    return segments, title, channel, duration, upload_date
                        except Exception as e:
                            logger.warning(f"yt-dlp URL fetch failed: {e}")

    return segments, title, channel, duration, upload_date


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
    upload_date = ""
    method = "unknown"
    errors = []

    # Strategy 1: yt-dlp (handles anti-bot, downloads subs properly)
    try:
        logger.info("Strategy 1: yt-dlp subtitle download...")
        segments, title, channel, duration, upload_date = _fetch_captions_ytdlp(video_id)
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

    # Format upload date from YYYYMMDD to YYYY-MM-DD
    formatted_date = ""
    if upload_date and len(upload_date) == 8:
        formatted_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"

    logger.info(f"Done: {len(segments)} segments, ~{duration_min:.0f} min, method={method}, date={formatted_date}")
    return TranscriptResult(
        title=title, source=channel, date=formatted_date,
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
    def _clean_title(t: str) -> str:
        if not t:
            return ""
        # Strip common Spotify prefixes
        t = re.sub(r"^(Spotify Episode:\s*|Spotify Podcast:\s*)", "", t).strip()
        return t

    def _extract_meta(html: str) -> dict:
        def og(prop):
            m = re.search(rf'<meta property="og:{prop}" content="([^"]+)"', html)
            return m.group(1) if m else ""

        def meta_name(name):
            m = re.search(rf'<meta name="{name}" content="([^"]+)"', html)
            return m.group(1) if m else ""

        title = _clean_title(og("title"))
        show_name = ""

        # Method 1: parse ALL ld+json blocks — find the PodcastEpisode one
        for ld_match in re.finditer(
            r'<script type="application/ld\+json"[^>]*>(.*?)</script>', html, re.DOTALL
        ):
            try:
                data = json.loads(ld_match.group(1))
                if isinstance(data, list):
                    candidates = data
                else:
                    candidates = [data]
                for d in candidates:
                    if not isinstance(d, dict):
                        continue
                    t = d.get("@type", "")
                    if "PodcastEpisode" in str(t) or "Episode" in str(t):
                        if d.get("name"):
                            title = _clean_title(d["name"])
                        pos = d.get("partOfSeries") or {}
                        if isinstance(pos, dict) and pos.get("name"):
                            show_name = pos["name"]
                        elif isinstance(pos, list) and pos:
                            show_name = pos[0].get("name", "")
            except Exception:
                continue

        # Method 2: parse the description meta tag
        # Spotify usually sets: "Listen to this episode from SHOWNAME on Spotify. ..."
        if not show_name:
            desc = meta_name("description") or og("description")
            m = re.match(r"Listen to this episode from (.+?) on Spotify", desc)
            if m:
                show_name = m.group(1).strip()

        # Method 3: look for music:album or show link
        if not show_name:
            show_link = og("music:album") or meta_name("music:album")
            if "/show/" in show_link:
                # Try to find the show's title in the HTML near that URL
                show_id_match = re.search(r"/show/([A-Za-z0-9]+)", show_link)
                if show_id_match:
                    sid = show_id_match.group(1)
                    show_title_match = re.search(
                        rf'"show"\s*:\s*{{[^}}]*"id"\s*:\s*"{sid}"[^}}]*"name"\s*:\s*"([^"]+)"',
                        html,
                    )
                    if show_title_match:
                        show_name = show_title_match.group(1)

        if not show_name:
            show_name = "Spotify Podcast"

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


def _find_rss_audio(show_name: str, episode_title: str) -> Optional[dict]:
    """Try to find the podcast episode audio URL via iTunes/RSS."""
    import xml.etree.ElementTree as ET

    logger.info(f"Searching iTunes for RSS feed: {show_name}")
    try:
        # Search iTunes API for the podcast
        resp = requests.get(
            "https://itunes.apple.com/search",
            params={"term": show_name, "media": "podcast", "limit": 5},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        for pod in results:
            feed_url = pod.get("feedUrl")
            if not feed_url:
                continue

            logger.info(f"Checking RSS feed: {feed_url}")
            try:
                feed_resp = requests.get(feed_url, timeout=20, headers={
                    "User-Agent": "Mozilla/5.0",
                })
                feed_resp.raise_for_status()
                root = ET.fromstring(feed_resp.content)

                # Search episodes for a title match
                ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
                for item in root.findall(".//item"):
                    item_title = item.findtext("title", "")
                    # Fuzzy match: check if most words from the episode title appear
                    title_words = set(episode_title.lower().split())
                    item_words = set(item_title.lower().split())
                    overlap = len(title_words & item_words)
                    if overlap >= max(2, len(title_words) * 0.5):
                        enclosure = item.find("enclosure")
                        if enclosure is not None:
                            audio_url = enclosure.get("url", "")
                            if audio_url:
                                pub_date = item.findtext("pubDate", "")
                                duration_text = item.findtext("itunes:duration", "", ns)
                                logger.info(f"Found RSS audio match: {item_title}")
                                return {
                                    "audio_url": audio_url,
                                    "title": item_title,
                                    "show": pod.get("collectionName", show_name),
                                    "pub_date": pub_date,
                                    "duration_text": duration_text,
                                }
            except Exception as e:
                logger.warning(f"RSS feed parse failed for {feed_url}: {e}")
                continue
    except Exception as e:
        logger.warning(f"iTunes search failed: {e}")
    return None


def _transcribe_audio_assemblyai(audio_url: str) -> list:
    """Transcribe an audio URL using AssemblyAI. Returns segments."""
    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError("No ASSEMBLYAI_API_KEY set — cannot transcribe audio")

    headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}
    api_base = "https://api.assemblyai.com/v2"

    logger.info(f"Submitting audio to AssemblyAI: {audio_url[:80]}...")
    # Submit transcription job
    resp = requests.post(
        f"{api_base}/transcript",
        headers=headers,
        json={"audio_url": audio_url, "language_code": "en"},
        timeout=30,
    )
    resp.raise_for_status()
    transcript_id = resp.json()["id"]
    logger.info(f"AssemblyAI job submitted: {transcript_id}")

    # Poll for completion
    import time
    for _ in range(180):  # up to 30 minutes
        time.sleep(10)
        resp = requests.get(f"{api_base}/transcript/{transcript_id}", headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        status = data["status"]
        if status == "completed":
            logger.info(f"AssemblyAI transcription complete: {len(data.get('words', []))} words")
            # Convert to segments (group by sentences)
            segments = []
            for utt in data.get("utterances", []) or []:
                segments.append({
                    "start": utt["start"] / 1000.0,
                    "text": utt["text"],
                })
            # Fallback: if no utterances, use words grouped in chunks
            if not segments and data.get("words"):
                chunk = []
                chunk_start = 0
                for w in data["words"]:
                    if not chunk:
                        chunk_start = w["start"] / 1000.0
                    chunk.append(w["text"])
                    if w.get("text", "").endswith((".", "!", "?")) or len(chunk) >= 30:
                        segments.append({"start": chunk_start, "text": " ".join(chunk)})
                        chunk = []
                if chunk:
                    segments.append({"start": chunk_start, "text": " ".join(chunk)})
            # Last fallback: single block
            if not segments and data.get("text"):
                segments.append({"start": 0, "text": data["text"]})
            return segments
        elif status == "error":
            raise RuntimeError(f"AssemblyAI error: {data.get('error', 'unknown')}")
        logger.info(f"AssemblyAI status: {status}...")

    raise RuntimeError("AssemblyAI transcription timed out")


def _parse_duration_text(text: str) -> float:
    """Parse duration like '01:23:45' or '3600' into minutes."""
    if not text:
        return 0
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
        elif len(parts) == 2:
            return int(parts[0]) + int(parts[1]) / 60
    try:
        return int(text) / 60  # seconds
    except ValueError:
        return 0


def transcribe_spotify(url: str) -> TranscriptResult:
    logger.info(f"Processing Spotify URL: {url}")
    meta = _fetch_spotify_meta(url)
    title = meta["title"]
    show_name = meta["show_name"]

    if not title:
        raise ValueError(
            "Could not read episode info from Spotify. "
            "Please check the URL is correct."
        )

    # Strategy 1: Find on YouTube (fastest — uses existing captions)
    logger.info("Spotify strategy 1: search YouTube...")
    yt_url = _search_youtube(f"{show_name} {title}")
    if yt_url:
        try:
            result = transcribe_youtube(yt_url)
            result.url = url
            return result
        except Exception as e:
            logger.warning(f"YouTube transcription failed: {e}")

    # Strategy 2: Find RSS feed audio → transcribe with AssemblyAI
    logger.info("Spotify strategy 2: RSS feed + AssemblyAI...")
    rss_info = _find_rss_audio(show_name, title)
    if rss_info and ASSEMBLYAI_API_KEY:
        try:
            segments = _transcribe_audio_assemblyai(rss_info["audio_url"])
            if segments:
                duration_min = _parse_duration_text(rss_info.get("duration_text", ""))
                if not duration_min and segments:
                    duration_min = segments[-1]["start"] / 60.0

                # Parse pub_date
                pub_date = ""
                raw_date = rss_info.get("pub_date", "")
                if raw_date:
                    try:
                        from email.utils import parsedate_to_datetime
                        dt = parsedate_to_datetime(raw_date)
                        pub_date = dt.strftime("%Y-%m-%d")
                    except Exception:
                        pass

                return TranscriptResult(
                    title=rss_info.get("title", title),
                    source=rss_info.get("show", show_name),
                    date=pub_date,
                    duration_minutes=duration_min,
                    url=url,
                    segments=segments,
                    method="assemblyai",
                )
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed: {e}")

    # Nothing worked
    errors = []
    if not yt_url:
        errors.append("not found on YouTube")
    if not rss_info:
        errors.append("RSS feed not found")
    elif not ASSEMBLYAI_API_KEY:
        errors.append("no ASSEMBLYAI_API_KEY for audio transcription")

    raise RuntimeError(
        f"Could not transcribe this Spotify episode.\n"
        f"Show: {show_name}\nEpisode: {title}\n"
        f"Tried: {', '.join(errors)}"
    )


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
