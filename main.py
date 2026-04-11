import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from transcribe import transcribe_url  # noqa: E402 — after load_dotenv

# ---------------------------------------------------------------------------
# AI Insights via Claude API
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


def generate_insights(transcript: str, title: str, url: str) -> list[dict]:
    """Call Claude to extract 5 key insights with timestamps from a transcript."""
    if not ANTHROPIC_API_KEY:
        logger.warning("No ANTHROPIC_API_KEY set — skipping insights")
        return []

    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Truncate very long transcripts to ~60k chars to stay within context
    max_chars = 60_000
    trunc = transcript[:max_chars]
    if len(transcript) > max_chars:
        trunc += "\n\n[transcript truncated for summarisation]"

    prompt = (
        f"Here is a transcript of \"{title}\":\n\n"
        f"{trunc}\n\n"
        "Based on this transcript, provide exactly 5 key insights, learnings, "
        "or takeaways. Focus on the most valuable, surprising, or actionable "
        "ideas discussed.\n\n"
        "Format EXACTLY like this (no preamble, no closing remarks):\n"
        "1. [HH:MM:SS] **Bold insight headline here**\n"
        "   - Supporting detail or example (1 sentence)\n"
        "   - Another supporting detail (1 sentence)\n"
        "2. [HH:MM:SS] **Next insight headline**\n"
        "   - Supporting detail\n"
        "...\n\n"
        "Rules:\n"
        "- Exactly 5 numbered insights\n"
        "- Each insight MUST start with [HH:MM:SS] — the timestamp from the transcript where this topic is discussed\n"
        "- Each insight has a bold headline (concise, ~10 words)\n"
        "- Each insight has 1-2 supporting bullet points with concrete details from the discussion\n"
        "- Supporting bullets should add real substance — specific examples, data, quotes, or context\n"
        "- Use the timestamps from the transcript (e.g. [00:15:32]) to reference where the insight appears"
    )

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        logger.info(f"Raw insights response:\n{raw}")

        # Parse into structured insights: list of {headline, bullets, timestamp}
        insights = []
        current = None
        for line in raw.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # Numbered headline: "1. [00:15:32] **Something**" or "1. **Something**"
            m = re.match(r"^\d+[\.\)]\s*(.*)", stripped)
            if m:
                if current:
                    insights.append(current)
                rest = m.group(1).strip()
                # Extract timestamp [HH:MM:SS]
                ts_match = re.match(r"\[(\d{1,2}:\d{2}:\d{2})\]\s*(.*)", rest)
                timestamp = ""
                if ts_match:
                    timestamp = ts_match.group(1)
                    rest = ts_match.group(2).strip()
                # Remove bold markdown for clean text
                headline = re.sub(r"\*\*(.*?)\*\*", r"\1", rest)
                current = {"headline": headline, "bullets": [], "timestamp": timestamp}
            # Sub-bullet: "- Something" or "• Something"
            elif re.match(r"^[-•–]\s+", stripped) and current is not None:
                bullet = re.sub(r"^[-•–]\s+", "", stripped)
                current["bullets"].append(bullet)
        if current:
            insights.append(current)

        return insights
    except Exception as e:
        logger.error(f"Insights generation failed: {e}", exc_info=True)
        return []


import re  # noqa: E402 — needed for generate_insights

from db import init_db, save_episode, get_all_episodes, get_episode  # noqa: E402

app = FastAPI(title="Transcriber")


@app.on_event("startup")
def startup():
    init_db()

# ---------------------------------------------------------------------------
# Inline HTML page
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Transcriber</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f3f4f6;
    color: #111;
    padding: 2.5rem 1rem;
    min-height: 100vh;
  }
  .container { max-width: 820px; margin: 0 auto; }
  h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.3rem; }
  .subtitle { color: #666; font-size: 0.88rem; margin-bottom: 1.5rem; }
  .input-row { display: flex; gap: 0.5rem; margin-bottom: 0.6rem; }
  input[type=text] {
    flex: 1;
    padding: 0.65rem 0.9rem;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    font-size: 0.95rem;
    outline: none;
    transition: border-color 0.15s;
  }
  input[type=text]:focus { border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,0.15); }
  button.primary {
    padding: 0.65rem 1.3rem;
    background: #2563eb;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.15s;
  }
  button.primary:hover:not(:disabled) { background: #1d4ed8; }
  button.primary:disabled { background: #93c5fd; cursor: not-allowed; }
  #status {
    font-size: 0.88rem;
    min-height: 1.4rem;
    color: #555;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  #status.error { color: #dc2626; }
  .spinner {
    display: none;
    width: 14px; height: 14px;
    border: 2px solid #ccc;
    border-top-color: #2563eb;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Result container */
  #result-box {
    display: none;
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    overflow: hidden;
  }
  .result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #e5e7eb;
    background: #f9fafb;
  }
  .result-meta { font-size: 0.82rem; color: #555; }
  .result-meta strong { color: #111; }
  .action-btns { display: flex; gap: 0.4rem; }
  button.sec {
    padding: 0.35rem 0.75rem;
    background: #fff;
    color: #374151;
    border: 1px solid #d1d5db;
    border-radius: 5px;
    font-size: 0.82rem;
    cursor: pointer;
    transition: background 0.1s;
  }
  button.sec:hover { background: #f3f4f6; }

  /* Insights section */
  #insights-section {
    padding: 1.2rem 1rem 0.6rem;
    border-bottom: 1px solid #e5e7eb;
  }
  #insights-section h2 {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: #1e3a5f;
  }
  .insight-item {
    position: relative;
    padding: 0 0 1rem 2.4rem;
    margin-bottom: 0.2rem;
  }
  .insight-item:last-child { padding-bottom: 0.5rem; }
  .insight-num {
    position: absolute;
    left: 0;
    top: 0;
    width: 1.6rem;
    height: 1.6rem;
    background: #2563eb;
    color: #fff;
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .insight-headline {
    font-size: 0.92rem;
    font-weight: 600;
    color: #111;
    line-height: 1.6;
    margin-bottom: 0.3rem;
  }
  .insight-timestamp {
    display: inline-block;
    font-size: 0.76rem;
    font-weight: 600;
    color: #2563eb;
    background: #eef2ff;
    padding: 0.1rem 0.45rem;
    border-radius: 4px;
    margin-right: 0.4rem;
    text-decoration: none;
    cursor: pointer;
    vertical-align: middle;
  }
  .insight-timestamp:hover { background: #dbeafe; }
  .insight-bullets {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .insight-bullets li {
    position: relative;
    padding-left: 0.9rem;
    font-size: 0.84rem;
    line-height: 1.55;
    color: #4b5563;
    margin-bottom: 0.15rem;
  }
  .insight-bullets li::before {
    content: "–";
    position: absolute;
    left: 0;
    color: #9ca3af;
  }

  /* Toggle transcript button */
  #toggle-transcript-btn {
    display: block;
    width: 100%;
    padding: 0.75rem 1rem;
    background: #f9fafb;
    border: none;
    border-bottom: 1px solid #e5e7eb;
    font-size: 0.88rem;
    font-weight: 600;
    color: #2563eb;
    cursor: pointer;
    text-align: center;
    transition: background 0.15s;
  }
  #toggle-transcript-btn:hover { background: #eef2ff; }

  /* Transcript */
  #transcript-section { display: none; }
  #transcript {
    padding: 1.2rem 1rem;
    white-space: pre-wrap;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    max-height: 62vh;
    overflow-y: auto;
    color: #1f2937;
  }
  .footer-note { margin-top: 0.8rem; font-size: 0.78rem; color: #9ca3af; }
</style>
</head>
<body>
<div class="container">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.3rem">
    <h1>Transcriber</h1>
    <a href="/feed" style="font-size:0.88rem;color:#2563eb;text-decoration:none;font-weight:600">My Feed</a>
  </div>
  <p class="subtitle">Paste a YouTube or Spotify podcast URL and get key insights + a timestamped transcript.</p>

  <div class="input-row">
    <input type="text" id="url-input"
      placeholder="https://www.youtube.com/watch?v=... or https://open.spotify.com/episode/..." />
    <button class="primary" id="submit-btn" onclick="startTranscription()">Transcribe</button>
  </div>

  <div id="status">
    <span class="spinner" id="spinner"></span>
    <span id="status-text"></span>
  </div>

  <div id="result-box">
    <div class="result-header">
      <span class="result-meta" id="result-meta"></span>
      <div class="action-btns">
        <button class="sec" onclick="copyTranscript()">Copy</button>
        <button class="sec" onclick="downloadTranscript()">Download .md</button>
      </div>
    </div>

    <!-- Key Insights -->
    <div id="insights-section">
      <h2>Key Takeaways</h2>
      <div id="insights-list"></div>
    </div>

    <!-- Toggle Transcript -->
    <button id="toggle-transcript-btn" onclick="toggleTranscript()">View Full Transcript</button>

    <!-- Full Transcript (hidden by default) -->
    <div id="transcript-section">
      <pre id="transcript"></pre>
    </div>
  </div>

  <p class="footer-note" id="footer-note"></p>
</div>

<script>
let currentTranscript = '';
let currentTitle = '';
let currentUrl = '';
let transcriptVisible = false;

async function startTranscription() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) return;

  const btn = document.getElementById('submit-btn');
  const spinner = document.getElementById('spinner');
  const statusText = document.getElementById('status-text');
  const statusEl = document.getElementById('status');
  const resultBox = document.getElementById('result-box');

  btn.disabled = true;
  spinner.style.display = 'block';
  statusText.textContent = 'Processing... (may take a few minutes for long content)';
  statusEl.className = '';
  resultBox.style.display = 'none';
  transcriptVisible = false;
  document.getElementById('transcript-section').style.display = 'none';
  document.getElementById('toggle-transcript-btn').textContent = 'View Full Transcript';

  try {
    const resp = await fetch('/transcribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      statusText.textContent = data.error || 'Unknown error.';
      statusEl.className = 'error';
    } else {
      currentTranscript = data.transcript;
      currentTitle = data.title || 'transcript';
      currentUrl = data.url || '';

      // Meta header
      document.getElementById('result-meta').innerHTML =
        '<strong>' + escHtml(data.title) + '</strong>'
        + ' &middot; ' + escHtml(data.source)
        + (data.date ? ' &middot; ' + escHtml(data.date) : '');

      // Insights
      const insightsList = document.getElementById('insights-list');
      const insightsSection = document.getElementById('insights-section');
      insightsList.innerHTML = '';
      if (data.insights && data.insights.length > 0) {
        data.insights.forEach((insight, i) => {
          const div = document.createElement('div');
          div.className = 'insight-item';

          const num = document.createElement('span');
          num.className = 'insight-num';
          num.textContent = i + 1;
          div.appendChild(num);

          const headline = document.createElement('div');
          headline.className = 'insight-headline';

          // Add timestamp link if available
          if (insight.timestamp) {
            const tsLink = document.createElement('a');
            tsLink.className = 'insight-timestamp';
            tsLink.textContent = insight.timestamp;
            tsLink.href = buildTimestampUrl(currentUrl, insight.timestamp);
            tsLink.target = '_blank';
            tsLink.rel = 'noopener';
            headline.appendChild(tsLink);
          }

          headline.appendChild(document.createTextNode(insight.headline || insight));
          div.appendChild(headline);

          if (insight.bullets && insight.bullets.length > 0) {
            const ul = document.createElement('ul');
            ul.className = 'insight-bullets';
            insight.bullets.forEach(b => {
              const li = document.createElement('li');
              li.textContent = b;
              ul.appendChild(li);
            });
            div.appendChild(ul);
          }

          insightsList.appendChild(div);
        });
        insightsSection.style.display = 'block';
      } else {
        insightsSection.style.display = 'none';
      }

      // Transcript (hidden initially)
      document.getElementById('transcript').textContent = data.transcript;

      resultBox.style.display = 'block';
      statusText.textContent = 'Done.';
      document.getElementById('footer-note').textContent = 'Method: ' + data.method;
    }
  } catch (e) {
    statusText.textContent = 'Request failed: ' + e.message;
    statusEl.className = 'error';
  } finally {
    btn.disabled = false;
    spinner.style.display = 'none';
  }
}

function buildTimestampUrl(url, timestamp) {
  // Convert HH:MM:SS to total seconds
  const parts = timestamp.split(':').map(Number);
  let seconds = 0;
  if (parts.length === 3) seconds = parts[0] * 3600 + parts[1] * 60 + parts[2];
  else if (parts.length === 2) seconds = parts[0] * 60 + parts[1];
  else seconds = parts[0];

  // YouTube URL with timestamp
  if (url.includes('youtube.com') || url.includes('youtu.be')) {
    const vidMatch = url.match(/(?:v=|youtu\.be\/)([A-Za-z0-9_-]{11})/);
    if (vidMatch) return 'https://www.youtube.com/watch?v=' + vidMatch[1] + '&t=' + seconds + 's';
  }
  return url;
}

function toggleTranscript() {
  transcriptVisible = !transcriptVisible;
  document.getElementById('transcript-section').style.display = transcriptVisible ? 'block' : 'none';
  document.getElementById('toggle-transcript-btn').textContent =
    transcriptVisible ? 'Hide Full Transcript' : 'View Full Transcript';
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function copyTranscript() {
  navigator.clipboard.writeText(currentTranscript).then(() => {
    const btn = event.target;
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 1500);
  });
}

function downloadTranscript() {
  const blob = new Blob([currentTranscript], { type: 'text/markdown' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = currentTitle.replace(/[^a-z0-9]/gi, '_').slice(0, 60) + '.md';
  a.click();
}

document.getElementById('url-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') startTranscription();
});

</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class TranscribeRequest(BaseModel):
    url: str


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML



@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    logger.info(f"Request: {req.url}")
    try:
        result = transcribe_url(req.url)
        logger.info(f"Done: {result['title']} [{result['method']}]")
        # Generate AI insights
        insights = generate_insights(result["transcript"], result["title"], result["url"])
        result["insights"] = insights
        logger.info(f"Generated {len(insights)} insights")
        # Save to database
        ep_id = save_episode(result, insights)
        if ep_id:
            logger.info(f"Saved episode id={ep_id}")
        return result
    except ValueError as e:
        logger.warning(f"Bad request: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})



# ---------------------------------------------------------------------------
# Feed API + page
# ---------------------------------------------------------------------------

@app.get("/api/feed")
def api_feed():
    return get_all_episodes()


@app.get("/api/episode/{episode_id}")
def api_episode(episode_id: int):
    ep = get_episode(episode_id)
    if not ep:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return ep


FEED_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>My Podcast Feed</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f3f4f6;
    color: #111;
    padding: 2rem 1rem;
    min-height: 100vh;
  }
  .container { max-width: 820px; margin: 0 auto; }
  .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem; }
  h1 { font-size: 1.4rem; font-weight: 700; }
  a.nav-link {
    font-size: 0.88rem;
    color: #2563eb;
    text-decoration: none;
    font-weight: 600;
  }
  a.nav-link:hover { text-decoration: underline; }
  .empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #9ca3af;
    font-size: 0.95rem;
  }
  .episode-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    margin-bottom: 1rem;
    overflow: hidden;
    transition: box-shadow 0.15s;
  }
  .episode-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .ep-header {
    padding: 0.9rem 1rem;
    border-bottom: 1px solid #f3f4f6;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.8rem;
  }
  .ep-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #111;
    line-height: 1.4;
  }
  .ep-meta {
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 0.2rem;
  }
  .ep-date-badge {
    font-size: 0.72rem;
    background: #f3f4f6;
    color: #6b7280;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .ep-insights {
    padding: 0.6rem 1rem 0.8rem;
  }
  .insight-row {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 0.35rem 0;
  }
  .insight-num-sm {
    width: 1.3rem;
    height: 1.3rem;
    background: #2563eb;
    color: #fff;
    border-radius: 50%;
    font-size: 0.65rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 0.1rem;
  }
  .insight-text {
    font-size: 0.85rem;
    line-height: 1.5;
    color: #1f2937;
  }
  .insight-text .ts-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    color: #2563eb;
    background: #eef2ff;
    padding: 0.05rem 0.35rem;
    border-radius: 3px;
    margin-right: 0.3rem;
    text-decoration: none;
  }
  .insight-text .ts-badge:hover { background: #dbeafe; }
  .insight-sub {
    font-size: 0.8rem;
    color: #6b7280;
    padding-left: 1.8rem;
    line-height: 1.45;
  }
  .insight-sub::before { content: "- "; color: #d1d5db; }
  .ep-actions {
    padding: 0 1rem 0.7rem;
    display: flex;
    gap: 0.5rem;
  }
  .ep-actions a, .ep-actions button {
    font-size: 0.78rem;
    color: #2563eb;
    text-decoration: none;
    cursor: pointer;
    background: none;
    border: none;
    font-weight: 600;
  }
  .ep-actions a:hover, .ep-actions button:hover { text-decoration: underline; }
  .transcript-expand {
    display: none;
    padding: 0 1rem 1rem;
  }
  .transcript-expand pre {
    white-space: pre-wrap;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.78rem;
    line-height: 1.6;
    max-height: 50vh;
    overflow-y: auto;
    background: #f9fafb;
    padding: 0.8rem;
    border-radius: 6px;
    color: #374151;
  }
  .loading { text-align: center; padding: 2rem; color: #9ca3af; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>My Podcast Feed</h1>
    <a class="nav-link" href="/">+ Transcribe New</a>
  </div>
  <div id="feed">
    <div class="loading">Loading your feed...</div>
  </div>
</div>
<script>
function buildTsUrl(url, timestamp) {
  if (!timestamp) return url;
  const parts = timestamp.split(':').map(Number);
  let s = 0;
  if (parts.length === 3) s = parts[0]*3600 + parts[1]*60 + parts[2];
  else if (parts.length === 2) s = parts[0]*60 + parts[1];
  const m = url.match(/(?:v=|youtu\\.be\\/)([A-Za-z0-9_-]{11})/);
  if (m) return 'https://www.youtube.com/watch?v=' + m[1] + '&t=' + s + 's';
  return url;
}
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function loadFeed() {
  const container = document.getElementById('feed');
  try {
    const resp = await fetch('/api/feed');
    const episodes = await resp.json();
    if (!episodes.length) {
      container.innerHTML = '<div class="empty-state">No podcasts yet. <a href="/">Transcribe your first one!</a></div>';
      return;
    }
    container.innerHTML = episodes.map((ep, idx) => {
      const insights = ep.insights || [];
      const insightsHtml = insights.map((ins, i) => {
        const ts = ins.timestamp
          ? '<a class="ts-badge" href="' + esc(buildTsUrl(ep.url, ins.timestamp)) + '" target="_blank">' + esc(ins.timestamp) + '</a>'
          : '';
        const bulletsHtml = (ins.bullets || []).map(b => '<div class="insight-sub">' + esc(b) + '</div>').join('');
        return '<div class="insight-row">'
          + '<span class="insight-num-sm">' + (i+1) + '</span>'
          + '<div><div class="insight-text">' + ts + esc(ins.headline || ins) + '</div>' + bulletsHtml + '</div>'
          + '</div>';
      }).join('');

      const dateStr = ep.created_at ? new Date(ep.created_at).toLocaleDateString('en-US', {month:'short', day:'numeric', year:'numeric'}) : '';

      return '<div class="episode-card">'
        + '<div class="ep-header"><div>'
        + '<div class="ep-title">' + esc(ep.title) + '</div>'
        + '<div class="ep-meta">' + esc(ep.source) + (ep.duration_minutes ? ' &middot; ' + Math.round(ep.duration_minutes) + ' min' : '') + '</div>'
        + '</div>'
        + (dateStr ? '<span class="ep-date-badge">' + esc(dateStr) + '</span>' : '')
        + '</div>'
        + (insightsHtml ? '<div class="ep-insights">' + insightsHtml + '</div>' : '')
        + '<div class="ep-actions">'
        + '<a href="' + esc(ep.url) + '" target="_blank">Open Video</a>'
        + '<button onclick="toggleTx(this,' + ep.id + ')">View Transcript</button>'
        + '</div>'
        + '<div class="transcript-expand" id="tx-' + ep.id + '"><pre>Loading...</pre></div>'
        + '</div>';
    }).join('');
  } catch(e) {
    container.innerHTML = '<div class="empty-state">Failed to load feed: ' + e.message + '</div>';
  }
}

async function toggleTx(btn, id) {
  const el = document.getElementById('tx-' + id);
  if (el.style.display === 'block') {
    el.style.display = 'none';
    btn.textContent = 'View Transcript';
    return;
  }
  el.style.display = 'block';
  btn.textContent = 'Hide Transcript';
  if (el.dataset.loaded) return;
  try {
    const resp = await fetch('/api/episode/' + id);
    const ep = await resp.json();
    el.querySelector('pre').textContent = ep.transcript || 'No transcript available';
    el.dataset.loaded = '1';
  } catch(e) {
    el.querySelector('pre').textContent = 'Failed to load: ' + e.message;
  }
}

loadFeed();
</script>
</body>
</html>
"""


@app.get("/feed", response_class=HTMLResponse)
def feed_page():
    return HTMLResponse(FEED_HTML)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
