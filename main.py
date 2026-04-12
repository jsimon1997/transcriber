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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    min-height: 100vh;
  }

  /* Hero header */
  .hero {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-bottom: 1px solid #1e293b;
    padding: 2rem 1rem 1.5rem;
    position: sticky;
    top: 0;
    z-index: 10;
    backdrop-filter: blur(12px);
  }
  .hero-inner { max-width: 880px; margin: 0 auto; }
  .hero h1 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
  }
  .hero h1 span { color: #818cf8; }

  /* Add bar */
  .add-bar {
    display: flex;
    gap: 0.5rem;
  }
  .add-bar input {
    flex: 1;
    padding: 0.7rem 1rem;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    font-size: 0.9rem;
    color: #e2e8f0;
    outline: none;
    font-family: inherit;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  .add-bar input::placeholder { color: #64748b; }
  .add-bar input:focus { border-color: #818cf8; box-shadow: 0 0 0 3px rgba(129,140,248,0.2); }
  .add-bar button {
    padding: 0.7rem 1.5rem;
    background: linear-gradient(135deg, #6366f1, #818cf8);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    font-family: inherit;
    transition: opacity 0.15s;
  }
  .add-bar button:hover:not(:disabled) { opacity: 0.9; }
  .add-bar button:disabled { opacity: 0.5; cursor: not-allowed; }
  #add-status {
    font-size: 0.8rem;
    min-height: 1.2rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  #add-status.error { color: #f87171; }
  .add-spinner {
    display: none;
    width: 12px; height: 12px;
    border: 2px solid #475569;
    border-top-color: #818cf8;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Feed container */
  .container { max-width: 880px; margin: 0 auto; padding: 1.5rem 1rem 3rem; }
  .empty-state {
    text-align: center;
    padding: 4rem 1rem;
    color: #64748b;
    font-size: 0.95rem;
  }
  .empty-state a { color: #818cf8; }

  /* Episode card */
  .episode-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .episode-card:hover {
    border-color: #475569;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
  }

  /* Thumbnail */
  .ep-thumb-wrap {
    position: relative;
    overflow: hidden;
  }
  .ep-thumb {
    width: 100%;
    aspect-ratio: 16/8;
    object-fit: cover;
    display: block;
    background: #334155;
    transition: transform 0.3s;
  }
  .episode-card:hover .ep-thumb { transform: scale(1.02); }
  .ep-thumb-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 50%;
    background: linear-gradient(transparent, rgba(15,23,42,0.85));
    pointer-events: none;
  }
  .ep-date-float {
    position: absolute;
    top: 0.75rem;
    right: 0.75rem;
    background: rgba(15,23,42,0.75);
    backdrop-filter: blur(6px);
    color: #e2e8f0;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    letter-spacing: 0.02em;
  }
  .ep-duration-float {
    position: absolute;
    bottom: 0.75rem;
    right: 0.75rem;
    background: rgba(0,0,0,0.7);
    color: #e2e8f0;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
  }

  .ep-header {
    padding: 1rem 1.1rem 0.5rem;
  }
  .ep-title {
    font-size: 1rem;
    font-weight: 600;
    color: #f1f5f9;
    line-height: 1.4;
    letter-spacing: -0.01em;
  }
  .ep-meta {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.25rem;
  }

  /* Insights */
  .ep-insights { padding: 0.3rem 1.1rem 0.5rem; }
  .insight-row {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    padding: 0.35rem 0;
  }
  .insight-num-sm {
    width: 1.25rem;
    height: 1.25rem;
    background: linear-gradient(135deg, #6366f1, #818cf8);
    color: #fff;
    border-radius: 50%;
    font-size: 0.62rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 0.15rem;
  }
  .insight-content { flex: 1; min-width: 0; }
  .insight-headline {
    font-size: 0.84rem;
    line-height: 1.5;
    color: #cbd5e1;
    cursor: pointer;
    display: flex;
    align-items: baseline;
    gap: 0.35rem;
    user-select: none;
  }
  .insight-headline:hover { color: #f1f5f9; }
  .insight-headline .ts-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    color: #a5b4fc;
    background: rgba(99,102,241,0.15);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    text-decoration: none;
    flex-shrink: 0;
    border: 1px solid rgba(99,102,241,0.25);
  }
  .insight-headline .ts-badge:hover { background: rgba(99,102,241,0.25); }
  .insight-toggle {
    color: #64748b;
    font-size: 0.65rem;
    margin-left: 0.15rem;
    transition: transform 0.2s;
    display: inline-block;
  }
  .insight-toggle.open { transform: rotate(90deg); }
  .insight-bullets {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.25s ease, padding 0.25s ease;
    padding-top: 0;
  }
  .insight-bullets.open {
    max-height: 10rem;
    padding-top: 0.25rem;
  }
  .insight-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    padding-left: 0.1rem;
    line-height: 1.5;
    margin-bottom: 0.1rem;
  }
  .insight-sub::before { content: "\\2013\\00a0"; color: #475569; }

  /* Actions bar */
  .ep-actions {
    padding: 0.4rem 1.1rem 0.9rem;
    display: flex;
    gap: 1rem;
    border-top: 1px solid #1e293b;
    margin-top: 0.2rem;
  }
  .ep-actions a, .ep-actions button {
    font-size: 0.78rem;
    color: #818cf8;
    text-decoration: none;
    cursor: pointer;
    background: none;
    border: none;
    font-weight: 600;
    font-family: inherit;
    padding: 0;
  }
  .ep-actions a:hover, .ep-actions button:hover { color: #a5b4fc; }
  .transcript-expand {
    display: none;
    padding: 0 1.1rem 1rem;
  }
  .transcript-expand pre {
    white-space: pre-wrap;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    font-size: 0.76rem;
    line-height: 1.6;
    max-height: 50vh;
    overflow-y: auto;
    background: #0f172a;
    padding: 1rem;
    border-radius: 8px;
    color: #94a3b8;
    border: 1px solid #1e293b;
  }
  .loading { text-align: center; padding: 3rem; color: #64748b; }
</style>
</head>
<body>
<div class="hero">
  <div class="hero-inner">
    <h1>My <span>Podcast</span> Feed</h1>
    <div class="add-bar">
      <input type="text" id="add-url" placeholder="Paste a YouTube or Spotify URL..." />
      <button id="add-btn" onclick="addPodcast()">Add</button>
    </div>
    <div id="add-status">
      <span class="add-spinner" id="add-spinner"></span>
      <span id="add-status-text"></span>
    </div>
  </div>
</div>
<div class="container">
  <div id="feed">
    <div class="loading">Loading your feed...</div>
  </div>
</div>
<script>
function getVideoId(url) {
  const m = url.match(/(?:v=|youtu\\.be\\/)([A-Za-z0-9_-]{11})/);
  return m ? m[1] : null;
}
function buildTsUrl(url, timestamp) {
  if (!timestamp) return url;
  const parts = timestamp.split(':').map(Number);
  let s = 0;
  if (parts.length === 3) s = parts[0]*3600 + parts[1]*60 + parts[2];
  else if (parts.length === 2) s = parts[0]*60 + parts[1];
  const vid = getVideoId(url);
  if (vid) return 'https://www.youtube.com/watch?v=' + vid + '&t=' + s + 's';
  return url;
}
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function toggleBullets(el) {
  const bullets = el.closest('.insight-content').querySelector('.insight-bullets');
  const arrow = el.querySelector('.insight-toggle');
  if (!bullets) return;
  bullets.classList.toggle('open');
  arrow.classList.toggle('open');
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
      const vid = getVideoId(ep.url);
      const thumbUrl = vid ? 'https://img.youtube.com/vi/' + vid + '/maxresdefault.jpg' : '';

      const insightsHtml = insights.map((ins, i) => {
        const ts = ins.timestamp
          ? '<a class="ts-badge" href="' + esc(buildTsUrl(ep.url, ins.timestamp)) + '" target="_blank" onclick="event.stopPropagation()">' + esc(ins.timestamp) + '</a>'
          : '';
        const hasBullets = ins.bullets && ins.bullets.length > 0;
        const arrow = hasBullets ? '<span class="insight-toggle">&#9654;</span>' : '';
        const bulletsHtml = hasBullets
          ? '<div class="insight-bullets">' + ins.bullets.map(b => '<div class="insight-sub">' + esc(b) + '</div>').join('') + '</div>'
          : '';
        return '<div class="insight-row">'
          + '<span class="insight-num-sm">' + (i+1) + '</span>'
          + '<div class="insight-content">'
          + '<div class="insight-headline" onclick="toggleBullets(this)">' + ts + esc(ins.headline || ins) + ' ' + arrow + '</div>'
          + bulletsHtml
          + '</div>'
          + '</div>';
      }).join('');

      // Use video release date if available, otherwise created_at
      const rawDate = ep.video_date || ep.created_at;
      const dateStr = rawDate ? new Date(rawDate).toLocaleDateString('en-US', {month:'short', day:'numeric', year:'numeric'}) : '';
      const durationStr = ep.duration_minutes ? Math.round(ep.duration_minutes) + ' min' : '';

      const thumbHtml = thumbUrl
        ? '<a href="' + esc(ep.url) + '" target="_blank" class="ep-thumb-wrap">'
          + '<img class="ep-thumb" src="' + esc(thumbUrl) + '" alt="" loading="lazy">'
          + '<div class="ep-thumb-overlay"></div>'
          + (dateStr ? '<span class="ep-date-float">' + esc(dateStr) + '</span>' : '')
          + (durationStr ? '<span class="ep-duration-float">' + esc(durationStr) + '</span>' : '')
          + '</a>'
        : '';

      return '<div class="episode-card">'
        + thumbHtml
        + '<div class="ep-header">'
        + '<div class="ep-title">' + esc(ep.title) + '</div>'
        + '<div class="ep-meta">' + esc(ep.source) + '</div>'
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

async function addPodcast() {
  const url = document.getElementById('add-url').value.trim();
  if (!url) return;
  const btn = document.getElementById('add-btn');
  const spinner = document.getElementById('add-spinner');
  const statusText = document.getElementById('add-status-text');
  const statusEl = document.getElementById('add-status');

  btn.disabled = true;
  spinner.style.display = 'block';
  statusText.textContent = 'Transcribing... (may take a few minutes)';
  statusEl.className = '';

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
      statusText.textContent = 'Added! Refreshing feed...';
      document.getElementById('add-url').value = '';
      await loadFeed();
      statusText.textContent = '';
    }
  } catch(e) {
    statusText.textContent = 'Failed: ' + e.message;
    statusEl.className = 'error';
  } finally {
    btn.disabled = false;
    spinner.style.display = 'none';
  }
}

document.getElementById('add-url').addEventListener('keydown', e => {
  if (e.key === 'Enter') addPodcast();
});

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
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
