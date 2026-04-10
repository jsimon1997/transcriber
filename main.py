import logging

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

app = FastAPI(title="Transcriber")

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
  <h1>Transcriber</h1>
  <p class="subtitle">Paste a YouTube or Spotify podcast URL and get a timestamped transcript.</p>

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
    <pre id="transcript"></pre>
  </div>

  <p class="footer-note" id="footer-note"></p>
</div>

<script>
let currentTranscript = '';
let currentTitle = '';

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
  statusText.textContent = 'Processing\u2026 (may take a few minutes for long content)';
  statusEl.className = '';
  resultBox.style.display = 'none';

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
      document.getElementById('transcript').textContent = data.transcript;
      document.getElementById('result-meta').innerHTML =
        '<strong>' + escHtml(data.title) + '</strong>'
        + ' &middot; ' + escHtml(data.source)
        + (data.date ? ' &middot; ' + escHtml(data.date) : '');
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
        return result
    except ValueError as e:
        logger.warning(f"Bad request: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
