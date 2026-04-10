# Transcriber

Paste a YouTube or Spotify podcast URL, get a timestamped transcript.

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on your PATH (required for audio conversion)

## Setup

```bash
cd transcriber

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt

# Copy env file and edit if needed
cp .env.example .env
```

## Run

```bash
python main.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Configuration

Edit `.env` to change the Whisper model:

| Model  | Size   | Speed  | Quality |
|--------|--------|--------|---------|
| base   | ~74MB  | fast   | good    |
| small  | ~244MB | medium | better  |
| medium | ~769MB | slow   | great   |
| large  | ~1.5GB | slowest| best    |

Or set inline: `WHISPER_MODEL=small python main.py`

## How it works

**YouTube URL** (`youtube.com` / `youtu.be`):
1. Tries existing captions via `youtube-transcript-api` — instant if found
2. Falls back to downloading audio + running Whisper locally

**Spotify episode URL** (`open.spotify.com/episode/...`):
1. Scrapes episode title and show name from the page
2. Searches YouTube for the episode — uses YouTube pipeline if found
3. Falls back to finding the show's RSS feed (via iTunes search) and downloading the MP3
4. Returns a clear error if the episode can't be found outside Spotify

## Notes

- The first Whisper run downloads the model weights (~74MB for `base`) — subsequent runs are fast.
- Long episodes (60+ min) may take several minutes to transcribe with Whisper.
- Audio files are deleted immediately after transcription.
- Transcripts can be copied or downloaded as `.md` files from the browser.
