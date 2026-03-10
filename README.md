# Whisper WebGUI

A web-based GUI for OpenAI's Whisper speech recognition model.

## Prerequisites

- Python 3.10+
- ffmpeg (system package)

Install ffmpeg on Ubuntu/Debian:
```bash
sudo apt install ffmpeg
```

Install ffmpeg on macOS:
```bash
brew install ffmpeg
```

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install them manually:
```bash
pip install fastapi python-multipart uvicorn
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper
```

## Running

Start the server:
```bash
./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

Or use the run script:
```bash
./run.sh
```

Open your browser to: **http://localhost:8000**

## Usage

1. Select a model size from the dropdown:
   - **Tiny** (39M) - Fastest, least accurate
   - **Base** (74M) - Default
   - **Small** (244M)
   - **Medium** (769M)
   - **Large** (1550M) - Slowest, most accurate

2. Drag & drop an audio file or click to browse

3. Click "Transcribe" to transcribe the audio

4. View the transcription result

## Notes

- First run will download the Whisper model (~74MB for base model)
- CPU-only version is installed by default
- Supported audio formats: mp3, wav, m4a, ogg, flac, and more
