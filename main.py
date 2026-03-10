import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import whisper
import torch

app = FastAPI()

MODEL_SIZES = ["tiny", "base", "small", "medium", "large"]
current_model = None
current_model_size = "base"
MODEL_CACHE_DIR = os.path.expanduser("~/.cache/whisper")

UPLOAD_DIR = tempfile.mkdtemp()


def ensure_model_downloaded(size: str):
    model_file = f"{size}.pt"
    model_path = os.path.join(MODEL_CACHE_DIR, model_file)
    if not os.path.exists(model_path):
        print(f"Downloading Whisper model: {size}...")
        whisper.load_model(size)
        print(f"Model '{size}' downloaded!")
    return True


def get_model(size: str):
    global current_model, current_model_size
    if current_model is None or current_model_size != size:
        ensure_model_downloaded(size)
        print(f"Loading Whisper model: {size}...")
        current_model = whisper.load_model(size)
        current_model_size = size
        print(f"Model '{size}' loaded successfully!")
    return current_model


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), model: str = Form("base")):
    if model not in MODEL_SIZES:
        raise HTTPException(status_code=400, detail=f"Invalid model size. Choose from: {MODEL_SIZES}")
    
    filename = file.filename or "audio_file"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        model_instance = get_model(model)
        result = model_instance.transcribe(file_path)
        return {"text": result["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.get("/models")
async def list_models():
    available = []
    for size in MODEL_SIZES:
        model_file = f"{size}.pt"
        model_path = os.path.join(MODEL_CACHE_DIR, model_file)
        available.append({
            "size": size,
            "downloaded": os.path.exists(model_path)
        })
    return {"models": available, "current": current_model_size}


@app.post("/models/{size}/download")
async def download_model(size: str):
    if size not in MODEL_SIZES:
        raise HTTPException(status_code=400, detail=f"Invalid model size. Choose from: {MODEL_SIZES}")
    try:
        ensure_model_downloaded(size)
        return {"status": "downloaded", "size": size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{size}")
async def delete_model(size: str):
    if size not in MODEL_SIZES:
        raise HTTPException(status_code=400, detail=f"Invalid model size. Choose from: {MODEL_SIZES}")
    global current_model, current_model_size
    model_file = f"{size}.pt"
    model_path = os.path.join(MODEL_CACHE_DIR, model_file)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{size}' not found")
    try:
        if current_model_size == size:
            current_model = None
            current_model_size = None
        os.remove(model_path)
        return {"status": "deleted", "size": size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Whisper WebGUI</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            :root {
                --bg: #f4f4f4;
                --paper: #fff;
                --text: #111;
                --text-muted: #555;
                --border: #ccc;
                --border-hover: #888;
                --btn-bg: #333;
                --btn-text: #fff;
                --card-bg: #fff;
            }
            :root.dark {
                --bg: #111;
                --paper: #1a1a1a;
                --text: #eee;
                --text-muted: #888;
                --border: #444;
                --border-hover: #666;
                --btn-bg: #ddd;
                --btn-text: #111;
                --card-bg: #1a1a1a;
            }
            @media (prefers-color-scheme: dark) {
                :root:not(.light) {
                    --bg: #111;
                    --paper: #1a1a1a;
                    --text: #eee;
                    --text-muted: #888;
                    --border: #444;
                    --border-hover: #666;
                    --btn-bg: #ddd;
                    --btn-text: #111;
                    --card-bg: #1a1a1a;
                }
            }
            body {
                font-family: 'Georgia', 'Times New Roman', serif;
                background: var(--bg);
                color: var(--text);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
                font-weight: normal;
                letter-spacing: 2px;
                text-transform: uppercase;
                font-size: 1.5rem;
            }
            .card {
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 24px;
                margin-bottom: 20px;
            }
            .drop-zone {
                border: 2px dashed var(--border);
                border-radius: 4px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
            }
            .drop-zone:hover, .drop-zone.dragover {
                border-color: var(--border-hover);
                background: rgba(0,0,0,0.02);
            }
            .drop-zone input {
                display: none;
            }
            .model-select {
                margin-bottom: 20px;
            }
            .delete-btn {
                padding: 10px 16px;
                background: transparent;
                color: var(--text-muted);
                border: 1px solid var(--border);
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                font-family: inherit;
                transition: all 0.2s;
            }
            .delete-btn:hover:not(:disabled) {
                border-color: var(--text);
                color: var(--text);
            }
            .delete-btn:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
            }
            .model-status {
                font-size: 12px;
                margin-top: 4px;
                font-family: monospace;
            }
            .model-status.downloaded {
                color: var(--text);
            }
            .model-status.not-downloaded {
                color: var(--text-muted);
            }
            .model-status.downloading {
                color: var(--text-muted);
                font-style: italic;
            }
            label {
                display: block;
                margin-bottom: 8px;
                color: var(--text-muted);
                font-size: 0.9rem;
            }
            select, button {
                width: 100%;
                padding: 12px;
                border-radius: 4px;
                border: 1px solid var(--border);
                font-size: 16px;
                font-family: inherit;
            }
            select {
                background: var(--paper);
                color: var(--text);
                cursor: pointer;
            }
            button {
                background: var(--btn-bg);
                color: var(--btn-text);
                font-weight: normal;
                cursor: pointer;
                transition: all 0.2s;
                margin-top: 20px;
            }
            button:hover:not(:disabled) {
                opacity: 0.9;
            }
            button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            .status {
                text-align: center;
                padding: 20px;
                color: var(--text-muted);
                font-family: monospace;
            }
            .status.loading {
                font-style: italic;
            }
            .result {
                white-space: pre-wrap;
                line-height: 1.6;
                background: var(--paper);
                padding: 20px;
                border: 1px solid var(--border);
                border-radius: 4px;
                max-height: 400px;
                overflow-y: auto;
                font-size: 0.95rem;
            }
            .file-info {
                margin-top: 10px;
                color: var(--text-muted);
                font-family: monospace;
                font-size: 0.85rem;
            }
            .theme-toggle {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: var(--card-bg);
                border: 1px solid var(--border);
                color: var(--text);
                cursor: pointer;
                font-size: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
            }
            .theme-toggle:hover {
                border-color: var(--border-hover);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Whisper WebGUI</h1>
            
            <div class="card">
                <div class="model-select">
                    <label>Model Size</label>
                    <select id="modelSize">
                        <option value="tiny">Tiny (39M) - Fastest</option>
                        <option value="base" selected>Base (74M)</option>
                        <option value="small">Small (244M)</option>
                        <option value="medium">Medium (769M)</option>
                        <option value="large">Large (1550M) - Most accurate</option>
                    </select>
                    <div class="model-status" id="modelStatus"></div>
                    <button class="delete-btn" id="deleteBtn" style="margin-top: 10px;">Delete Model</button>
                </div>
                
                <div class="drop-zone" id="dropZone">
                    <input type="file" id="audioFile" accept="audio/*">
                    <p>Drop audio file here or click to browse</p>
                    <p class="file-info" id="fileInfo"></p>
                </div>
                
                <button id="transcribeBtn" style="margin-top: 20px;">Transcribe</button>
            </div>
            
            <div class="card" id="statusCard" style="display: none;">
                <div class="status" id="status"></div>
            </div>
            
            <div class="card" id="resultCard" style="display: none;">
                <label>Transcription</label>
                <div class="result" id="result"></div>
            </div>
        </div>
        
        <button class="theme-toggle" id="themeToggle" title="Toggle theme">☀</button>
        
        <script>
            const themeToggle = document.getElementById('themeToggle');
            
            function setTheme(dark) {
                if (dark) {
                    document.documentElement.classList.add('dark');
                    document.documentElement.classList.remove('light');
                    themeToggle.textContent = '☾';
                } else {
                    document.documentElement.classList.add('light');
                    document.documentElement.classList.remove('dark');
                    themeToggle.textContent = '☀';
                }
                localStorage.setItem('theme', dark ? 'dark' : 'light');
            }
            
            const savedTheme = localStorage.getItem('theme');
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            const isDark = savedTheme ? savedTheme === 'dark' : prefersDark;
            themeToggle.textContent = isDark ? '☾' : '☀';
            
            themeToggle.addEventListener('click', () => {
                const isDark = document.documentElement.classList.contains('dark');
                setTheme(!isDark);
            });
            
            const dropZone = document.getElementById('dropZone');
            const audioFile = document.getElementById('audioFile');
            const fileInfo = document.getElementById('fileInfo');
            const transcribeBtn = document.getElementById('transcribeBtn');
            const statusCard = document.getElementById('statusCard');
            const status = document.getElementById('status');
            const resultCard = document.getElementById('resultCard');
            const result = document.getElementById('result');
            const modelSize = document.getElementById('modelSize');
            const modelStatus = document.getElementById('modelStatus');
            const deleteBtn = document.getElementById('deleteBtn');
            
            let selectedFile = null;
            
            async function checkModelStatus(size) {
                try {
                    const response = await fetch('/models');
                    const data = await response.json();
                    const model = data.models.find(m => m.size === size);
                    
                    if (model.downloaded) {
                        modelStatus.textContent = 'Ready';
                        modelStatus.className = 'model-status downloaded';
                    } else {
                        modelStatus.textContent = 'Not downloaded';
                        modelStatus.className = 'model-status not-downloaded';
                    }
                } catch (err) {
                    modelStatus.textContent = 'Error checking model';
                    modelStatus.className = 'model-status not-downloaded';
                }
            }
            
            async function ensureModelDownloaded(size) {
                try {
                    const response = await fetch('/models');
                    const data = await response.json();
                    const model = data.models.find(m => m.size === size);
                    
                    if (!model.downloaded) {
                        modelStatus.textContent = 'Downloading model...';
                        modelStatus.className = 'model-status downloading';
                        
                        await fetch(`/models/${size}/download`, { method: 'POST' });
                    }
                    
                    modelStatus.textContent = 'Ready';
                    modelStatus.className = 'model-status downloaded';
                } catch (err) {
                    modelStatus.textContent = 'Error downloading model';
                    modelStatus.className = 'model-status not-downloaded';
                    throw err;
                }
            }
            
            checkModelStatus(modelSize.value);
            
            modelSize.addEventListener('change', () => {
                checkModelStatus(modelSize.value);
            });
            
            deleteBtn.addEventListener('click', async () => {
                const size = modelSize.value;
                if (!confirm(`Delete the ${size} model?`)) return;
                
                try {
                    deleteBtn.disabled = true;
                    const response = await fetch(`/models/${size}`, { method: 'DELETE' });
                    
                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Delete failed');
                    }
                    
                    modelStatus.textContent = 'Deleted';
                    modelStatus.className = 'model-status not-downloaded';
                    checkModelStatus(size);
                } catch (err) {
                    alert('Error: ' + err.message);
                } finally {
                    deleteBtn.disabled = false;
                }
            });
            
            dropZone.addEventListener('click', () => audioFile.click());
            
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            });
            
            dropZone.addEventListener('dragleave', () => {
                dropZone.classList.remove('dragover');
            });
            
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
                if (e.dataTransfer.files.length) {
                    handleFile(e.dataTransfer.files[0]);
                }
            });
            
            audioFile.addEventListener('change', (e) => {
                if (e.target.files.length) {
                    handleFile(e.target.files[0]);
                }
            });
            
            function handleFile(file) {
                selectedFile = file;
                fileInfo.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
            }
            
            transcribeBtn.addEventListener('click', async () => {
                if (!selectedFile) {
                    alert('Please select an audio file');
                    return;
                }
                
                statusCard.style.display = 'block';
                resultCard.style.display = 'none';
                transcribeBtn.disabled = true;
                
                try {
                    status.textContent = 'Loading model...';
                    status.className = 'status loading';
                    
                    await ensureModelDownloaded(modelSize.value);
                    
                    status.textContent = 'Transcribing...';
                    
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    formData.append('model', modelSize.value);
                    
                    const response = await fetch('/transcribe', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Transcription failed');
                    }
                    
                    const data = await response.json();
                    result.textContent = data.text;
                    resultCard.style.display = 'block';
                    statusCard.style.display = 'none';
                } catch (err) {
                    status.textContent = 'Error: ' + err.message;
                    status.className = 'status';
                } finally {
                    transcribeBtn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
