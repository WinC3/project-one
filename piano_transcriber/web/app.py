"""
FastAPI web application for piano transcription.

This module provides a web interface for the piano transcription model,
allowing users to upload audio files and receive MIDI/JSON transcriptions.
"""

import os
import tempfile
import uuid
import io
from pathlib import Path
from typing import Dict, Any, Optional

import uvicorn
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from starlette.requests import Request

from piano_transcriber.inference import PianoTranscriber

app = FastAPI(
    title="Piano Transcription API",
    description="Convert piano audio to MIDI using neural networks",
    version="1.0.0"
)

# Setup paths
current_dir = Path(__file__).parent
static_dir = current_dir / "static"
templates_dir = current_dir / "templates"

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Global transcriber instance (lazy loaded)
transcriber: Optional[PianoTranscriber] = None

# Job storage (in production, use Redis or database)
transcription_jobs: Dict[str, Dict[str, Any]] = {}

def get_transcriber() -> PianoTranscriber:
    """Get or create the global transcriber instance."""
    global transcriber
    if transcriber is None:
        # Use specific model in piano_transcriber/model directory
        model_path = Path(__file__).parent.parent / "model" / "model_epoch_2454.pth"
        
        if model_path.exists():
            transcriber = PianoTranscriber(str(model_path))
        else:
            raise RuntimeError(f"Model not found at: {model_path}")
    return transcriber

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload an audio file for transcription.
    
    Returns a job ID that can be used to check transcription status.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Test model loading early to catch errors
        try:
            get_transcriber()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Initialize job status
        transcription_jobs[job_id] = {
            "status": "processing",
            "filename": file.filename,
            "temp_path": temp_path,
            "result": None,
            "error": None
        }
        
        # Start background transcription
        background_tasks.add_task(process_transcription, job_id)
        
        return {"job_id": job_id, "status": "processing"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_transcription(job_id: str):
    """Background task to process audio transcription."""
    job = transcription_jobs[job_id]
    
    try:
        # Get transcriber instance
        transcriber = get_transcriber()
        
        # Process the audio file - returns predictions dictionary
        predictions = transcriber.transcribe_audio(job["temp_path"])
        
        # Convert predictions to MIDI and save to temporary file
        midi_obj = transcriber.predictions_to_midi(predictions)
        
        # Save MIDI to temporary file for later download
        midi_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
        midi_obj.write(midi_temp_file.name)
        midi_temp_file.close()
        
        # Convert predictions to note list
        notes = transcriber.predictions_to_json(predictions)
        
        # Calculate duration from audio file
        waveform, sample_rate = torchaudio.load(job["temp_path"])
        duration = waveform.shape[1] / sample_rate
        
        # Update job with results (no binary data)
        job["status"] = "completed"
        job["result"] = {
            "midi_path": midi_temp_file.name,  # Store file path instead of binary data
            "notes": notes,
            "duration": duration
        }
        
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    
    finally:
        # Clean up temporary audio file
        try:
            os.unlink(job["temp_path"])
        except:
            pass

@app.get("/status/{job_id}")
async def get_transcription_status(job_id: str):
    """Check the status of a transcription job."""
    if job_id not in transcription_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = transcription_jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job["status"],
        "filename": job["filename"]
    }
    
    if job["status"] == "completed":
        response["result"] = job["result"]
    elif job["status"] == "error":
        response["error"] = job["error"]
    
    return response

@app.get("/download/{job_id}/midi")
async def download_midi(job_id: str):
    """Download MIDI file for a completed transcription."""
    if job_id not in transcription_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = transcription_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Transcription not completed")
    
    if not job["result"] or not job["result"]["midi_path"]:
        raise HTTPException(status_code=400, detail="No MIDI data available")
    
    midi_path = job["result"]["midi_path"]
    
    if not os.path.exists(midi_path):
        raise HTTPException(status_code=404, detail="MIDI file not found")
    
    # Cleanup function for background task
    def cleanup_midi_file():
        try:
            if os.path.exists(midi_path):
                os.unlink(midi_path)
        except:
            pass  # Ignore cleanup errors
    
    # Return file response with cleanup
    filename = f"{os.path.splitext(job['filename'])[0]}.mid"
    return FileResponse(
        midi_path,
        media_type="audio/midi",
        filename=filename,
        background=BackgroundTasks([cleanup_midi_file])
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return {
            "status": "healthy",
            "device": device,
            "torch_version": torch.__version__
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)