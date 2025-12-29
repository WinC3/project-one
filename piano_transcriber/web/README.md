# Piano Transcription Web Platform

## Quick Start

### Local Development

1. **Install web dependencies:**
   ```bash
   pip install -r piano_transcriber/web/requirements-web.txt
   ```

2. **Start the development server:**
   ```bash
   cd piano_transcriber/web
   python run_server.py
   ```
   
   Or with custom options:
   ```bash
   python run_server.py --host 0.0.0.0 --port 8080 --reload
   ```

3. **Open your browser:**
   Navigate to http://localhost:8000

### Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Web interface: http://localhost:8000
   - With nginx proxy: http://localhost

3. **Production deployment with nginx:**
   ```bash
   docker-compose --profile production up --build
   ```

### Manual Docker Build

```bash
# Build the image
docker build -f Dockerfile.web -t piano-transcriber-web .

# Run the container
docker run -p 8000:8000 -v ./uploads:/app/uploads piano-transcriber-web
```

## API Endpoints

### Upload and Transcribe
```http
POST /transcribe
Content-Type: multipart/form-data

file: [audio file]
```

Response:
```json
{
  "job_id": "unique-job-id",
  "status": "processing",
  "message": "Transcription started"
}
```

### Check Status
```http
GET /status/{job_id}
```

Response (processing):
```json
{
  "status": "processing",
  "job_id": "unique-job-id"
}
```

Response (completed):
```json
{
  "status": "completed",
  "job_id": "unique-job-id",
  "filename": "audio.wav",
  "result": {
    "duration": 30.5,
    "notes": [
      {
        "note": "C4",
        "start": 0.5,
        "end": 1.2,
        "velocity": 0.8
      }
    ]
  }
}
```

### Download MIDI
```http
GET /download/{job_id}/midi
```

Returns: MIDI file download

### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Features

- **Drag-and-Drop Interface**: Easy file upload with visual feedback
- **Real-time Progress**: Live status updates during transcription
- **MIDI Export**: Download transcribed music as MIDI files
- **Note Preview**: View detected notes before download
- **Mobile-Friendly**: Responsive design works on all devices
- **Background Processing**: Non-blocking transcription with job tracking
- **Error Handling**: Graceful error messages and recovery options

## Supported Audio Formats

- WAV, MP3, M4A, FLAC, OGG
- Any format supported by torchaudio/ffmpeg

## Configuration

### Environment Variables

- `TRANSCRIBER_MODEL_PATH`: Path to model checkpoint (default: auto-detect latest)
- `UPLOAD_DIR`: Directory for uploaded files (default: ./uploads)
- `TEMP_DIR`: Directory for temporary files (default: ./temp)
- `MAX_FILE_SIZE`: Maximum upload file size in MB (default: 100)
- `CLEANUP_INTERVAL`: File cleanup interval in seconds (default: 3600)

### Model Selection

The web platform automatically uses the latest available checkpoint from the `checkpoints/` directory. To use a specific model:

1. Set environment variable:
   ```bash
   export TRANSCRIBER_MODEL_PATH=/path/to/model.pth
   ```

2. Or modify the Docker Compose file:
   ```yaml
   environment:
     - TRANSCRIBER_MODEL_PATH=/app/checkpoints/model_epoch_1700.pth
   ```

## Production Deployment

### CPU-Only Deployment

The web platform is optimized for CPU inference and can run on basic cloud instances:

- **Memory**: 4GB+ recommended
- **CPU**: 2+ cores recommended
- **Storage**: 10GB+ for models and temporary files

### GPU Acceleration

For faster transcription with GPU support:

1. Use CUDA-enabled PyTorch in requirements
2. Deploy on GPU-enabled cloud instances (AWS p3, Google Cloud GPU, etc.)
3. Set `CUDA_VISIBLE_DEVICES` environment variable

### Cloud Deployment Options

#### Heroku
```bash
# Add buildpacks for Python and system dependencies
heroku buildpacks:add heroku/python
heroku buildpacks:add https://github.com/jonathanong/heroku-buildpack-ffmpeg-latest.git
```

#### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/YOUR-PROJECT/piano-transcriber-web
gcloud run deploy --image gcr.io/YOUR-PROJECT/piano-transcriber-web --platform managed
```

#### AWS EC2/ECS
Use the provided Dockerfile with AWS container services.

## Monitoring and Logging

The application provides:

- Health check endpoint (`/health`) for load balancer monitoring
- Request logging via uvicorn
- Error tracking and reporting
- Background task status tracking

## Security Considerations

- File uploads are validated for audio formats
- Temporary files are automatically cleaned up
- No persistent storage of user data
- CORS headers configured for web security
- Rate limiting can be added via nginx

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure checkpoint files are in `checkpoints/` directory
2. **CUDA errors**: Use CPU-only PyTorch for compatibility
3. **File size limits**: Adjust `client_max_body_size` in nginx config
4. **Timeout errors**: Increase timeout values for long audio files

### Debug Mode

Run with debug logging:
```bash
python run_server.py --reload --host 0.0.0.0
```

### Performance Optimization

- Use nginx for static file serving
- Enable gzip compression
- Use CDN for global distribution
- Implement caching for repeated requests