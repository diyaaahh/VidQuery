# Video Search Engine

A FastAPI-based video search engine that supports two types of search:
1. **Audio Search**: Search using Whisper transcriptions
2. **Video Search**: Visual search using CLIP + BLIP hybrid approach (weighted 0.6 CLIP + 0.4 BLIP)

## Features

- Upload videos and automatically process them for search
- **Audio Search**: Uses Whisper to transcribe audio and search through spoken content
- **Visual Search**: Uses CLIP for visual understanding and BLIP for image captioning
- Hybrid approach with weighted results (0.6 visual + 0.4 caption)
- Vector storage using Qdrant
- RESTful API with FastAPI

## Prerequisites

- Python 3.8+
- Docker (for Qdrant)
- CUDA-capable GPU (optional, for faster processing)

## Setup

### 1. Start Qdrant Database

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### Upload Video
```http
POST /upload-video
```
Upload a video file for processing. Supports MP4, AVI, MOV, MKV formats.

### Search Video (Visual)
```http
POST /search/video
```
Search using visual content (CLIP + BLIP hybrid approach).

**Request Body:**
```json
{
  "query": "person drinking coffee",
  "top_k": 5
}
```

### Search Audio
```http
POST /search/audio
```
Search using audio transcriptions (Whisper).

**Request Body:**
```json
{
  "query": "add sugar to the mixture",
  "top_k": 5
}
```

### Collection Status
```http
GET /collections/status
```
Check the status of Qdrant collections.

## Project Structure

```
├── main.py                 # FastAPI server
├── audio_processor.py      # Whisper audio processing
├── video_processor.py      # CLIP + BLIP video processing
├── search_engine.py        # Search functions
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## How It Works

### Audio Processing
1. Extracts audio from video using FFmpeg
2. Transcribes audio using Whisper
3. Generates embeddings using SentenceTransformer
4. Stores in Qdrant for search

### Video Processing
1. Extracts frames every 2 seconds
2. Generates CLIP embeddings for visual understanding
3. Creates captions using BLIP
4. Generates text embeddings for captions
5. Stores both in separate Qdrant collections

### Search
- **Audio Search**: Direct text similarity search on Whisper transcriptions
- **Video Search**: Hybrid approach combining CLIP visual similarity (0.6 weight) and BLIP caption similarity (0.4 weight)

## Usage Example

```bash
# Upload a video
curl -X POST "http://localhost:8000/upload-video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_video.mp4"

# Search video content
curl -X POST "http://localhost:8000/search/video" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"query": "person cooking", "top_k": 5}'

# Search audio content
curl -X POST "http://localhost:8000/search/audio" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"query": "add ingredients", "top_k": 5}'
```

## Configuration

- **Frame Extraction**: Every 2 seconds (configurable in `video_processor.py`)
- **Batch Size**: 4 images per batch for BLIP processing
- **Embedding Models**: 
  - Visual: CLIP ViT-B/32
  - Text: all-MiniLM-L6-v2
  - Captioning: BLIP base model
  - Audio: Whisper base model

## Performance Notes

- BLIP1 is used instead of BLIP2 for faster processing
- GPU acceleration is automatically detected and used when available
- Batch processing for efficient GPU utilization
- Memory cleanup after each batch to prevent OOM errors