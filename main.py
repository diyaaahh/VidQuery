from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path

from audio_processor import process_video_for_audio_captions
from video_processor import process_video_hybrid_embeddings
from search_engine import hybrid_search, audio_search, get_available_videos, delete_video_collections
from video_utils import get_video_id

app = FastAPI(title="Video Search Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    video_id: Optional[str] = None  # Optional video filtering
    top_k: int = 5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    search_type: str
    video_id: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Video Search Engine API v2.0 - Per-Video Collections", "version": "2.0.0"}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload and process video with per-video collections"""
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Get video ID
        video_id = get_video_id(temp_path)
        
        # Process video for audio captions (Whisper)
        print(f"Processing audio for video {video_id}...")
        process_video_for_audio_captions(temp_path)
        
        # Process video for visual search (CLIP + BLIP)
        print(f"Processing video frames for video {video_id}...")
        process_video_hybrid_embeddings(temp_path)
        
        return {
            "message": "Video processed successfully",
            "filename": file.filename,
            "video_id": video_id,
            "status": "completed"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/search/video", response_model=SearchResponse)
async def search_video(request: SearchRequest):
    """Search video using visual content with optional video filtering"""
    
    try:
        results = hybrid_search(request.query, video_id=request.video_id, top_k=request.top_k)
        
        formatted_results = []
        for timestamp, data in results:
            formatted_results.append({
                "timestamp": timestamp,
                "score": data['score'],
                "sources": data['sources'],
                "caption": data['payload'].get('caption', ''),
                "video_path": data['payload'].get('video_path', ''),
                "video_id": data['payload'].get('video_id', '')
            })
        
        return SearchResponse(
            results=formatted_results,
            query=request.query,
            search_type="video_visual",
            video_id=request.video_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/audio", response_model=SearchResponse)
async def search_audio(request: SearchRequest):
    """Search video using audio transcriptions with optional video filtering"""
    
    try:
        results = audio_search(request.query, video_id=request.video_id, top_k=request.top_k)
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "timestamp": result.payload['timestamp'],
                "score": result.score,
                "transcription": result.payload.get('caption', ''),
                "video_path": result.payload.get('video_path', ''),
                "video_id": result.payload.get('video_id', '')
            })
        
        return SearchResponse(
            results=formatted_results,
            query=request.query,
            search_type="audio_transcription",
            video_id=request.video_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/videos")
async def list_videos():
    """List all available videos"""
    try:
        video_ids = get_available_videos()
        return {"videos": video_ids, "count": len(video_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list videos: {str(e)}")

@app.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and its collections"""
    try:
        delete_video_collections(video_id)
        return {"message": f"Video {video_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {str(e)}")

@app.get("/collections/status")
async def get_collections_status():
    """Get status of Qdrant collections"""
    from qdrant_client import QdrantClient
    
    try:
        qdrant = QdrantClient("localhost", port=6333)
        collections = qdrant.get_collections().collections
        
        status = {}
        for collection in collections:
            info = qdrant.get_collection(collection.name)
            status[collection.name] = {
                "points_count": info.points_count,
                "status": info.status
            }
        
        return {"collections": status}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)