from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from audio_processor import process_video_for_audio_captions
from video_processor import process_video_hybrid_embeddings
from search_engine import (
    hybrid_search, 
    audio_search, 
    multimodal_fusion_search,
    scene_search,
    scene_search_multimodal,
    get_available_videos, 
    delete_video_collections
)
from video_utils import get_video_id, get_video_metadata, save_video_metadata, delete_video_metadata, get_all_video_metadata

app = FastAPI(title="Video Search Engine", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class SearchRequest(BaseModel):
    query: str
    video_id: Optional[str] = None
    top_k: int = 5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    search_type: str
    video_id: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Video Search Engine API v2.1 - Multimodal Scene Fusion", "version": "2.1.0"}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload and process video with per-video collections and scene segmentation"""
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    # Create a safe filename
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in ('_', '-', '.')).rstrip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{safe_filename}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save uploaded file permanently
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    try:
        # Get video ID
        video_id = get_video_id(str(file_path))
        
        # Save video metadata
        metadata = {
            "video_id": video_id,
            "original_name": file.filename,
            "file_path": str(file_path),
            "uploaded_at": datetime.now().isoformat(),
            "file_size": os.path.getsize(file_path)
        }
        save_video_metadata(video_id, metadata)
        
        # Process video for audio captions (Whisper)
        print(f"Processing audio for video {video_id}...")
        process_video_for_audio_captions(str(file_path))
        
        # Process video for visual search (CLIP + BLIP) with scene segmentation
        print(f"Processing video frames for video {video_id}...")
        process_video_hybrid_embeddings(
            str(file_path),
            enable_scenes=True,
            scene_threshold=0.95,
            min_scene_duration=2.0,
            clip_weight=0.6,
            caption_weight=0.4
        )
        
        return {
            "message": "Video processed successfully",
            "filename": file.filename,
            "video_id": video_id,
            "file_path": str(file_path),
            "status": "completed"
        }
    
    except Exception as e:
        # Clean up file if processing failed
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/search/video", response_model=SearchResponse)
async def search_video(request: SearchRequest):
    """Search video using visual content with optional video filtering"""
    
    try:
        results = hybrid_search(request.query, video_id=request.video_id, top_k=request.top_k)
        
        formatted_results = []
        for timestamp, data in results:
            video_id = data['payload'].get('video_id', '')
            metadata = get_video_metadata(video_id)
            
            formatted_results.append({
                "timestamp": timestamp,
                "score": data['score'],
                "sources": data['sources'],
                "caption": data['payload'].get('caption', ''),
                "video_path": data['payload'].get('video_path', ''),
                "video_id": video_id,
                "original_name": metadata.get('original_name', '') if metadata else ''
            })
        
        return SearchResponse(
            results=formatted_results,
            query=request.query,
            search_type="video_visual",
            video_id=request.video_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/scenes", response_model=SearchResponse)
async def search_scenes(request: SearchRequest):
    """
    Search video using scene-level embeddings (CLIP-only).
    Returns complete scenes instead of individual frames.
    """
    
    try:
        results = scene_search(request.query, video_id=request.video_id, top_k=request.top_k)
        
        formatted_results = []
        for timestamp, data in results:
            video_id = data['payload'].get('video_id', '')
            metadata = get_video_metadata(video_id)
            
            formatted_results.append({
                "timestamp": timestamp,
                "end_timestamp": data['payload'].get('end_timestamp', timestamp),
                "duration": data['payload'].get('duration', 0),
                "score": data['score'],
                "sources": data['sources'],
                "scene_id": data['payload'].get('scene_id', 0),
                "scene_summary": data['payload'].get('scene_summary', ''),
                "num_frames": data['payload'].get('num_frames', 0),
                "video_path": data['payload'].get('video_path', ''),
                "video_id": video_id,
                "original_name": metadata.get('original_name', '') if metadata else '',
                "type": "scene"
            })
        
        return SearchResponse(
            results=formatted_results,
            query=request.query,
            search_type="scene_based",
            video_id=request.video_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/scenes-multimodal", response_model=SearchResponse)
async def search_scenes_multimodal(request: SearchRequest):
    """
    Search video using multimodal scene-level embeddings (CLIP + Caption fusion).
    Returns complete scenes with fused embeddings that combine visual and textual understanding.
    This provides the best semantic matching for complex queries.
    """
    
    try:
        
        results = scene_search_multimodal(
            request.query, 
            video_id=request.video_id, 
            top_k=request.top_k,
            clip_weight=0.6,
            caption_weight=0.4
        )
        
        print(f"Got {len(results)} results")
        
        formatted_results = []
        for i, (timestamp, data) in enumerate(results):
            
            video_id = data['payload'].get('video_id', '')
            
            metadata = get_video_metadata(video_id)
            print(f"  Metadata: {metadata is not None}")
            
            formatted_results.append({
                "timestamp": timestamp,
                "end_timestamp": data['payload'].get('end_timestamp', timestamp),
                "duration": data['payload'].get('duration', 0),
                "score": data['score'],
                "sources": data['sources'],
                "scene_id": data['payload'].get('scene_id', 0),
                "scene_summary": data['payload'].get('scene_summary', ''),
                "num_frames": data['payload'].get('num_frames', 0),
                "video_path": data['payload'].get('video_path', ''),
                "video_id": video_id,
                "original_name": metadata.get('original_name', '') if metadata else '',
                "type": "multimodal_scene",
                "fusion_weights": {
                    "clip": data['payload'].get('clip_weight', 0.6),
                    "caption": data['payload'].get('caption_weight', 0.4)
                }
            })
        
        print(f"Formatted {len(formatted_results)} results successfully")
        
        return SearchResponse(
            results=formatted_results,
            query=request.query,
            search_type="multimodal_scene_fusion",
            video_id=request.video_id
        )
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in search_scenes_multimodal:")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}\n\nTraceback:\n{error_trace}")

@app.post("/search/multimodal", response_model=SearchResponse)
async def search_multimodal(request: SearchRequest):
    """
    Multimodal fusion search combining CLIP (0.6) and BLIP caption (0.4) scores
    Returns frames that match in both visual and caption embeddings
    """
    
    try:
        results = multimodal_fusion_search(request.query, video_id=request.video_id, top_k=request.top_k)
        
        formatted_results = []
        for timestamp, data in results:
            video_id = data['payload'].get('video_id', '')
            metadata = get_video_metadata(video_id)
            
            formatted_results.append({
                "timestamp": timestamp,
                "score": data['score'],
                "sources": data['sources'],
                "caption": data['payload'].get('caption', ''),
                "video_path": data['payload'].get('video_path', ''),
                "video_id": video_id,
                "original_name": metadata.get('original_name', '') if metadata else '',
                "clip_score": data.get('clip_score', 0),
                "caption_score": data.get('caption_score', 0)
            })
        
        return SearchResponse(
            results=formatted_results,
            query=request.query,
            search_type="multimodal_fusion",
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
            video_id = result.payload.get('video_id', '')
            metadata = get_video_metadata(video_id)
            
            formatted_results.append({
                "timestamp": result.payload['timestamp'],
                "score": result.score,
                "transcription": result.payload.get('caption', ''),
                "video_path": result.payload.get('video_path', ''),
                "video_id": video_id,
                "original_name": metadata.get('original_name', '') if metadata else ''
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
    """List all available videos with metadata"""
    try:
        videos_metadata = get_all_video_metadata()
        return {"videos": videos_metadata, "count": len(videos_metadata)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list videos: {str(e)}")

@app.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and its collections"""
    try:
        # Get metadata to find file path
        metadata = get_video_metadata(video_id)
        
        # Delete collections
        delete_video_collections(video_id)
        
        # Delete video file
        if metadata and 'file_path' in metadata:
            file_path = Path(metadata['file_path'])
            if file_path.exists():
                os.remove(file_path)
        
        # Delete metadata
        delete_video_metadata(video_id)
        
        return {"message": f"Video {video_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {str(e)}")

@app.get("/videos/{video_id}/stream")
async def stream_video(video_id: str):
    """Stream video file"""
    try:
        metadata = get_video_metadata(video_id)
        if not metadata or 'file_path' not in metadata:
            raise HTTPException(status_code=404, detail="Video not found")
        
        file_path = Path(metadata['file_path'])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        return FileResponse(
            file_path,
            media_type="video/mp4",
            filename=metadata.get('original_name', 'video.mp4')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stream video: {str(e)}")

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