import hashlib
from pathlib import Path

def get_video_id(video_path: str) -> str:
    """Generate unique ID for video based on filename and path"""
    return hashlib.md5(video_path.encode()).hexdigest()[:12]

def get_collection_names(video_id: str):
    """Get collection names for a specific video"""
    return {
        'clip': f"video-{video_id}-clip",
        'caption': f"video-{video_id}-captions", 
        'audio': f"video-{video_id}-audio"
    }
