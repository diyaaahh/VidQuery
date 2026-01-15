import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

# Directory to store video metadata
METADATA_DIR = Path("metadata")
METADATA_DIR.mkdir(exist_ok=True)

def get_video_id(video_path: str) -> str:
    """Generate unique ID for video based on filename and path"""
    return hashlib.md5(video_path.encode()).hexdigest()[:12]

def get_collection_names(video_id: str):
    """Get collection names for a specific video"""
    return {
        'clip': f"video-{video_id}-clip",
        'caption': f"video-{video_id}-captions", 
        'audio': f"video-{video_id}-audio",
        'scenes': f"video-{video_id}-scenes"  # New scene collection
    }

def save_video_metadata(video_id: str, metadata: Dict):
    """Save video metadata to JSON file"""
    metadata_file = METADATA_DIR / f"{video_id}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def get_video_metadata(video_id: str) -> Optional[Dict]:
    """Get video metadata from JSON file"""
    metadata_file = METADATA_DIR / f"{video_id}.json"
    if not metadata_file.exists():
        return None
    
    with open(metadata_file, 'r') as f:
        return json.load(f)

def get_all_video_metadata() -> List[Dict]:
    """Get metadata for all videos"""
    metadata_files = METADATA_DIR.glob("*.json")
    all_metadata = []
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                all_metadata.append(metadata)
        except Exception as e:
            print(f"Error reading metadata file {metadata_file}: {e}")
    
    return all_metadata

def delete_video_metadata(video_id: str):
    """Delete video metadata file"""
    metadata_file = METADATA_DIR / f"{video_id}.json"
    if metadata_file.exists():
        metadata_file.unlink()