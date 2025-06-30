import clip
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import List, Tuple, Dict, Any, Optional

# Import our utility functions
from video_utils import get_video_id, get_collection_names

# Setup device and models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant
qdrant = QdrantClient("localhost", port=6333)

def get_available_videos() -> List[str]:
    """Get list of available video IDs from collection names"""
    collections = qdrant.get_collections().collections
    video_ids = set()
    
    for collection in collections:
        if collection.name.startswith('video-') and collection.name.endswith(('-clip', '-captions', '-audio')):
            # Extract video ID from collection name
            parts = collection.name.split('-')
            if len(parts) >= 3:
                video_id = parts[1]  # video-{ID}-{type}
                video_ids.add(video_id)
    
    return list(video_ids)

def hybrid_search(query: str, video_id: str = None, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
    """Search using hybrid approach with optional video filtering"""
    
    if video_id:
        # Search specific video
        return _search_specific_video(query, video_id, top_k)
    else:
        # Search all videos
        return _search_all_videos(query, top_k)

def _search_specific_video(query: str, video_id: str, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
    """Search within a specific video's collections"""
    collections = get_collection_names(video_id)
    
    # Check if collections exist
    existing_collections = [c.name for c in qdrant.get_collections().collections]
    
    results = []
    
    # Search CLIP collection if exists
    if collections['clip'] in existing_collections:
        with torch.no_grad():  # Disable gradient computation
            query_clip = clip.tokenize([query]).to(device)
            query_embedding_clip = clip_model.encode_text(query_clip).squeeze().detach().cpu().numpy()
        
        clip_results = qdrant.search(
            collection_name=collections['clip'],
            query_vector=query_embedding_clip.tolist(),
            limit=top_k
        )
        
        # Apply CLIP weight (0.6)
        for result in clip_results:
            weighted_score = result.score * 0.6
            results.append((
                result.payload['timestamp'],
                {
                    'score': weighted_score,
                    'sources': ['clip'],
                    'payload': result.payload
                }
            ))
    
    # Search caption collection if exists
    if collections['caption'] in existing_collections:
        query_embedding_caption = sentence_model.encode([query])[0]
        
        caption_results = qdrant.search(
            collection_name=collections['caption'],
            query_vector=query_embedding_caption.tolist(),
            limit=top_k
        )
        
        # Apply caption weight (0.4)
        for result in caption_results:
            weighted_score = result.score * 0.4
            results.append((
                result.payload['timestamp'],
                {
                    'score': weighted_score,
                    'sources': ['caption'],
                    'payload': result.payload
                }
            ))
    
    # Combine results by timestamp if they exist for the same frame
    combined_results = {}
    for timestamp, data in results:
        if timestamp in combined_results:
            # Combine scores and sources
            combined_results[timestamp]['score'] += data['score']
            combined_results[timestamp]['sources'].extend(data['sources'])
        else:
            combined_results[timestamp] = data
    
    # Convert back to list and sort by combined score
    final_results = [(timestamp, data) for timestamp, data in combined_results.items()]
    final_results.sort(key=lambda x: x[1]['score'], reverse=True)
    
    return final_results[:top_k]

def _search_all_videos(query: str, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
    """Search across all video collections"""
    video_ids = get_available_videos()
    all_results = []
    
    for video_id in video_ids:
        video_results = _search_specific_video(query, video_id, top_k)
        all_results.extend(video_results)
    
    # Sort all results by score and return top results
    all_results.sort(key=lambda x: x[1]['score'], reverse=True)
    return all_results[:top_k]

def audio_search(query: str, video_id: str = None, top_k: int = 5):
    """Search audio transcriptions with optional video filtering"""
    
    if video_id:
        # Search specific video's audio collection
        collections = get_collection_names(video_id)
        collection_name = collections['audio']
        
        # Check if collection exists
        existing_collections = [c.name for c in qdrant.get_collections().collections]
        if collection_name not in existing_collections:
            return []
        
        query_embedding = sentence_model.encode([query])[0]
        
        results = qdrant.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        return results
    else:
        # Search all video audio collections
        video_ids = get_available_videos()
        all_results = []
        
        for vid_id in video_ids:
            collections = get_collection_names(vid_id)
            collection_name = collections['audio']
            
            existing_collections = [c.name for c in qdrant.get_collections().collections]
            if collection_name not in existing_collections:
                continue
            
            query_embedding = sentence_model.encode([query])[0]
            
            results = qdrant.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            
            all_results.extend(results)
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

def delete_video_collections(video_id: str):
    """Delete all collections for a specific video"""
    collections = get_collection_names(video_id)
    
    for collection_name in collections.values():
        try:
            qdrant.delete_collection(collection_name)
            print(f"Deleted collection: {collection_name}")
        except Exception as e:
            print(f"Failed to delete collection {collection_name}: {e}")