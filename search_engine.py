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
        if collection.name.startswith('video-') and collection.name.endswith(('-clip', '-captions', '-audio', '-scenes')):
            # Extract video ID from collection name
            parts = collection.name.split('-')
            if len(parts) >= 3:
                video_id = parts[1]  # video-{ID}-{type}
                video_ids.add(video_id)
    
    return list(video_ids)

def create_multimodal_query_embedding(
    query: str, 
    clip_weight: float = 0.6, 
    caption_weight: float = 0.4
) -> np.ndarray:
    """
    Create a fused multimodal query embedding combining CLIP and text embeddings.
    Handles dimension mismatch by padding caption embedding to match CLIP (512D).
    
    Args:
        query: Search query string
        clip_weight: Weight for CLIP embedding
        caption_weight: Weight for caption embedding
    
    Returns:
        Fused and normalized query embedding (512D)
    """
    # Generate CLIP text embedding (512D)
    with torch.no_grad():
        query_clip = clip.tokenize([query]).to(device)
        clip_embedding = clip_model.encode_text(query_clip).squeeze().detach().cpu().numpy()
    
    # Generate caption embedding (384D)
    caption_embedding = sentence_model.encode([query])[0]
    
    # Get dimensions
    clip_dim = clip_embedding.shape[0]
    caption_dim = caption_embedding.shape[0]
    
    # Pad caption embedding to match CLIP dimension (512D)
    if caption_dim < clip_dim:
        pad_size = clip_dim - caption_dim
        caption_padded = np.pad(caption_embedding, (0, pad_size), mode='constant', constant_values=0)
    else:
        caption_padded = caption_embedding
    
    # Normalize both embeddings (both should now be 512D)
    clip_norm = clip_embedding / (np.linalg.norm(clip_embedding) + 1e-8)
    caption_norm = caption_padded / (np.linalg.norm(caption_padded) + 1e-8)
    
    # Fuse embeddings with weights
    fused = clip_weight * clip_norm + caption_weight * caption_norm
    
    # Re-normalize the fused embedding
    fused_norm = fused / (np.linalg.norm(fused) + 1e-8)
    
    return fused_norm

def scene_search_multimodal(
    query: str, 
    video_id: str = None, 
    top_k: int = 5,
    clip_weight: float = 0.6,
    caption_weight: float = 0.4
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Search using multimodal scene-level embeddings.
    Creates a fused query embedding that matches the scene fusion approach.
    
    Args:
        query: Search query
        video_id: Optional specific video ID
        top_k: Number of results to return
        clip_weight: Weight for CLIP component (should match scene creation)
        caption_weight: Weight for caption component (should match scene creation)
    
    Returns:
        List of (timestamp, result_data) tuples
    """
    if video_id:
        return _scene_search_multimodal_specific_video(query, video_id, top_k, clip_weight, caption_weight)
    else:
        return _scene_search_multimodal_all_videos(query, top_k, clip_weight, caption_weight)

def _scene_search_multimodal_specific_video(
    query: str, 
    video_id: str, 
    top_k: int,
    clip_weight: float,
    caption_weight: float
) -> List[Tuple[float, Dict[str, Any]]]:
    """Search multimodal scenes within a specific video"""
    collections = get_collection_names(video_id)
    existing_collections = [c.name for c in qdrant.get_collections().collections]
    
    if collections['scenes'] not in existing_collections:
        return []
    
    # Create multimodal query embedding with same weights as scene creation
    query_embedding = create_multimodal_query_embedding(query, clip_weight, caption_weight)
    
    # Search scene collection
    scene_results = qdrant.search(
        collection_name=collections['scenes'],
        query_vector=query_embedding.tolist(),
        limit=top_k
    )
    
    results = []
    for result in scene_results:
        results.append((
            result.payload['start_timestamp'],
            {
                'score': result.score,
                'sources': ['multimodal_scene'],
                'payload': result.payload,
                'search_type': 'multimodal_fusion'
            }
        ))
    
    return results

def _scene_search_multimodal_all_videos(
    query: str, 
    top_k: int,
    clip_weight: float,
    caption_weight: float
) -> List[Tuple[float, Dict[str, Any]]]:
    """Search multimodal scenes across all videos"""
    video_ids = get_available_videos()
    all_results = []
    
    for video_id in video_ids:
        video_results = _scene_search_multimodal_specific_video(
            query, video_id, top_k, clip_weight, caption_weight
        )
        all_results.extend(video_results)
    
    # Sort all results by score
    all_results.sort(key=lambda x: x[1]['score'], reverse=True)
    return all_results[:top_k]

def scene_search(query: str, video_id: str = None, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Search using scene-level embeddings (CLIP-only, for backward compatibility).
    For multimodal scene search, use scene_search_multimodal instead.
    """
    if video_id:
        return _scene_search_specific_video(query, video_id, top_k)
    else:
        return _scene_search_all_videos(query, top_k)

def _scene_search_specific_video(query: str, video_id: str, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
    """Search scenes within a specific video using CLIP only"""
    collections = get_collection_names(video_id)
    existing_collections = [c.name for c in qdrant.get_collections().collections]
    
    if collections['scenes'] not in existing_collections:
        return []
    
    # Generate CLIP embedding for query
    with torch.no_grad():
        query_clip = clip.tokenize([query]).to(device)
        query_embedding = clip_model.encode_text(query_clip).squeeze().detach().cpu().numpy()
    
    # Search scene collection
    scene_results = qdrant.search(
        collection_name=collections['scenes'],
        query_vector=query_embedding.tolist(),
        limit=top_k
    )
    
    results = []
    for result in scene_results:
        results.append((
            result.payload['start_timestamp'],
            {
                'score': result.score,
                'sources': ['scene'],
                'payload': result.payload
            }
        ))
    
    return results

def _scene_search_all_videos(query: str, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
    """Search scenes across all videos"""
    video_ids = get_available_videos()
    all_results = []
    
    for video_id in video_ids:
        video_results = _scene_search_specific_video(query, video_id, top_k)
        all_results.extend(video_results)
    
    # Sort all results by score
    all_results.sort(key=lambda x: x[1]['score'], reverse=True)
    return all_results[:top_k]

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

def multimodal_fusion_search(query: str, video_id: str = None, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Multimodal fusion search combining CLIP and BLIP caption scores (frame-level)
    Score = 0.6 * clip_score + 0.4 * caption_score
    
    Returns results for frames that appear in both CLIP and caption searches
    """
    
    if video_id:
        return _multimodal_fusion_specific_video(query, video_id, top_k)
    else:
        return _multimodal_fusion_all_videos(query, top_k)

def _multimodal_fusion_specific_video(query: str, video_id: str, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
    """Multimodal fusion for a specific video"""
    collections = get_collection_names(video_id)
    existing_collections = [c.name for c in qdrant.get_collections().collections]
    
    # Get a larger set of results to ensure we have enough matches
    search_limit = top_k * 3
    
    clip_results_dict = {}
    caption_results_dict = {}
    
    # Search CLIP collection
    if collections['clip'] in existing_collections:
        with torch.no_grad():
            query_clip = clip.tokenize([query]).to(device)
            query_embedding_clip = clip_model.encode_text(query_clip).squeeze().detach().cpu().numpy()
        
        clip_results = qdrant.search(
            collection_name=collections['clip'],
            query_vector=query_embedding_clip.tolist(),
            limit=search_limit
        )
        
        for result in clip_results:
            frame_idx = result.payload['frame_index']
            clip_results_dict[frame_idx] = {
                'score': result.score,
                'payload': result.payload,
                'timestamp': result.payload['timestamp']
            }
    
    # Search caption collection
    if collections['caption'] in existing_collections:
        query_embedding_caption = sentence_model.encode([query])[0]
        
        caption_results = qdrant.search(
            collection_name=collections['caption'],
            query_vector=query_embedding_caption.tolist(),
            limit=search_limit
        )
        
        for result in caption_results:
            frame_idx = result.payload['frame_index']
            caption_results_dict[frame_idx] = {
                'score': result.score,
                'payload': result.payload,
                'timestamp': result.payload['timestamp']
            }
    
    # Combine results - only for frames that appear in both
    fused_results = []
    
    # Find common frame indices
    common_frames = set(clip_results_dict.keys()) & set(caption_results_dict.keys())
    
    for frame_idx in common_frames:
        clip_data = clip_results_dict[frame_idx]
        caption_data = caption_results_dict[frame_idx]
        
        # Calculate fused score: 0.6 * clip + 0.4 * caption
        fused_score = 0.6 * clip_data['score'] + 0.4 * caption_data['score']
        
        # Merge payload (prefer caption payload as it has caption text)
        merged_payload = caption_data['payload'].copy()
        
        fused_results.append((
            caption_data['timestamp'],
            {
                'score': fused_score,
                'sources': ['clip', 'caption'],
                'clip_score': clip_data['score'],
                'caption_score': caption_data['score'],
                'payload': merged_payload
            }
        ))
    
    # Sort by fused score
    fused_results.sort(key=lambda x: x[1]['score'], reverse=True)
    
    return fused_results[:top_k]

def _multimodal_fusion_all_videos(query: str, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
    """Multimodal fusion across all videos"""
    video_ids = get_available_videos()
    all_results = []
    
    for video_id in video_ids:
        video_results = _multimodal_fusion_specific_video(query, video_id, top_k)
        all_results.extend(video_results)
    
    # Sort all results by fused score
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