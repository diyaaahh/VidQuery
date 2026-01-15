import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

def detect_scene_boundaries(embeddings: np.ndarray, threshold: float = 0.85) -> List[int]:
    """
    Detect scene boundaries based on cosine similarity between consecutive frames.
    
    Args:
        embeddings: Array of frame embeddings (n_frames, embedding_dim)
        threshold: Similarity threshold. Frames with similarity below this are considered scene boundaries.
    
    Returns:
        List of frame indices where scene changes occur
    """
    if len(embeddings) < 2:
        return []
    
    scene_boundaries = [0]  # First frame is always a scene boundary
    
    for i in range(1, len(embeddings)):
        # Calculate cosine similarity between consecutive frames
        similarity = cosine_similarity(
            embeddings[i-1:i], 
            embeddings[i:i+1]
        )[0][0]
        
        # If similarity drops below threshold, mark as scene boundary
        if similarity < threshold:
            scene_boundaries.append(i)
    
    return scene_boundaries

def normalize_embeddings_to_same_dim(clip_emb: np.ndarray, caption_emb: np.ndarray) -> tuple:
    """
    Normalize embeddings to the same dimension for fusion.
    Since CLIP is 512D and captions are 384D, we need to handle this.
    
    We'll use the approach of normalizing each separately and keeping them in their
    native dimensions, then concatenating and re-normalizing for fusion.
    
    Args:
        clip_emb: CLIP embedding (512D)
        caption_emb: Caption embedding (384D)
    
    Returns:
        Tuple of (normalized_clip, normalized_caption)
    """
    # Normalize each embedding
    clip_norm = clip_emb / (np.linalg.norm(clip_emb) + 1e-8)
    caption_norm = caption_emb / (np.linalg.norm(caption_emb) + 1e-8)
    
    return clip_norm, caption_norm

def fuse_embeddings(clip_emb: np.ndarray, caption_emb: np.ndarray, 
                   clip_weight: float = 0.6, caption_weight: float = 0.4) -> np.ndarray:
    """
    Fuse CLIP and caption embeddings into a single embedding.
    
    Since CLIP (512D) and caption (384D) have different dimensions, we:
    1. Pad caption embedding to 512D with zeros
    2. Normalize both
    3. Weighted fusion
    4. Re-normalize
    
    Args:
        clip_emb: CLIP embedding (512D)
        caption_emb: Caption embedding (384D)
        clip_weight: Weight for CLIP
        caption_weight: Weight for caption
    
    Returns:
        Fused embedding (512D)
    """
    # Pad caption embedding to match CLIP dimension (512D)
    caption_dim = caption_emb.shape[-1]
    clip_dim = clip_emb.shape[-1]
    
    if caption_dim < clip_dim:
        # Pad with zeros
        pad_size = clip_dim - caption_dim
        if caption_emb.ndim == 1:
            caption_padded = np.pad(caption_emb, (0, pad_size), mode='constant', constant_values=0)
        else:
            caption_padded = np.pad(caption_emb, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
    else:
        caption_padded = caption_emb
    
    # Normalize both
    clip_norm = clip_emb / (np.linalg.norm(clip_emb, axis=-1, keepdims=True) + 1e-8)
    caption_norm = caption_padded / (np.linalg.norm(caption_padded, axis=-1, keepdims=True) + 1e-8)
    
    # Fuse with weights
    fused = clip_weight * clip_norm + caption_weight * caption_norm
    
    # Re-normalize
    fused_norm = fused / (np.linalg.norm(fused, axis=-1, keepdims=True) + 1e-8)
    
    return fused_norm

def create_scenes_with_multimodal(
    clip_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
    timestamps: List[float],
    captions: List[str] = None,
    frame_indices: List[int] = None,
    threshold: float = 0.85,
    min_scene_length: int = 3,
    clip_weight: float = 0.6,
    caption_weight: float = 0.4
) -> List[dict]:
    """
    Create scene segments with multimodal fusion (CLIP + Caption embeddings).
    
    Args:
        clip_embeddings: Array of CLIP visual embeddings (N, 512)
        caption_embeddings: Array of caption embeddings (N, 384)
        timestamps: List of frame timestamps
        captions: Optional list of frame captions
        frame_indices: Optional list of frame indices
        threshold: Similarity threshold for scene detection
        min_scene_length: Minimum number of frames in a scene
        clip_weight: Weight for CLIP embeddings (default 0.6)
        caption_weight: Weight for caption embeddings (default 0.4)
    
    Returns:
        List of scene dictionaries containing multimodal scene information
    """
    if len(clip_embeddings) == 0:
        return []
    
    print(f"Creating scenes with multimodal fusion...")
    print(f"  CLIP embeddings shape: {clip_embeddings.shape}")
    print(f"  Caption embeddings shape: {caption_embeddings.shape}")
    
    # Detect scene boundaries using CLIP embeddings
    boundaries = detect_scene_boundaries(clip_embeddings, threshold)
    boundaries.append(len(clip_embeddings))  # Add end boundary
    
    print(f"  Detected {len(boundaries)-1} initial scene boundaries")
    
    scenes = []
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        # Skip scenes that are too short
        if end_idx - start_idx < min_scene_length:
            # Merge with previous scene if possible
            if scenes:
                scenes[-1]['end_frame'] = end_idx
                scenes[-1]['end_timestamp'] = timestamps[end_idx - 1]
                scenes[-1]['clip_embeddings'] = np.vstack([
                    scenes[-1]['clip_embeddings'],
                    clip_embeddings[start_idx:end_idx]
                ])
                scenes[-1]['caption_embeddings'] = np.vstack([
                    scenes[-1]['caption_embeddings'],
                    caption_embeddings[start_idx:end_idx]
                ])
                if captions:
                    scenes[-1]['captions'].extend(captions[start_idx:end_idx])
                if frame_indices:
                    scenes[-1]['frame_indices'].extend(frame_indices[start_idx:end_idx])
                continue
            else:
                # If it's the first scene and too short, just include it
                pass
        
        # Get embeddings for this scene
        scene_clip_embeddings = clip_embeddings[start_idx:end_idx]
        scene_caption_embeddings = caption_embeddings[start_idx:end_idx]
        
        # Create scene-level embeddings by averaging
        scene_clip_embedding = np.mean(scene_clip_embeddings, axis=0)
        scene_caption_embedding = np.mean(scene_caption_embeddings, axis=0)
        
        # Create fused embedding
        scene_fused_embedding = fuse_embeddings(
            scene_clip_embedding, 
            scene_caption_embedding, 
            clip_weight, 
            caption_weight
        )
        
        scene = {
            'scene_id': i,
            'start_frame': start_idx,
            'end_frame': end_idx,
            'start_timestamp': timestamps[start_idx],
            'end_timestamp': timestamps[end_idx - 1],
            'duration': timestamps[end_idx - 1] - timestamps[start_idx],
            'num_frames': end_idx - start_idx,
            
            # Multimodal embeddings
            'scene_embedding': scene_fused_embedding,  # Main fused embedding (512D)
            'scene_clip_embedding': scene_clip_embedding,  # CLIP only (512D)
            'scene_caption_embedding': scene_caption_embedding,  # Caption only (384D)
            
            # Frame-level embeddings
            'clip_embeddings': scene_clip_embeddings,
            'caption_embeddings': scene_caption_embeddings,
            
            # Metadata
            'captions': captions[start_idx:end_idx] if captions else [],
            'frame_indices': frame_indices[start_idx:end_idx] if frame_indices else list(range(start_idx, end_idx)),
            
            # Fusion weights used
            'clip_weight': clip_weight,
            'caption_weight': caption_weight
        }
        
        scenes.append(scene)
    
    print(f"  Created {len(scenes)} scenes")
    return scenes

def create_scenes(
    embeddings: np.ndarray, 
    timestamps: List[float],
    captions: List[str] = None,
    frame_indices: List[int] = None,
    threshold: float = 0.85,
    min_scene_length: int = 3
) -> List[dict]:
    """
    Create scene segments from frame embeddings (backward compatibility).
    
    Args:
        embeddings: Array of frame embeddings
        timestamps: List of frame timestamps
        captions: Optional list of frame captions
        frame_indices: Optional list of frame indices
        threshold: Similarity threshold for scene detection
        min_scene_length: Minimum number of frames in a scene
    
    Returns:
        List of scene dictionaries containing scene information
    """
    if len(embeddings) == 0:
        return []
    
    # Detect scene boundaries
    boundaries = detect_scene_boundaries(embeddings, threshold)
    boundaries.append(len(embeddings))  # Add end boundary
    
    scenes = []
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        # Skip scenes that are too short
        if end_idx - start_idx < min_scene_length:
            # Merge with previous scene if possible
            if scenes:
                scenes[-1]['end_frame'] = end_idx
                scenes[-1]['end_timestamp'] = timestamps[end_idx - 1]
                scenes[-1]['frame_embeddings'] = np.vstack([
                    scenes[-1]['frame_embeddings'],
                    embeddings[start_idx:end_idx]
                ])
                if captions:
                    scenes[-1]['captions'].extend(captions[start_idx:end_idx])
                if frame_indices:
                    scenes[-1]['frame_indices'].extend(frame_indices[start_idx:end_idx])
                continue
            else:
                # If it's the first scene and too short, just include it
                pass
        
        # Create scene embedding by averaging frame embeddings
        scene_embedding = np.mean(embeddings[start_idx:end_idx], axis=0)
        
        scene = {
            'scene_id': i,
            'start_frame': start_idx,
            'end_frame': end_idx,
            'start_timestamp': timestamps[start_idx],
            'end_timestamp': timestamps[end_idx - 1],
            'duration': timestamps[end_idx - 1] - timestamps[start_idx],
            'num_frames': end_idx - start_idx,
            'scene_embedding': scene_embedding,
            'frame_embeddings': embeddings[start_idx:end_idx],
            'captions': captions[start_idx:end_idx] if captions else [],
            'frame_indices': frame_indices[start_idx:end_idx] if frame_indices else list(range(start_idx, end_idx))
        }
        
        scenes.append(scene)
    
    return scenes

def merge_short_scenes(scenes: List[dict], min_duration: float = 2.0) -> List[dict]:
    """
    Merge scenes that are too short with adjacent scenes.
    Works with both regular and multimodal scenes.
    
    Args:
        scenes: List of scene dictionaries
        min_duration: Minimum duration in seconds for a scene
    
    Returns:
        List of merged scenes
    """
    if not scenes:
        return []
    
    merged_scenes = []
    current_scene = scenes[0].copy()
    is_multimodal = 'scene_clip_embedding' in current_scene
    
    for i in range(1, len(scenes)):
        if current_scene['duration'] < min_duration:
            # Merge with next scene
            current_scene['end_frame'] = scenes[i]['end_frame']
            current_scene['end_timestamp'] = scenes[i]['end_timestamp']
            current_scene['duration'] = current_scene['end_timestamp'] - current_scene['start_timestamp']
            current_scene['num_frames'] += scenes[i]['num_frames']
            
            if is_multimodal:
                # Combine multimodal embeddings
                all_clip = np.vstack([
                    current_scene['clip_embeddings'],
                    scenes[i]['clip_embeddings']
                ])
                all_caption = np.vstack([
                    current_scene['caption_embeddings'],
                    scenes[i]['caption_embeddings']
                ])
                
                current_scene['clip_embeddings'] = all_clip
                current_scene['caption_embeddings'] = all_caption
                
                # Recalculate scene-level embeddings
                scene_clip = np.mean(all_clip, axis=0)
                scene_caption = np.mean(all_caption, axis=0)
                
                clip_weight = current_scene.get('clip_weight', 0.6)
                caption_weight = current_scene.get('caption_weight', 0.4)
                
                # Recalculate fused embedding
                scene_fused = fuse_embeddings(scene_clip, scene_caption, clip_weight, caption_weight)
                
                current_scene['scene_embedding'] = scene_fused
                current_scene['scene_clip_embedding'] = scene_clip
                current_scene['scene_caption_embedding'] = scene_caption
            else:
                # Regular embeddings
                all_embeddings = np.vstack([
                    current_scene['frame_embeddings'],
                    scenes[i]['frame_embeddings']
                ])
                current_scene['frame_embeddings'] = all_embeddings
                current_scene['scene_embedding'] = np.mean(all_embeddings, axis=0)
            
            # Combine captions and indices
            current_scene['captions'].extend(scenes[i]['captions'])
            current_scene['frame_indices'].extend(scenes[i]['frame_indices'])
        else:
            # Save current scene and start a new one
            merged_scenes.append(current_scene)
            current_scene = scenes[i].copy()
    
    # Add the last scene
    merged_scenes.append(current_scene)
    
    # Re-assign scene IDs
    for i, scene in enumerate(merged_scenes):
        scene['scene_id'] = i
    
    return merged_scenes

def get_scene_summary(scene: dict) -> str:
    """
    Generate a text summary of a scene from its captions.
    
    Args:
        scene: Scene dictionary
    
    Returns:
        Combined caption summary
    """
    if not scene['captions']:
        return ""
    
    # Remove duplicates while preserving order
    unique_captions = []
    seen = set()
    for caption in scene['captions']:
        if caption not in seen:
            unique_captions.append(caption)
            seen.add(caption)
    
    # Join captions, limit to most representative ones
    if len(unique_captions) <= 3:
        return " | ".join(unique_captions)
    else:
        # Take first, middle, and last caption
        return f"{unique_captions[0]} | {unique_captions[len(unique_captions)//2]} | {unique_captions[-1]}"