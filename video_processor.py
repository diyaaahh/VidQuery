import cv2
import torch
import clip
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import time
from typing import List, Tuple
import gc

# Import our utility functions
from video_utils import get_video_id, get_collection_names
from scene_segmentation import create_scenes_with_multimodal, merge_short_scenes, get_scene_summary

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.to(device)
blip_model.eval()

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant
qdrant = QdrantClient("localhost", port=6333)

def setup_qdrant_for_video(video_id: str):
    """Setup Qdrant collections for a specific video"""
    collections = get_collection_names(video_id)
    existing = [c.name for c in qdrant.get_collections().collections]
    
    for collection_type, collection_name in collections.items():
        if collection_name not in existing:
            if collection_type == 'clip':
                size = 512  # CLIP embedding size
            elif collection_type == 'scenes':
                size = 512  # Scene embeddings (fused CLIP+Caption, normalized)
            else:  # caption
                size = 384  # Sentence transformer size
                
            qdrant.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )
            print(f"Created collection: {collection_name}")

def extract_frames(video_path: str, frame_interval: int = 2) -> List[Tuple[Image.Image, float]]:
    """Extract frames from video at specified intervals"""
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_id = 0
    frames_data = []
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        if frame_id % (frame_rate * frame_interval) == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            frames_data.append((img, timestamp))
        
        frame_id += 1
    
    cap.release()
    return frames_data

def generate_captions_batch(images: List[Image.Image], batch_size: int = 4) -> List[str]:
    """Generate captions using BLIP"""
    captions = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            inputs = blip_processor(batch_images, return_tensors="pt").to(device)
            
            generated_ids = blip_model.generate(
                **inputs,
                max_length=25,
                num_beams=3,
                do_sample=False,
                early_stopping=True
            )
            
            batch_captions = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
            captions.extend(batch_captions)
            
            if device == "cuda":
                torch.cuda.empty_cache()
            
            del inputs, generated_ids
            gc.collect()
    
    return captions

def generate_clip_embeddings(images: List[Image.Image]) -> List[np.ndarray]:
    """Generate CLIP embeddings for images"""
    embeddings = []
    
    with torch.no_grad():
        for img in images:
            image_input = clip_preprocess(img).unsqueeze(0).to(device)
            embedding = clip_model.encode_image(image_input).squeeze().cpu().numpy()
            embeddings.append(embedding)
    
    return embeddings

def generate_caption_embeddings(captions: List[str]) -> List[np.ndarray]:
    """Generate sentence transformer embeddings for captions"""
    embeddings = sentence_model.encode(captions, convert_to_numpy=True)
    return embeddings

def process_video_hybrid_embeddings(
    video_path: str, 
    frame_interval: int = 2, 
    batch_size: int = 4,
    scene_threshold: float = 0.85,
    min_scene_duration: float = 2.0,
    enable_scenes: bool = True,
    clip_weight: float = 0.6,
    caption_weight: float = 0.4
):
    """Process video with multimodal scene segmentation and fusion"""
    video_id = get_video_id(video_path)
    collections = get_collection_names(video_id)
    
    print(f"Processing video with ID: {video_id}")
    print(f"Collections: {collections}")
    print(f"Scene segmentation: {'Enabled' if enable_scenes else 'Disabled'}")
    print(f"Multimodal fusion weights: CLIP={clip_weight}, Caption={caption_weight}")
    
    # Setup video-specific collections
    setup_qdrant_for_video(video_id)
    
    # Extract frames
    frames_data = extract_frames(video_path, frame_interval)
    images = [frame_data[0] for frame_data in frames_data]
    timestamps = [frame_data[1] for frame_data in frames_data]
    
    if not images:
        print("No frames extracted!")
        return video_id
    
    # Generate captions and embeddings
    print(f"Generating captions for {len(images)} frames...")
    captions = generate_captions_batch(images, batch_size)
    
    print("Generating CLIP embeddings...")
    clip_embeddings = generate_clip_embeddings(images)
    clip_embeddings_array = np.array(clip_embeddings)
    
    print("Generating caption embeddings...")
    caption_embeddings = generate_caption_embeddings(captions)
    caption_embeddings_array = np.array(caption_embeddings)
    
    # Upload frame-level CLIP embeddings
    clip_points = []
    for i, (embedding, timestamp) in enumerate(zip(clip_embeddings, timestamps)):
        clip_points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "timestamp": timestamp,
                    "frame_index": i,
                    "video_path": video_path,
                    "video_id": video_id,
                    "type": "frame"
                }
            )
        )
    
    qdrant.upsert(collection_name=collections['clip'], points=clip_points)
    print(f"Uploaded {len(clip_points)} CLIP embeddings to {collections['clip']}")
    
    # Upload caption embeddings
    caption_points = []
    for i, (embedding, caption, timestamp) in enumerate(zip(caption_embeddings, captions, timestamps)):
        caption_points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "timestamp": timestamp,
                    "caption": caption,
                    "frame_index": i,
                    "video_path": video_path,
                    "video_id": video_id,
                    "type": "frame"
                }
            )
        )
    
    qdrant.upsert(collection_name=collections['caption'], points=caption_points)
    print(f"Uploaded {len(caption_points)} caption embeddings to {collections['caption']}")
    
    # Multimodal Scene Segmentation
    if enable_scenes:
        print("\n=== Multimodal Scene Segmentation ===")
        print(f"Detecting scenes with threshold: {scene_threshold}")
        
        scenes = create_scenes_with_multimodal(
            clip_embeddings=clip_embeddings_array,
            caption_embeddings=caption_embeddings_array,
            timestamps=timestamps,
            captions=captions,
            frame_indices=list(range(len(images))),
            threshold=scene_threshold,
            min_scene_length=3,
            clip_weight=clip_weight,
            caption_weight=caption_weight
        )
        
        print(f"Detected {len(scenes)} initial scenes")
        
        # Merge short scenes
        scenes = merge_short_scenes(scenes, min_duration=min_scene_duration)
        print(f"After merging: {len(scenes)} scenes")
        
        # Upload scene embeddings with multimodal fusion
        scene_points = []
        for scene in scenes:
            scene_summary = get_scene_summary(scene)
            
            scene_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=scene['scene_embedding'].tolist(),  # Fused embedding
                    payload={
                        "scene_id": scene['scene_id'],
                        "start_timestamp": scene['start_timestamp'],
                        "end_timestamp": scene['end_timestamp'],
                        "duration": scene['duration'],
                        "num_frames": scene['num_frames'],
                        "start_frame": scene['start_frame'],
                        "end_frame": scene['end_frame'],
                        "scene_summary": scene_summary,
                        "captions": scene['captions'],
                        "video_path": video_path,
                        "video_id": video_id,
                        "type": "scene",
                        "fusion_type": "multimodal",
                        "clip_weight": clip_weight,
                        "caption_weight": caption_weight
                    }
                )
            )
        
        qdrant.upsert(collection_name=collections['scenes'], points=scene_points)
        print(f"Uploaded {len(scene_points)} multimodal scene embeddings to {collections['scenes']}")
        
        # Print scene information
        print("\nScene Summary:")
        for scene in scenes:
            print(f"  Scene {scene['scene_id']}: "
                  f"{scene['start_timestamp']:.1f}s - {scene['end_timestamp']:.1f}s "
                  f"({scene['duration']:.1f}s, {scene['num_frames']} frames)")
            print(f"    Summary: {get_scene_summary(scene)[:100]}...")
    
    print(f"\nVideo processing completed for {video_id}: {len(images)} frames processed")
    if enable_scenes:
        print(f"Scenes created: {len(scenes)} (with multimodal fusion)")
    
    return video_id