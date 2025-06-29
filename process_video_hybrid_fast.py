# process_video_hybrid_fast.py
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
import os
from typing import List, Tuple
import gc

# Collection names
CLIP_COLLECTION = "video-frames-clip"
CAPTION_COLLECTION = "video-frames-captions"

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load models
print("Loading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

print("Loading BLIP1 model...")
# Use BLIP1 base model - much faster than BLIP2
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Move BLIP to device
blip_model.to(device)
blip_model.eval()

print("Loading sentence transformer...")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant
qdrant = QdrantClient("localhost", port=6333)

def setup_qdrant():
    """Setup Qdrant collections for both CLIP and caption embeddings"""
    collections = qdrant.get_collections().collections
    collection_names = [c.name for c in collections]
    
    # CLIP embeddings collection (512 dimensions)
    if CLIP_COLLECTION not in collection_names:
        qdrant.recreate_collection(
            collection_name=CLIP_COLLECTION,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        print(f"Created collection: {CLIP_COLLECTION}")
    
    # Caption embeddings collection (384 dimensions for all-MiniLM-L6-v2)
    if CAPTION_COLLECTION not in collection_names:
        qdrant.recreate_collection(
            collection_name=CAPTION_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"Created collection: {CAPTION_COLLECTION}")

def extract_frames(video_path: str, frame_interval: int = 2) -> List[Tuple[Image.Image, float]]:
    """Extract frames from video at specified intervals"""
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_id = 0
    frames_data = []
    
    print(f"Video FPS: {frame_rate}, extracting every {frame_interval} second(s)")
    
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
    print(f"Extracted {len(frames_data)} frames")
    return frames_data

def generate_captions_batch(images: List[Image.Image], batch_size: int = 4) -> List[str]:
    """Generate captions using BLIP1 - much faster than BLIP2"""
    captions = []
    total_batches = (len(images) + batch_size - 1) // batch_size
    
    print(f"Generating captions in {total_batches} batches of size {batch_size}")
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_start_time = time.time()
            
            # Process batch - BLIP1 is much faster with batch processing
            inputs = blip_processor(batch_images, return_tensors="pt").to(device)
            
            # Fast generation with BLIP1
            generated_ids = blip_model.generate(
                **inputs,
                max_length=25,
                num_beams=3,
                do_sample=False,
                early_stopping=True
            )
            
            # Decode captions
            batch_captions = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
            captions.extend(batch_captions)
            
            batch_time = time.time() - batch_start_time
            avg_time_per_image = batch_time / len(batch_images)
            
            print(f"Batch {(i//batch_size)+1}/{total_batches} completed in {batch_time:.2f}s | "
                  f"~{avg_time_per_image:.2f}s per image")
            
            # Clear GPU memory
            if device == "cuda":
                torch.cuda.empty_cache()
            
            del inputs, generated_ids
            gc.collect()
    
    return captions

def generate_clip_embeddings(images: List[Image.Image]) -> List[np.ndarray]:
    """Generate CLIP embeddings for images"""
    print("Generating CLIP embeddings...")
    embeddings = []
    
    with torch.no_grad():
        for i, img in enumerate(images):
            image_input = clip_preprocess(img).unsqueeze(0).to(device)
            embedding = clip_model.encode_image(image_input).squeeze().cpu().numpy()
            embeddings.append(embedding)
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(images)} images for CLIP embeddings")
    
    return embeddings

def generate_caption_embeddings(captions: List[str]) -> List[np.ndarray]:
    """Generate sentence transformer embeddings for captions"""
    print("Generating caption embeddings...")
    embeddings = sentence_model.encode(captions, convert_to_numpy=True)
    return embeddings

def process_video_hybrid(video_path: str, frame_interval: int = 2, batch_size: int = 4):
    """Process video with hybrid approach: CLIP + BLIP1 captions"""
    print(f"Processing video: {video_path}")
    start_time = time.time()
    
    # Step 1: Extract frames
    frames_data = extract_frames(video_path, frame_interval)
    images = [frame_data[0] for frame_data in frames_data]
    timestamps = [frame_data[1] for frame_data in frames_data]
    
    if not images:
        print("No frames extracted!")
        return
    
    print(f"⚡ Processing {len(images)} frames with BLIP1 (much faster than BLIP2)")
    
    # Step 2: Generate captions using BLIP1 (much faster)
    print("\n--- Processing BLIP1 Captions ---")
    captions = generate_captions_batch(images, batch_size)
    
    # Step 3: Generate CLIP embeddings
    print("\n--- Processing CLIP Embeddings ---")
    clip_embeddings = generate_clip_embeddings(images)
    
    # Step 4: Generate caption embeddings
    print("\n--- Processing Caption Embeddings ---")
    caption_embeddings = generate_caption_embeddings(captions)
    
    # Step 5: Upload CLIP embeddings to Qdrant
    print("\n--- Uploading CLIP Embeddings ---")
    clip_points = []
    for i, (embedding, timestamp) in enumerate(zip(clip_embeddings, timestamps)):
        clip_points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "timestamp": timestamp,
                    "frame_index": i,
                    "video_path": video_path
                }
            )
        )
    
    qdrant.upsert(collection_name=CLIP_COLLECTION, points=clip_points)
    print(f"Uploaded {len(clip_points)} CLIP embeddings")
    
    # Step 6: Upload caption embeddings to Qdrant
    print("\n--- Uploading Caption Embeddings ---")
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
                    "video_path": video_path
                }
            )
        )
    
    qdrant.upsert(collection_name=CAPTION_COLLECTION, points=caption_points)
    print(f"Uploaded {len(caption_points)} caption embeddings")
    
    # Display sample captions
    print("\n--- Sample Generated Captions ---")
    for i, (caption, timestamp) in enumerate(zip(captions[:5], timestamps[:5])):
        print(f"⏱️  {timestamp:.1f}s: {caption}")
    
    total_time = time.time() - start_time
    print(f"\n✅ Video processing completed in {total_time:.1f} seconds")
    print(f"📊 Processed {len(images)} frames with {len(captions)} captions")
    print(f"⚡ Average time per frame: {total_time/len(images):.2f}s")

if __name__ == "__main__":
    # Setup Qdrant collections
    setup_qdrant()
    
    # Process video
    video_path = "Video/Coffee.mp4"  # Update with your video path
    
    # Parameters - optimized for speed
    frame_interval = 2  # Extract frame every 2 seconds
    batch_size = 4      # BLIP1 can handle larger batches efficiently
    
    process_video_hybrid(video_path, frame_interval, batch_size)