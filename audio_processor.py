import os
import uuid
import whisper
import ffmpeg
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Import our new utility functions
from video_utils import get_video_id, get_collection_names

# Load BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant
qdrant = QdrantClient("localhost", port=6333)

def setup_qdrant_for_audio(video_id: str):
    """Create Qdrant collection for audio if it doesn't exist"""
    collections = get_collection_names(video_id)
    existing = [c.name for c in qdrant.get_collections().collections]
    
    if collections['audio'] not in existing:
        qdrant.recreate_collection(
            collection_name=collections['audio'],
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"Created audio collection: {collections['audio']}")

def extract_audio(video_path, audio_path="temp_audio.wav"):
    """Extracts audio from the video using ffmpeg"""
    ffmpeg.input(video_path).output(audio_path, ac=1, ar='16k').run(overwrite_output=True)
    return audio_path

def transcribe_with_whisper(audio_path):
    """Runs Whisper on the extracted audio"""
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["segments"]

def save_captions(segments, output_file="captions.txt"):
    """Saves the transcribed segments to a file"""
    with open(output_file, "w", encoding="utf-8") as f:
        for seg in segments:
            start = round(seg['start'], 2)
            end = round(seg['end'], 2)
            text = seg['text'].strip()
            f.write(f"[{start}s - {end}s] {text}\n")
    print(f"Saved captions to {output_file}")

def upload_caption_embeddings(segments, video_path):
    """Generate and upload embeddings from whisper segments to video-specific collection"""
    video_id = get_video_id(video_path)
    collections = get_collection_names(video_id)
    
    texts = [seg['text'].strip() for seg in segments]
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "timestamp": seg['start'],
                "caption": seg['text'].strip(),
                "video_path": video_path,
                "video_id": video_id
            }
        )
        for seg, embedding in zip(segments, embeddings)
    ]
    
    qdrant.upsert(collection_name=collections['audio'], points=points)
    print(f"Uploaded {len(points)} audio caption embeddings to {collections['audio']}")

def process_video_for_audio_captions(video_path, output_file="captions.txt"):
    """Complete audio processing pipeline with per-video collections"""
    video_id = get_video_id(video_path)
    
    print(f"Processing audio for video ID: {video_id}")
    
    # Setup video-specific audio collection
    setup_qdrant_for_audio(video_id)
    
    audio_path = extract_audio(video_path)
    segments = transcribe_with_whisper(audio_path)
    save_captions(segments, output_file)
    upload_caption_embeddings(segments, video_path)
    
    # Clean up temp audio
    os.remove(audio_path)
    print(f"Audio processing completed for video {video_id}")
    return video_id