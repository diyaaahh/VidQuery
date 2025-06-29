import torch
import clip
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import numpy as np

# Constants
CLIP_COLLECTION = "video-frames-clip"
CAPTION_COLLECTION = "video-frames-captions"
AUDIO_COLLECTION = "video-audio-text"  # For Whisper transcriptions

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant
qdrant = QdrantClient("localhost", port=6333)

# CLIP embedding for visual search
def get_clip_embedding(query):
    text_token = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_token).squeeze().cpu().numpy()
    return text_embedding

# Text embedding for caption and audio search
def get_text_embedding(query):
    return bert_model.encode(query)

# Generic search function
def search_qdrant(collection_name, embedding, top_k=5):
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
        limit=top_k
    )
    return results

# Hybrid search: combines CLIP and caption results
def hybrid_search(query, top_k=5, clip_weight=0.6, caption_weight=0.4):
    # Get embeddings
    clip_embedding = get_clip_embedding(query)
    text_embedding = get_text_embedding(query)
    
    # Search both collections
    clip_results = search_qdrant(CLIP_COLLECTION, clip_embedding, top_k * 2)
    caption_results = search_qdrant(CAPTION_COLLECTION, text_embedding, top_k * 2)
    
    # Combine results with weighted scores
    combined_results = {}
    
    # Add CLIP results
    for result in clip_results:
        timestamp = result.payload['timestamp']
        combined_results[timestamp] = {
            'score': result.score * clip_weight,
            'payload': result.payload,
            'sources': ['visual']
        }
    
    # Add caption results
    for result in caption_results:
        timestamp = result.payload['timestamp']
        if timestamp in combined_results:
            # Combine scores if same timestamp
            combined_results[timestamp]['score'] += result.score * caption_weight
            combined_results[timestamp]['sources'].append('caption')
            combined_results[timestamp]['payload']['caption'] = result.payload.get('caption', '')
        else:
            combined_results[timestamp] = {
                'score': result.score * caption_weight,
                'payload': result.payload,
                'sources': ['caption']
            }
    
    # Sort by combined score and return top results
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1]['score'], reverse=True)
    return sorted_results[:top_k]

def main():
    print("🎬 Smart Video Search Engine")
    print("Choose search type:")
    print("1. Hybrid search (visual + caption)")
    print("2. Audio text search (Whisper transcriptions)")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        query = input("Enter your search query (e.g., 'person drinking coffee'): ")
        
        print(f"\n🔍 Searching for: '{query}'")
        print("🔄 Performing hybrid search (visual + caption)...")
        
        results = hybrid_search(query, top_k=5)
        
        print("\n📊 Top matching results:")
        for i, (timestamp, data) in enumerate(results, 1):
            sources = " + ".join(data['sources'])
            print(f"\n{i}. ⏱️ Timestamp: {timestamp}s | Score: {data['score']:.4f} | Sources: {sources}")
            
            # Show caption if available
            if 'caption' in data['payload']:
                print(f"   💬 Caption: {data['payload']['caption']}")
    
    elif choice == "2":
        query = input("Enter text to search in audio (e.g., 'add sugar'): ")
        
        print(f"\n🔍 Searching audio transcriptions for: '{query}'")
        
        # Search audio collection
        text_embedding = get_text_embedding(query)
        results = search_qdrant(AUDIO_COLLECTION, text_embedding, top_k=5)
        
        
        print("\n🎵 Top matching audio segments:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. ⏱️ Timestamp: {result.payload['timestamp']}s | Score: {result.score:.4f}")
            print(f"   🎤 Transcription: {result.payload.get('transcription', 'N/A')}")
    
    else:
        print("❌ Invalid option. Please run the script again and choose 1 or 2.")

if __name__ == "__main__":
    main()