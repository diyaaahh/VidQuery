# VidQuery - Video Search Engine

A video search engine that allows you to search through videos using both visual descriptions and spoken dialogue. The system uses CLIP for visual scene search and BERT for caption-based search.

## Features

- 🔍 Scene-based search using natural language descriptions
- 💬 Caption-based search using spoken dialogue
- ⚡ Fast vector similarity search using Qdrant
- 🎯 High-accuracy results using state-of-the-art AI models

## Prerequisites

- Python 3.8+
- Docker (for running Qdrant)
- CUDA-capable GPU (recommended for faster processing)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd VidQuery
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start Qdrant using Docker:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Project Structure

- `main.py` - Main search interface
- `processVideo.py` - Video processing and embedding generation
- `caption.py` - Caption processing and embedding generation
- `captions.txt` - Video captions file
- `Video/` - Directory containing video files

## Usage

### 1. Processing a Video

To process a new video and generate embeddings:

1. Place your video file in the `Video/` directory
2. Run the video processing script:
```bash
python processVideo.py
```

### 2. Processing Captions

To process video captions:
1. Run the caption processing script:
```bash
python caption.py
```

### 3. Searching the Video

Run the main search interface:
```bash
python main.py
```

You can then:
1. Choose search type:
   - Scene-based search (visual description)
   - Dialogue-based search (spoken words)
2. Enter your search query
3. View matching timestamps and scores

## Technical Details

### Models Used

- **CLIP (ViT-B/32)**: For visual scene understanding and search
- **BERT (all-MiniLM-L6-v2)**: For caption-based search
- **Qdrant**: Vector database for efficient similarity search
