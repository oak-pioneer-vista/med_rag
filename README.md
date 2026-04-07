# med_rag

Medical RAG (Retrieval-Augmented Generation) pipeline built on the MTSamples medical transcription dataset and Qdrant vector database.

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- [Kaggle API credentials](https://www.kaggle.com/docs/api) at `~/.kaggle/kaggle.json`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Download the dataset

```bash
python scripts/download_mtsamples.py
```

Downloads the [MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) dataset into `data/mtsamples/`.

### 2. Start Qdrant

```bash
docker compose up -d
```

Runs Qdrant on `localhost:6333` (REST) and `localhost:6334` (gRPC). Data is persisted in a Docker volume.

### 3. Ingest into Qdrant

```bash
python scripts/ingest_mtsamples.py
```

Embeds medical transcriptions using `all-MiniLM-L6-v2` and upserts them into a `mtsamples` Qdrant collection with metadata (specialty, description, keywords).
