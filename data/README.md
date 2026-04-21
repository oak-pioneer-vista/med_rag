This directory is used for:

- **Downloaded datasets** — MTSamples CSV downloaded via `python/ingestion/mtsamples/download_mtsamples.py`
- **Docker persistent data** — Qdrant storage and other service volumes

Contents are gitignored. To populate, run:

```bash
python python/ingestion/mtsamples/download_mtsamples.py
docker compose up -d
```
