This directory is used for:

- **Downloaded datasets** — MTSamples CSV downloaded via `scripts/download_mtsamples.py`
- **Docker persistent data** — Qdrant storage and other service volumes

Contents are gitignored. To populate, run:

```bash
python scripts/download_mtsamples.py
docker compose up -d
```
