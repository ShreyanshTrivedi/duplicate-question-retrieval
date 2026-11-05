# Deployment (local Docker)

The image runs the Streamlit search UI on port 8501. It is not tied to any cloud provider.

**Before the first run:** `data/processed/` must be generated locally with the data-prep pipeline (`python -m src.data.build_corpus`, then one UI or evaluation launch to create `embeddings.npy`). It is gitignored and is mounted into the container, not baked into the image.

## 1. Build the image

From the repo root:

```
docker build -t duplicate-question-retrieval:local .
```

The build installs CPU-only PyTorch, the Python requirements, and the NLTK data (`punkt`, `punkt_tab`, `stopwords`) and the `all-MiniLM-L6-v2` model. Both are baked into the image, so the running container does not need internet access. The first build takes several minutes because of the downloads.

## 2. Prepare the data once (on the host, not in the image)

The container reads `data/processed/` and `reports/` at runtime, and it expects them to exist on the host:

```
python -m src.data.download
python -m src.data.build_corpus
```

Then create the embeddings cache by launching the app once locally (`streamlit run src/ui/app.py`), or accept that the container will encode 100,000 questions on its first start.

## 3. Run the container

```
docker run -p 8501:8501 -v ${PWD}/data:/app/data -v ${PWD}/reports:/app/reports duplicate-question-retrieval:local
```

In Git Bash (MINGW64), prefix the command with `MSYS_NO_PATHCONV=1`. Without it, Git Bash rewrites `/app/data` to `C:\Program Files\Git\app\data`, the data is mounted in the wrong place, and the app fails with `FileNotFoundError: data/processed/corpus.parquet`:
```
MSYS_NO_PATHCONV=1 docker run -p 8501:8501 -v ${PWD}/data:/app/data -v ${PWD}/reports:/app/reports duplicate-question-retrieval:local
```

Then open http://localhost:8501.

- `${PWD}` is the repo root. In PowerShell it expands to the current directory, and the `MSYS_NO_PATHCONV` prefix is not needed. In `cmd.exe` use `%cd%` in its place.
- Both mounts are required. Without `data/`, the app fails at startup because `corpus.parquet` is missing. The `reports/` mount is for the Performance Comparison and Precision-Recall tabs.
- Startup takes several minutes. The app fits TF-IDF and BM25 over 537,929 questions before the Search tab is ready. Logs appear in the terminal.

## Notes

- `HF_HUB_OFFLINE=1` is set in the image, so the sentence-transformers model is loaded from the baked-in cache and no network call is made.
- The image does not contain `data/`, `docs/`, `tests/`, or the reporting scripts. See `.dockerignore`.
- To stop the container, press Ctrl+C in the terminal, or run `docker ps` and then `docker stop <container-id>`.
