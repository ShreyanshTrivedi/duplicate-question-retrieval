FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only torch first. The default Linux wheel pulls in several GB of CUDA
# libraries that a CPU-only Streamlit app never uses. sentence-transformers
# then finds torch already installed and does not replace it.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the NLTK tokenizer and stopword data and the sentence-transformers
# model into the image, so the container never needs internet at runtime.
# raise_on_error makes a failed download fail the build instead of being silent.
RUN python -c "import nltk; [nltk.download(p, download_dir='/usr/local/share/nltk_data', raise_on_error=True) for p in ['punkt', 'punkt_tab', 'stopwords']]" \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Offline at runtime: the model is already in HF_HOME.
ENV HF_HUB_OFFLINE=1

COPY config.yaml ./
COPY .streamlit/ ./.streamlit/
COPY src/ ./src/
COPY reports/ ./reports/

# data/ is NOT in the image. Mount it at runtime (docs/DEPLOYMENT.md).
EXPOSE 8501

CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
