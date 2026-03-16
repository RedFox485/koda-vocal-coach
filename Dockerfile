FROM python:3.11-slim

WORKDIR /app

# libsndfile1 = only dependency librosa actually needs on Linux
# No ffmpeg — we receive raw PCM from browser, never load audio files
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# torch CPU-only wheel (~700MB vs 2GB for CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# App deps
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# App code
COPY src/ src/
COPY models/ models/
COPY frontend/ frontend/
COPY config/ config/

# Cloud Run sets PORT env var; default to 8080
ENV PORT=8080
EXPOSE 8080

CMD ["python", "src/vocal_health_backend.py", "--port", "8080"]
