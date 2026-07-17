# Container for the hosted demo (Hugging Face Spaces, Docker SDK).
# Serves the Gradio UI at / and the FastAPI endpoints (/ask, /search,
# /health, /docs) from one uvicorn process. See api.py.

FROM python:3.11-slim

# HF Spaces convention: run as a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# Install deps first so this layer is cached across code-only rebuilds
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image: downloads happen at build time,
# not at container startup, so cold starts skip the HF Hub round-trip
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
