# Backend image for a Hugging Face Spaces *Docker* Space (or any container host).
#
# Serving needs two models at runtime:
#   - query embeddings: local Ollama nomic-embed-text (the Chroma index was built
#     with it; a different embedder would be dimension-incompatible) — baked in
#   - generation/grading LLM: set LLM_PROVIDER=groq + GROQ_API_KEY as Space
#     secrets (running a local chat model in the Space is too heavy)
#
# The chroma_med/ index is COPYed from the build context. It is gitignored, so
# for HF Spaces push it to the Space repo with Git LFS (see deploy/DEPLOY.md).

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/.huggingface \
    OLLAMA_MODELS=/data/ollama

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ollama for query embeddings only (small CPU model)
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /code

# CPU torch first (the default index would pull multi-GB CUDA wheels)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Bake the embedding model into the image so cold starts don't re-download it
RUN ollama serve & \
    sleep 5 && ollama pull nomic-embed-text && pkill ollama

COPY agent/ agent/
COPY app/ app/
COPY chroma_med/ chroma_med/

# Pre-download the cross-encoder so first request doesn't hit the HF Hub
RUN python -c "from langchain_community.cross_encoders import HuggingFaceCrossEncoder; \
    HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')"

# HF Docker Spaces route traffic to 7860 by default (override with app_port)
EXPOSE 7860
ENV PORT=7860

CMD ["/bin/sh", "-c", "ollama serve & sleep 3 && uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
