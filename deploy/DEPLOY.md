# Deployment

Two pieces: the FastAPI backend on a Hugging Face **Docker** Space, the Next.js
frontend on Vercel. (Fallback: skip both and run `app/gradio_app.py` inside the
same Space — see the note at the bottom.)

## Backend → Hugging Face Spaces (Docker)

The image bundles Ollama + nomic-embed-text for query embeddings (the Chroma
index was built with that model, so the embedder cannot be swapped without
reindexing). Generation uses Groq — no heavy chat model in the container.

```bash
# 1. Create the Space (type: Docker) — once
#    https://huggingface.co/new-space  → SDK: Docker → name: medical-rag-api

# 2. Clone it next to this repo
git clone https://huggingface.co/spaces/<your-user>/medical-rag-api hf-space
cd hf-space

# 3. Copy the backend pieces from this repo
cp -r ../medical-rag-agent/{Dockerfile,requirements-docker.txt,agent,app} .
cp ../medical-rag-agent/deploy/hf-space-README.md README.md   # Space front-matter lives here

# 4. The Chroma index is too big for plain git — use LFS
huggingface-cli lfs-enable-largefiles .
cp -r ../medical-rag-agent/chroma_med .
git lfs track "chroma_med/**"
git add .gitattributes . && git commit -m "Deploy medical-rag API" && git push

# 5. In the Space settings:
#    Secrets:   GROQ_API_KEY  (+ LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST if tracing)
#    Variables: LLM_PROVIDER=groq
#               FRONTEND_ORIGIN=https://<your-frontend>.vercel.app

# 6. Smoke test once it builds
curl https://<your-user>-medical-rag-api.hf.space/health
```

## Frontend → Vercel

Only one env var matters: `NEXT_PUBLIC_API_BASE_URL` → the Space URL.

```bash
cd frontend
npm i -g vercel        # or use the Vercel dashboard import flow
vercel                 # link / create the project (root = frontend/)
vercel env add NEXT_PUBLIC_API_BASE_URL production
#   value: https://<your-user>-medical-rag-api.hf.space
vercel --prod
```

Then set the backend's `FRONTEND_ORIGIN` variable to the deployed Vercel URL
(CORS) and restart the Space.

## Fallback: Gradio-only (fastest path)

On a 1-day budget, skip Vercel entirely: add `gradio` + `requests` to
`requirements-docker.txt`, change the Dockerfile CMD to also start
`python app/gradio_app.py`, and expose 7861 — or simplest of all, run the
Gradio app as the Space's only process and call `run_agent()` in-process.
One deployable, no CORS, no Node toolchain.
