"""Fast GPU rebuild of the FULL medical index: python ingest/build_index_fast.py

Re-embeds textbooks (125,847) + StatPearls (~365k) into chroma_med/ with
sentence-transformers nomic-embed-text-v1.5 on CUDA — batched (256, OOM fallback
128/64), normalized, with the nomic "search_document: " prefix. Each batch is
embedded on GPU then written with a SINGLE collection.add() of precomputed
vectors (never one add per snippet). ~10x faster than the per-snippet Ollama path.

This SUPERSEDES the Ollama-embedded index: those vectors live in a different
space, so a ".embedder" marker triggers a one-time wipe+rebuild. Re-runs after the
marker matches are resumable (existing ids are skipped). The retriever embeds
queries with the same model+prefix (agent/embeddings.py), keeping spaces aligned.

  --rebuild         force the wipe even if the marker matches
  --clinical-first  embed StatPearls clinical articles first (early verifiability)
  --limit N         only N records (smoke test)
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# reduce CUDA fragmentation spikes on the 6 GB card (must be set before torch init)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import medical_index
from agent.embeddings import get_medical_embeddings
from ingest.build_medical_index import iter_statpearls, iter_textbooks
from ingest.build_statpearls import CHUNK_DIR, _ordered_files

# per-embedder marker tag + default VRAM budget (chosen to hold ~3.4 GB on the 6 GB
# card; the longer/denser real corpus made 100M overshoot to 5.6 GB for nomic)
EMBEDDER = {
    "general": {"tag": "nomic-st-v1.5", "budget": 50_000_000},
    # MedCPT (BERT-base, CLS pool) is lighter+faster than nomic: 50M -> 70 emb/sec
    # @ 1.9 GB peak. Higher budgets are slightly SLOWER (100M -> 61/sec) and use
    # more VRAM for no gain (compute-bound) — its measured optimum is 50M.
    "medcpt": {"tag": "medcpt", "budget": 50_000_000},
}


def _open(reset, med_dir, med_collection, kind):
    from langchain_chroma import Chroma

    def fresh():
        return Chroma(
            collection_name=med_collection,
            persist_directory=str(med_dir),
            embedding_function=get_medical_embeddings(kind),
        )

    vs = fresh()
    if reset and vs._collection.count():
        print(f"Wiping {vs._collection.count():,} stale vectors (different embedder)")
        vs.delete_collection()
        vs = fresh()
    return vs


def _record_stream(clinical_first):
    statpearls_files = _ordered_files(clinical_first)
    if clinical_first:
        # clinical StatPearls (already ordered first) -> textbooks -> rest, so
        # clinical content is queryable well before the full ~491k finishes
        return itertools.chain(
            iter_statpearls(CHUNK_DIR, files=statpearls_files),
            iter_textbooks(),
        )
    return itertools.chain(iter_textbooks(), iter_statpearls(CHUNK_DIR, files=statpearls_files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=("general", "medcpt"), default="general")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--clinical-first", action="store_true")
    ap.add_argument("--batch-size", type=int, default=512, help="max sub-batch (bucketing caps it per length)")
    # VRAM budget (char^2*count). Defaults per embedder hold ~3.4 GB on the 6 GB
    # GTX 1660; the longer/denser real corpus made nomic's 100M overshoot to 5.6 GB
    # and OOM-crash. Higher budgets page on the 6 GB ceiling and are SLOWER
    # (120M->17/s) — do not raise to chase VRAM. 0 = per-embedder default.
    ap.add_argument("--budget", type=int, default=0, help="char^2*count budget; 0=embedder default")
    ap.add_argument("--buffer", type=int, default=8192, help="records sorted+encoded per Chroma add")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    spec = EMBEDDER[args.embedder]
    tag = spec["tag"]
    budget = args.budget or spec["budget"]
    med_dir, med_collection = medical_index(args.embedder)
    marker = med_dir / ".embedder"

    med_dir.mkdir(parents=True, exist_ok=True)
    marker_ok = marker.exists() and marker.read_text().strip() == tag
    reset = args.rebuild or not marker_ok
    if reset:
        print(f"Fresh {args.embedder} rebuild (marker_ok={marker_ok}, --rebuild={args.rebuild})")

    emb = get_medical_embeddings(args.embedder)
    vs = _open(reset, med_dir, med_collection, args.embedder)
    marker.write_text(tag)
    col = vs._collection

    skip = set(col.get(include=[])["ids"]) if not reset else set()
    if skip:
        print(f"Resuming: {len(skip):,} vectors already embedded with {tag}")

    import torch

    records = _record_stream(args.clinical_first)
    if args.limit:
        records = itertools.islice(records, args.limit)

    added = 0
    buf = []
    t0 = time.perf_counter()

    def flush():
        """Length-bucketed GPU encode of the buffer + one Chroma add."""
        nonlocal added
        if not buf:
            return
        vecs = emb.encode_bucketed(
            [r["text"] for r in buf], budget=budget, max_bs=args.batch_size
        ).tolist()
        # chromadb caps a single add() at 5461 rows; chunk the write
        for s in range(0, len(buf), 5000):
            sub = buf[s : s + 5000]
            col.add(
                ids=[r["id"] for r in sub],
                embeddings=vecs[s : s + 5000],
                documents=[r["text"] for r in sub],
                metadatas=[r["metadata"] for r in sub],
            )
        added += len(buf)
        el = time.perf_counter() - t0
        print(
            f"  embedded {added:>7,} (+{len(skip):,} prior) @ "
            f"{added / el if el else 0:,.0f} emb/sec | "
            f"peakVRAM {torch.cuda.max_memory_allocated() / 1e6:,.0f} MB | {el / 60:.1f} min",
            flush=True,
        )
        buf.clear()

    for rec in records:
        if rec["id"] in skip:
            continue
        buf.append(rec)
        if len(buf) >= args.buffer:
            flush()
    flush()

    total = col.count()
    el = time.perf_counter() - t0
    print(
        f"\nDone: added {added:,} this run @ {added / el if el else 0:,.0f} emb/sec "
        f"in {el / 60:.1f} min. chroma_med/ total now {total:,}."
    )


if __name__ == "__main__":
    main()
