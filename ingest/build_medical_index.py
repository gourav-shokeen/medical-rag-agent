"""Build the medical Chroma index: python ingest/build_medical_index.py

Sources:
  - MedRAG/textbooks (HF Hub): 125,847 pre-chunked USMLE-textbook snippets.
  - StatPearls: NOT distributable via the HF Hub (MedRAG/statpearls is an empty
    repo for licensing reasons — verified). If you build the chunks locally with
    the MedRAG toolkit (https://github.com/Teddy-XiongGZ/MedRAG, statpearls.py),
    pass --statpearls-dir pointing at its chunk/*.jsonl output and they are
    ingested with source="statpearls".

Embeddings: the SAME model as the finance index (Ollama nomic-embed-text), so
the agent's query embedder stays identical across corpora.

Properties: batched (one embed call per batch), resumable (picks up at
collection.count(); ids are deterministic and writes are upserts, so overlap is
harmless), idempotent (skips when complete unless --rebuild). GPU/CPU is
handled by the Ollama server. Prints throughput-based ETA and final count.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # cp1252 console + medical text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import EMBEDDING_MODEL, MEDICAL_CHROMA_DIR, MEDICAL_COLLECTION


def get_vectorstore(retriever=None):
    """Shared medical Chroma handle for the active (or given) retriever kind."""
    from langchain_chroma import Chroma

    from agent.config import medical_index
    from agent.embeddings import get_medical_embeddings

    med_dir, med_collection = medical_index(retriever)
    return Chroma(
        collection_name=med_collection,
        persist_directory=str(med_dir),
        embedding_function=get_medical_embeddings(retriever),
    )


def batched_add(vectorstore, records, *, batch_size=128, total=None, skip_ids=None,
                label="ingest"):
    """Stream `records` (dicts: id/text/metadata) into Chroma in batches.

    RAM-light (consumes an iterator, never materializes everything), resumable
    (skip_ids are not re-embedded), idempotent (ids are deterministic upserts).
    Prints throughput + ETA. Returns the number of records added.
    """
    skip = skip_ids or set()
    added = 0
    batch = []
    t0 = time.perf_counter()

    def flush():
        nonlocal added
        if not batch:
            return
        vectorstore.add_texts(
            texts=[r["text"] for r in batch],
            metadatas=[r["metadata"] for r in batch],
            ids=[r["id"] for r in batch],
        )
        added += len(batch)
        batch.clear()

    for rec in records:
        if rec["id"] in skip:
            continue
        batch.append(rec)
        if len(batch) >= batch_size:
            flush()
            if (added // batch_size) % 10 == 0:
                elapsed = time.perf_counter() - t0
                rate = added / elapsed if elapsed else 0
                eta = (total - added) / rate / 60 if (total and rate) else 0
                tot = f"/{total:,}" if total else ""
                print(
                    f"  [{label}] {added:>7,}{tot} ({rate:,.0f}/s, ETA {eta:,.1f} min)",
                    flush=True,
                )
    flush()
    print(f"  [{label}] added {added:,} in {(time.perf_counter() - t0) / 60:.1f} min")
    return added


def iter_textbooks():
    from datasets import load_dataset

    ds = load_dataset("MedRAG/textbooks", split="train")
    for row in ds:
        # 'contents' is the title-prefixed snippet (what MedRAG embeds);
        # fall back to building it if absent
        text = row.get("contents") or f"{row['title']}. {row['content']}"
        yield {
            "id": f"textbook-{row['id']}",
            "text": text,
            "metadata": {
                "source": "textbook",
                "title": row["title"],
                "snippet_id": row["id"],
            },
        }


def read_statpearls_file(fp: Path):
    """Yield ingest records from one StatPearls chunk JSONL file."""
    for line in fp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("contents") or f"{row.get('title', '')}. {row.get('content', '')}"
        yield {
            "id": f"statpearls-{row['id']}",
            "text": text,
            "metadata": {
                "source": "statpearls",
                "title": row.get("title", "StatPearls"),
                "snippet_id": row["id"],
            },
        }


def iter_statpearls(chunk_dir: Path, files=None):
    """Stream StatPearls records. `files` overrides/ orders the file list."""
    for fp in (files if files is not None else sorted(chunk_dir.glob("*.jsonl"))):
        yield from read_statpearls_file(fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="drop and re-ingest")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="ingest only N (testing)")
    parser.add_argument("--statpearls-dir", type=Path, default=None)
    args = parser.parse_args()

    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    records = list(iter_textbooks())
    if args.statpearls_dir:
        if not args.statpearls_dir.is_dir():
            sys.exit(f"--statpearls-dir {args.statpearls_dir} is not a directory")
        records += list(iter_statpearls(args.statpearls_dir))
    else:
        print(
            "NOTE: ingesting MedRAG/textbooks only. StatPearls cannot be "
            "downloaded from the HF Hub (licensing); build it locally with the "
            "MedRAG toolkit and pass --statpearls-dir to include it."
        )
    if args.limit:
        records = records[: args.limit]
    total = len(records)
    print(f"Corpus to ingest: {total:,} snippets")

    vectorstore = Chroma(
        collection_name=MEDICAL_COLLECTION,
        persist_directory=str(MEDICAL_CHROMA_DIR),
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    )

    existing = vectorstore._collection.count()
    if args.rebuild and existing:
        print(f"--rebuild: dropping {existing:,} existing vectors")
        vectorstore.delete_collection()
        vectorstore = Chroma(
            collection_name=MEDICAL_COLLECTION,
            persist_directory=str(MEDICAL_CHROMA_DIR),
            embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        )
        existing = 0

    if existing >= total:
        print(f"Index already complete ({existing:,} >= {total:,}) — nothing to do.")
    else:
        if existing:
            print(f"Resuming at offset {existing:,} (deterministic order + upserts)")
        todo = records[existing:]
        done = 0
        t0 = time.perf_counter()
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i : i + args.batch_size]
            vectorstore.add_texts(
                texts=[r["text"] for r in batch],
                metadatas=[r["metadata"] for r in batch],
                ids=[r["id"] for r in batch],
            )
            done += len(batch)
            elapsed = time.perf_counter() - t0
            rate = done / elapsed
            remaining = (len(todo) - done) / rate if rate else 0
            if i // args.batch_size % 10 == 0 or done == len(todo):
                print(
                    f"  {existing + done:>7,}/{total:,} "
                    f"({rate:,.0f} snippets/s, ETA {remaining / 60:,.1f} min)",
                    flush=True,
                )
        print(f"Ingest finished in {(time.perf_counter() - t0) / 60:.1f} min")

    final = vectorstore._collection.count()
    print(f"Final collection count: {final:,}")

    print("\nSample similarity query: 'first-line treatment for community-acquired pneumonia'")
    for d in vectorstore.similarity_search(
        "first-line treatment for community-acquired pneumonia", k=3
    ):
        print(f"  [{d.metadata['source']}: {d.metadata['title']}] {d.page_content[:110]}")


if __name__ == "__main__":
    main()
