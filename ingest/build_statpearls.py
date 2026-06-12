"""Build StatPearls and ADD it to chroma_med/: python ingest/build_statpearls.py

Primary corpus source (matches the MedRAG StatPearls corpus):
  1. Download the StatPearls NXML archive from NCBI Bookshelf collection
     NBK430685 (https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz),
     ~9,330 articles. Cached so a re-run does not re-fetch.
  2. Chunk hierarchically with the EXACT MedRAG algorithm (Teddy-XiongGZ/MedRAG,
     src/data/statpearls.py) so snippets match their corpus: each paragraph is a
     snippet; the spliced "Article -- Section -- Subsection" headings form its
     title. Target ~301k snippets. Output JSONL is compatible with the existing
     --statpearls-dir hook in build_medical_index.py.
  3. Embed + ADD into the existing chroma_med/ (same collection, same
     nomic-embed-text embedder, metadata {source:"statpearls", title, snippet_id}).
     Textbooks are NOT re-embedded. Resumable + idempotent by id; source-scoped
     rebuild via --rebuild-statpearls.

The chunking functions below (ends_with_ending_punctuation, concat, extract_text,
is_subtitle, extract) are reproduced from MedRAG's statpearls.py so the produced
chunks are identical to the published StatPearls corpus.
"""

import argparse
import json
import sys
import tarfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # cp1252 console + medical text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.build_medical_index import (  # noqa: E402  (reuse — do not duplicate)
    batched_add,
    get_vectorstore,
    iter_statpearls,
    read_statpearls_file,
)

ARCHIVE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"
CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "statpearls"
ARCHIVE_PATH = CORPUS_DIR / "statpearls_NBK430685.tar.gz"
NXML_DIR = CORPUS_DIR / "statpearls_NBK430685"
CHUNK_DIR = CORPUS_DIR / "chunk"

# keywords from the Step-3 clinical verification queries — used only to order
# insertion (--clinical-first) so verification is possible before the full ~301k
# finishes embedding. Insertion order does NOT change the final corpus.
CLINICAL_KEYWORDS = (
    "pneumonia", "ketoacidosis", "hypertension", "diabetes", "antihypertensive",
    "community-acquired", "myocardial", "sepsis", "asthma", "antibiotic",
)


# --- MedRAG StatPearls chunker (verbatim from Teddy-XiongGZ/MedRAG) -----------


def ends_with_ending_punctuation(s):
    return any(s.endswith(c) for c in (".", "?", "!"))


def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    return title.strip() + ". " + content.strip()


def extract_text(element):
    text = (element.text or "").strip()
    for child in element:
        text += (" " if len(text) else "") + extract_text(child)
        if child.tail and len(child.tail.strip()) > 0:
            text += (" " if len(text) else "") + child.tail.strip()
    return text.strip()


def is_subtitle(element):
    if element.tag != "p":
        return False
    if len(list(element)) != 1:
        return False
    if list(element)[0].tag != "bold":
        return False
    if list(element)[0].tail and len(list(element)[0].tail.strip()) > 0:
        return False
    return True


def extract(fpath):
    fname = str(fpath).split("/")[-1].split("\\")[-1].replace(".nxml", "")
    tree = ET.parse(fpath)
    title = tree.getroot().find(".//title").text
    sections = tree.getroot().findall(".//sec")
    saved_text = []
    j = 0
    for sec in sections:
        sec_title_node = sec.find("./title")
        if sec_title_node is None or sec_title_node.text is None:
            continue
        sec_title = sec_title_node.text.strip()
        sub_title = ""
        prefix = " -- ".join([title, sec_title])
        last_text = None
        last_json = None
        last_node = None
        for ch in sec:
            if is_subtitle(ch):
                last_text = None
                last_json = None
                sub_title = extract_text(ch)
                prefix = " -- ".join(prefix.split(" -- ")[:2] + [sub_title])
            elif ch.tag == "p":
                curr_text = extract_text(ch)
                if len(curr_text) < 200 and last_text is not None and len(last_text + curr_text) < 1000:
                    last_text = " ".join([last_json["content"], curr_text])
                    last_json = {"id": last_json["id"], "title": last_json["title"], "content": last_text}
                    last_json["contents"] = concat(last_json["title"], last_json["content"])
                    saved_text[-1] = json.dumps(last_json)
                else:
                    last_text = curr_text
                    last_json = {"id": "_".join([fname, str(j)]), "title": prefix, "content": curr_text}
                    last_json["contents"] = concat(last_json["title"], last_json["content"])
                    saved_text.append(json.dumps(last_json))
                    j += 1
            elif ch.tag == "list":
                list_text = [extract_text(c) for c in ch]
                if last_text is not None and len(" ".join(list_text) + last_text) < 1000:
                    last_text = " ".join([last_json["content"]] + list_text)
                    last_json = {"id": last_json["id"], "title": last_json["title"], "content": last_text}
                    last_json["contents"] = concat(last_json["title"], last_json["content"])
                    saved_text[-1] = json.dumps(last_json)
                elif len(" ".join(list_text)) < 1000:
                    last_text = " ".join(list_text)
                    last_json = {"id": "_".join([fname, str(j)]), "title": prefix, "content": last_text}
                    last_json["contents"] = concat(last_json["title"], last_json["content"])
                    saved_text.append(json.dumps(last_json))
                    j += 1
                else:
                    last_text = None
                    last_json = None
                    for c in list_text:
                        saved_text.append(json.dumps({"id": "_".join([fname, str(j)]), "title": prefix, "content": c, "contents": concat(prefix, c)}))
                        j += 1
                if last_node is not None and is_subtitle(last_node):
                    sub_title = ""
                    prefix = " -- ".join([title, sec_title])
            last_node = ch
    return saved_text


# --- download / extract / chunk ----------------------------------------------


def download():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    expected = _remote_size()
    if ARCHIVE_PATH.exists() and expected and ARCHIVE_PATH.stat().st_size == expected:
        print(f"Archive already downloaded ({ARCHIVE_PATH.stat().st_size:,} bytes)")
        return
    print(f"Downloading StatPearls archive (~1.86 GB) from {ARCHIVE_URL}")
    tmp = ARCHIVE_PATH.with_suffix(".part")
    with urllib.request.urlopen(ARCHIVE_URL) as r, open(tmp, "wb") as f:
        done = 0
        t0 = time.perf_counter()
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if done % (64 << 20) < (1 << 20):
                mb = done / 1e6
                print(f"  {mb:,.0f} MB ({mb / (time.perf_counter() - t0):,.0f} MB/s)", flush=True)
    tmp.replace(ARCHIVE_PATH)
    print(f"Downloaded {ARCHIVE_PATH.stat().st_size:,} bytes")


def _remote_size():
    try:
        req = urllib.request.Request(ARCHIVE_URL, method="HEAD")
        with urllib.request.urlopen(req) as r:
            return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def extract_archive():
    if NXML_DIR.is_dir() and len(list(NXML_DIR.glob("*.nxml"))) > 9000:
        print(f"Archive already extracted ({len(list(NXML_DIR.glob('*.nxml'))):,} nxml files)")
        return
    print("Extracting archive...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(CORPUS_DIR)
    print(f"Extracted {len(list(NXML_DIR.glob('*.nxml'))):,} nxml files")


def chunk_all():
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    nxml_files = sorted(NXML_DIR.glob("*.nxml"))
    print(f"Chunking {len(nxml_files):,} articles (resumable)...")
    snippets = 0
    t0 = time.perf_counter()
    for i, fp in enumerate(nxml_files, 1):
        out = CHUNK_DIR / (fp.stem + ".jsonl")
        if out.exists():  # resumable: skip already-chunked
            continue
        try:
            saved = extract(fp)
        except Exception as exc:
            print(f"  WARN skipping {fp.name}: {exc}")
            continue
        if saved:
            out.write_text("\n".join(saved), encoding="utf-8")
            snippets += len(saved)
        if i % 1000 == 0:
            print(f"  {i:,}/{len(nxml_files):,} articles ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    total = sum(1 for fp in CHUNK_DIR.glob("*.jsonl") for _ in fp.open(encoding="utf-8"))
    print(f"Chunk dir holds {total:,} snippets across {len(list(CHUNK_DIR.glob('*.jsonl'))):,} files")
    return total


# --- embed + add into chroma_med/ --------------------------------------------


def _existing_statpearls_ids(vectorstore):
    got = vectorstore._collection.get(where={"source": "statpearls"}, include=[])
    return set(got["ids"])


def _ordered_files(clinical_first):
    files = sorted(CHUNK_DIR.glob("*.jsonl"))
    if not clinical_first:
        return files
    clinical, other = [], []
    for fp in files:
        blob = fp.read_text(encoding="utf-8").lower()
        (clinical if any(k in blob for k in CLINICAL_KEYWORDS) else other).append(fp)
    print(f"clinical-first: {len(clinical):,} clinically-relevant files queued ahead of {len(other):,}")
    return clinical + other


def add_to_index(rebuild=False, clinical_first=False, batch_size=128):
    vectorstore = get_vectorstore()
    if rebuild:
        existing = _existing_statpearls_ids(vectorstore)
        if existing:
            print(f"--rebuild-statpearls: deleting {len(existing):,} existing statpearls vectors")
            vectorstore._collection.delete(where={"source": "statpearls"})
    skip = _existing_statpearls_ids(vectorstore)
    total = sum(1 for fp in CHUNK_DIR.glob("*.jsonl") for _ in fp.open(encoding="utf-8"))
    print(f"StatPearls snippets to add: {total - len(skip):,} (of {total:,}; {len(skip):,} already present)")

    files = _ordered_files(clinical_first)
    added = batched_add(
        vectorstore,
        iter_statpearls(CHUNK_DIR, files=files),
        batch_size=batch_size,
        total=total - len(skip),
        skip_ids=skip,
        label="statpearls",
    )
    final = vectorstore._collection.count()
    print(f"Added {added:,} statpearls snippets. chroma_med/ total now {final:,}")
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-only", action="store_true", help="download + chunk, no embedding")
    p.add_argument("--add-only", action="store_true", help="embed/add from existing chunks")
    p.add_argument("--rebuild-statpearls", action="store_true", help="drop statpearls vectors first")
    p.add_argument("--clinical-first", action="store_true", help="embed clinically-relevant articles first")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    if not args.add_only:
        download()
        extract_archive()
        chunk_all()
    if not args.chunk_only:
        add_to_index(
            rebuild=args.rebuild_statpearls,
            clinical_first=args.clinical_first,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
