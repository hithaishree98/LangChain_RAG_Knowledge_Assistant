import os
import re
import logging
import functools
from typing import List

_log = logging.getLogger(__name__)

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

# ── EXPERIMENT KNOBS ─────────────────────────────────────────────────────────
CHUNKING_MODE = os.getenv("CHUNKING_MODE", "full").lower()
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "full").lower()
if CHUNKING_MODE != "full" or RETRIEVAL_MODE != "full":
    _log.info("experiment_mode_enabled chunking=%s retrieval=%s",
              CHUNKING_MODE, RETRIEVAL_MODE)

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

_embedder = None


def get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        _embedder = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={"trust_remote_code": True, "device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedder


@functools.lru_cache(maxsize=512)
def embed_cached(text: str) -> tuple:
    return tuple(get_embedder().embed_query(text))


class _LazyEmbedder(Embeddings):
    def embed_documents(self, texts): return get_embedder().embed_documents(texts)
    def embed_query(self, text):      return get_embedder().embed_query(text)


vectorstore = Chroma(collection_name="child_chunks", persist_directory=CHROMA_DIR,
                     embedding_function=_LazyEmbedder())
parent_store = Chroma(collection_name="parent_chunks", persist_directory=CHROMA_DIR,
                      embedding_function=_LazyEmbedder())


# ── Cross-encoder reranker ────────────────────────────────────────────────────
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers.cross_encoder import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank_docs(query: str, docs: List[Document], top_k: int = 4) -> List[Document]:
    if len(docs) <= top_k:
        return docs
    ce = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = ce.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def warmup_models() -> None:
    """Force-load the embedding model and cross-encoder at API startup.

    Without this, the 440MB nomic-embed model cold-loads on the first upload
    request. On CPU that can take 60-300+ seconds, blocking the request past
    any reasonable client timeout. Calling this from the FastAPI lifespan
    ensures /health only reports ready once the models are actually usable.
    """
    # Trigger embedder load and run one embed call so lazy transformer
    # weights are fully initialized (not just the constructor).
    get_embedder().embed_query("warmup")
    _get_cross_encoder().predict([("warmup", "warmup")])


# ── BM25 ──────────────────────────────────────────────────────────────────────
def _tokenize(text): return re.findall(r"[a-z0-9]+", text.lower())


def bm25_search(query, user_id, k=10):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        # Soft-degrade if rank_bm25 isn't installed — dense retrieval still works.
        # Logged once so the absence of BM25 is visible in deployment health checks.
        _log.warning("bm25_unavailable: rank_bm25 not installed; dense-only retrieval")
        return []
    try:
        result = vectorstore._collection.get(
            where={"user_id": {"$eq": user_id}},
            include=["documents", "metadatas"])
    except Exception as e:
        # A Chroma failure here means the vector store is unhealthy — caller
        # should NOT treat the empty list as "no BM25 hits." Logging the
        # exception so a health check / log scraper can detect the failure.
        # Returning [] keeps RRF working if dense still has results, but the
        # llm_breaker / health endpoint will surface the underlying issue.
        _log.error("bm25_search_chroma_failed", extra={"user_id": user_id, "error": str(e)})
        return []
    texts = result.get("documents") or []
    metas = result.get("metadatas") or []
    if not texts:
        return []
    idx = BM25Okapi([_tokenize(t) for t in texts])
    scores = idx.get_scores(_tokenize(query))
    top_n = min(k, len(texts))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [Document(page_content=texts[i], metadata=metas[i] or {}) for i in top_indices]


def rrf_merge(dense, sparse, k=60):
    scores = {}; doc_map = {}
    for rank, doc in enumerate(dense):
        key = doc.page_content[:120]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc
    for rank, doc in enumerate(sparse):
        key = doc.page_content[:120]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]


def fetch_parents(child_chunks, user_id=None):
    parent_ids = []
    for doc in child_chunks:
        pid = doc.metadata.get("parent_chunk_id")
        if pid and pid not in parent_ids:
            parent_ids.append(pid)
    if not parent_ids:
        # Children have no parent_chunk_id metadata — this is the flat-mode
        # case (baseline / sentence). Caller treats children as the LLM context.
        return child_chunks
    if not user_id:
        # Hard guard against cross-tenant data exposure: without a user_id we
        # CANNOT safely query parent_store. Falling back to children is safe
        # (they're already user-scoped from the dense search). Surfaced so
        # callers don't mistake degraded context for normal operation.
        _log.warning("fetch_parents_skipped: no user_id provided; returning child chunks")
        return child_chunks
    try:
        result = parent_store._collection.get(
            ids=parent_ids,
            where={"user_id": {"$eq": user_id}},
            include=["documents", "metadatas"],
        )
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        if docs:
            return [Document(page_content=docs[i], metadata=metas[i] or {}) for i in range(len(docs))]
        # Parent IDs didn't resolve — workspace may have been wiped between
        # child upload and query. Falling back to children is the only
        # recoverable option, but logged so operators can investigate.
        _log.warning("fetch_parents_empty_result", extra={
            "user_id": user_id, "parent_id_count": len(parent_ids)})
    except Exception as e:
        # parent_store unhealthy — log loudly so health checks / log scrapers
        # can detect the degraded state. The brief still completes (with
        # smaller child-chunk context) but quality will be lower until fixed.
        _log.error("fetch_parents_failed", extra={
            "user_id": user_id, "error": str(e)})
    return child_chunks


# ── Hybrid retriever with RETRIEVAL_MODE switch ───────────────────────────────
class HybridRetriever(BaseRetriever):
    user_id: str = "default"
    k_dense: int = 10
    k_bm25: int = 10
    k_rerank: int = 4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        _dense_retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.k_dense, "filter": {"user_id": self.user_id}}
        )
        try:
            dense = _dense_retriever.invoke(query)
        except AttributeError:
            dense = _dense_retriever.get_relevant_documents(query)

        if RETRIEVAL_MODE == "dense":
            # Dense only, top-k_rerank results
            return dense[:self.k_rerank]

        sparse = bm25_search(query, self.user_id, k=self.k_bm25)
        merged = rrf_merge(dense, sparse)

        if RETRIEVAL_MODE == "dense_bm25":
            # Dense + BM25 via RRF, no reranker, no parent fetch
            return merged[:self.k_rerank]

        # Full pipeline
        child_top = rerank_docs(query, merged, top_k=self.k_rerank)
        return fetch_parents(child_top, self.user_id)

    async def _aget_relevant_documents(self, query):
        return self._get_relevant_documents(query)


def get_retriever_for_user(user_id):
    return HybridRetriever(user_id=user_id)


# ── PDF / DOCX / HTML loaders (with sentence-aware path) ─────────────────────
def noise_filter(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped: continue
        if re.fullmatch(r"[-–—]\s*\d+\s*[-–—]?|\d+\s*[-–—]", stripped): continue  # require at least one dash so bare years ("2024") are preserved
        if re.fullmatch(r"[Pp]age\s+\d+(\s+of\s+\d+)?", stripped): continue
        if len(stripped.split()) < 4 and len(stripped) < 40:
            # Preserve section headers ("Risks:", "Summary:") and bullet points
            is_header = stripped.endswith(":")
            is_bullet = stripped[:1] in ("•", "-", "*", "–", "○", "▸", "·")
            if not is_header and not is_bullet:
                continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


def sentence_chunk(text, size=256, overlap=64):
    sentences = _split_sentences(text)
    if not sentences: return []
    chunks = []
    i = 0
    while i < len(sentences):
        group = []; word_count = 0; j = i
        while j < len(sentences) and word_count < size:
            group.append(sentences[j])
            word_count += len(sentences[j].split())
            j += 1
        chunk_text = " ".join(group).strip()
        if chunk_text:
            chunks.append(Document(page_content=chunk_text))
        overlap_words = 0; step = j
        while step > i + 1 and overlap_words < overlap:
            step -= 1
            overlap_words += len(sentences[step].split())
        i = max(i + 1, step)
    return chunks


def _split_sentences(text):
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
        try:
            nlp.enable_pipe("senter")
        except Exception:
            pass  # senter already active or unavailable; use whatever segmenter loaded
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]


def _load_pdf_full(file_path):
    """Full pipeline: pymupdf + noise + sentence chunk. Fallbacks kept."""
    try:
        import fitz
        pages = []
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text = page.get_text("text")
                if text:
                    pages.append(noise_filter(text))
        full_text = "\n\n".join(pages)
        chunks = sentence_chunk(full_text, size=256, overlap=64)
        for c in chunks:
            c.metadata["source"] = file_path
        return chunks
    except ImportError:
        pass
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    return splitter.split_documents(PyPDFLoader(file_path).load())


def _load_pdf_baseline(file_path):
    """Baseline: PyPDFLoader + RecursiveCharacterTextSplitter(800, 200). No noise filter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    return splitter.split_documents(PyPDFLoader(file_path).load())


def _load_docx_full(file_path):
    try:
        import docx
        d = docx.Document(file_path)
        chunks_raw = []
        current_heading = ""; current_paras = []
        def flush():
            if current_paras:
                text = (current_heading + "\n" + "\n".join(current_paras)).strip()
                chunks_raw.append(Document(page_content=text,
                    metadata={"source": file_path, "heading": current_heading}))
        for para in d.paragraphs:
            if para.style.name.startswith("Heading"):
                flush()
                current_heading = para.text.strip()
                current_paras = []
            elif para.text.strip():
                current_paras.append(para.text.strip())
        flush()
        if chunks_raw:
            return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(chunks_raw)
    except Exception:
        pass
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(
        Docx2txtLoader(file_path).load())


def _load_docx_baseline(file_path):
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(
        Docx2txtLoader(file_path).load())


def _load_html_full(file_path):
    try:
        from langchain_text_splitters import HTMLHeaderTextSplitter
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=[
            ("h1", "Header1"), ("h2", "Header2"), ("h3", "Header3")])
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()
        docs = splitter.split_text(html)
        for d in docs:
            d.metadata.setdefault("source", file_path)
        return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(docs)
    except Exception:
        return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(
            UnstructuredHTMLLoader(file_path).load())


def _load_html_baseline(file_path):
    # Read file directly to avoid UnstructuredHTMLLoader making external HTTP requests
    from bs4 import BeautifulSoup
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()
    text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    docs = [Document(page_content=text, metadata={"source": file_path})]
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(docs)


def _load_transcript(file_path):
    """Transcript and ticket chunkers are too specialized to have a 'baseline' — keep as-is."""
    from ingestion.transcript_parser import parse as parse_t
    from ingestion.transcript_chunker import chunk as chunk_t
    return chunk_t(parse_t(file_path), source=file_path)


def _load_ticket(file_path):
    from ingestion.ticket_parser import parse as parse_t
    from ingestion.ticket_chunker import chunk as chunk_t
    return chunk_t(parse_t(file_path), source=file_path)


def load_and_split_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if CHUNKING_MODE == "baseline":
        if ext == ".pdf":   return _load_pdf_baseline(file_path)
        if ext == ".docx":  return _load_docx_baseline(file_path)
        if ext == ".html":  return _load_html_baseline(file_path)
        if ext == ".txt":   return _load_transcript(file_path)
        if ext == ".json":  return _load_ticket(file_path)
    # "sentence" and "full" use the same advanced loaders (noise-filtered,
    # structure-aware, sentence-chunked for PDF). The only difference between
    # them is in index_document_to_chroma below: "full" builds a parent-child
    # index (children searched, parents returned); "sentence" indexes the
    # loader output as flat chunks with no parent store.
    if ext == ".pdf":   return _load_pdf_full(file_path)
    if ext == ".docx":  return _load_docx_full(file_path)
    if ext == ".html":  return _load_html_full(file_path)
    if ext == ".txt":   return _load_transcript(file_path)
    if ext == ".json":  return _load_ticket(file_path)
    raise ValueError(f"Unsupported: {file_path}")


# ── Indexing: parent-child for "full", flat for "baseline" and "sentence" ────
_PARENT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
_CHILD_SPLITTER  = RecursiveCharacterTextSplitter(chunk_size=500,  chunk_overlap=50)


_CONTEXTUAL_RETRIEVAL = os.getenv("CONTEXTUAL_RETRIEVAL", "").lower() in ("1", "true", "yes")

_CONTEXT_PROMPT = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Give a short (1-2 sentence) context to situate this chunk within the overall
document, for the purpose of improving search retrieval. Answer only with the
context itself — no preamble, no prefix, just the description."""


def _contextualize_chunks(chunks, filename):
    """Prepend an LLM-generated context sentence to each chunk.

    Implements Anthropic's contextual retrieval (Sep 2024): each chunk is
    situated within the whole document before embedding, so that cross-chunk
    references ("the fix" / "this customer" / "as mentioned above") and
    section context are recovered. Reported +35-49% retrieval accuracy on
    Anthropic's benchmarks. Enabled via CONTEXTUAL_RETRIEVAL=1.

    Falls back to the original chunk silently on any per-chunk LLM error so
    ingestion is never blocked by a transient model failure.
    """
    if not chunks:
        return chunks
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.documents import Document
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )
    # Reconstruct full document from chunks. Capped so we stay well below
    # Gemini's input-token budget even for very long documents.
    full_doc = "\n\n".join(c.page_content for c in chunks)[:30000]
    enriched = []
    for chunk in chunks:
        try:
            prompt = _CONTEXT_PROMPT.format(document=full_doc, chunk=chunk.page_content)
            context = llm.invoke(prompt).content.strip()
            if not context:
                # Empty LLM response — treat as a failed contextualization so
                # downstream strippers don't get confused by a missing prefix.
                enriched.append(chunk)
                continue
            enriched.append(Document(
                page_content=f"[Context: {context}]\n\n{chunk.page_content}",
                metadata={
                    **chunk.metadata,
                    "has_context_prefix": True,   # flag for _strip_context_prefix
                    "context_text": context,      # kept for debugging/transparency
                },
            ))
        except Exception as e:
            print(f"  [contextualize] {filename} chunk skipped: {e}")
            enriched.append(chunk)
    return enriched


def _existing_workspace_contextual_flag(user_id: str):
    """Read back the contextual-retrieval flag from an existing chunk in this
    workspace, or return None if the workspace has no chunks yet.

    Used to detect mixed-embedding scenarios: if a user ingests some docs with
    CONTEXTUAL_RETRIEVAL=0 and later uploads more with =1, the vector space is
    silently inconsistent (some chunks have an LLM-generated prefix baked into
    their embedding; others don't). Retrieval quality degrades without warning.
    """
    try:
        existing = vectorstore._collection.get(
            where={"user_id": user_id}, limit=1, include=["metadatas"]
        )
        metas = (existing or {}).get("metadatas") or []
        if not metas:
            return None
        return bool(metas[0].get("ingest_contextual_retrieval", False))
    except Exception:
        return None


def _warn_if_mixed_contextual(user_id: str) -> None:
    existing_flag = _existing_workspace_contextual_flag(user_id)
    current_flag = bool(_CONTEXTUAL_RETRIEVAL)
    if existing_flag is not None and existing_flag != current_flag:
        print(
            f"  [WARN] workspace '{user_id}' has existing chunks with "
            f"ingest_contextual_retrieval={existing_flag}, but the current "
            f"upload is using CONTEXTUAL_RETRIEVAL={current_flag}. "
            f"Retrieval quality will be inconsistent. Wipe the workspace "
            f"and re-ingest all docs under a single flag to recover."
        )


def index_document_to_chroma(file_path, file_id, user_id="default", filename=None):
    fname = filename or os.path.basename(file_path)
    _warn_if_mixed_contextual(user_id)
    try:
        raw_docs = load_and_split_document(file_path)
    except Exception as e:
        print(f"Error loading: {e}"); return False

    try:
        if CHUNKING_MODE == "full":
            # Parent-child pipeline
            parent_docs = _PARENT_SPLITTER.split_documents(raw_docs)
            for i, p in enumerate(parent_docs):
                p.metadata["parent_chunk_id"] = f"p_{file_id}_{i}"
                p.metadata["file_id"] = file_id
                p.metadata["user_id"] = user_id
                p.metadata["filename"] = fname
            parent_ids = [p.metadata["parent_chunk_id"] for p in parent_docs]
            parent_store.add_documents(parent_docs, ids=parent_ids)

            child_docs = []
            for p in parent_docs:
                for c in _CHILD_SPLITTER.split_documents([p]):
                    c.metadata["file_id"] = file_id
                    c.metadata["user_id"] = user_id
                    c.metadata["parent_chunk_id"] = p.metadata["parent_chunk_id"]
                    c.metadata["filename"] = fname
                    child_docs.append(c)
            # Contextual retrieval (Anthropic, Sep 2024): prepend an LLM-
            # generated doc-level context to each child before embedding.
            if _CONTEXTUAL_RETRIEVAL:
                print(f"  [contextualize] {fname}: {len(child_docs)} children")
                child_docs = _contextualize_chunks(child_docs, fname)
            for c in child_docs:
                c.metadata["ingest_contextual_retrieval"] = bool(_CONTEXTUAL_RETRIEVAL)
            vectorstore.add_documents(child_docs)
        else:
            # Flat: raw_docs go straight into the child store. No parent lookup.
            for d in raw_docs:
                d.metadata["file_id"] = file_id
                d.metadata["user_id"] = user_id
                d.metadata["filename"] = fname
                # No parent_chunk_id → fetch_parents becomes a no-op passthrough
            if _CONTEXTUAL_RETRIEVAL:
                print(f"  [contextualize] {fname}: {len(raw_docs)} chunks")
                raw_docs = _contextualize_chunks(raw_docs, fname)
            for d in raw_docs:
                d.metadata["ingest_contextual_retrieval"] = bool(_CONTEXTUAL_RETRIEVAL)
            vectorstore.add_documents(raw_docs)
        return True
    except Exception as e:
        print(f"Error indexing: {e}"); return False


def delete_doc_from_chroma(file_id, user_id="default"):
    try:
        cond = {"$and": [{"file_id": {"$eq": file_id}}, {"user_id": {"$eq": user_id}}]}
        vectorstore._collection.delete(where=cond)
        parent_store._collection.delete(where=cond)
        return True
    except Exception as e:
        print(f"Error deleting: {e}"); return False
