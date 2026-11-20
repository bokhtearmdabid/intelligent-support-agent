import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pypdf import PdfReader

from .embeddings import embed_documents
from .models import KnowledgeChunk, KnowledgeDocument

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "1"     # chunking-এর যুক্তি বদলালে বাড়াও: সব নতুন করে প্রক্রিয়া হবে
MIN_CHUNK_CHARS = 30
MARKDOWN_SUFFIXES = {".md", ".markdown"}
SUPPORTED_SUFFIXES = MARKDOWN_SUFFIXES | {".txt", ".pdf"}
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


@dataclass(frozen=True)
class ChunkDraft:
    heading_path: str
    text: str

    @property
    def embedding_input(self):
        """Embed করার লেখা: heading সহ, যাতে chunk নিজেই প্রসঙ্গ বহন করে।"""
        if self.heading_path:
            return f"{self.heading_path}\n{self.text}"
        return self.text


# ---------- নিরাপত্তা ----------
def resolve_source(source_name):
    """source_name অবশ্যই KNOWLEDGE_DIR-এর ভেতরের পথ হতে হবে (path traversal প্রতিরক্ষা)।"""
    base = Path(settings.KNOWLEDGE_DIR).resolve()
    path = (base / source_name).resolve()
    if base not in path.parents:
        raise ValueError("Path is outside the knowledge directory")
    return path


# ---------- ১. Load ----------
def read_text(path):
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8")


# ---------- ২. Split ----------
def split_text(text, *, markdown):
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    drafts = []
    if markdown:
        sections = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS, strip_headers=True
        ).split_text(text)
        for section in sections:
            path = " > ".join(
                section.metadata[key]
                for key in ("h1", "h2", "h3")
                if key in section.metadata
            )
            for piece in recursive.split_text(section.page_content):
                drafts.append(ChunkDraft(path, piece.strip()))
    else:
        for piece in recursive.split_text(text):
            drafts.append(ChunkDraft("", piece.strip()))
    return [d for d in drafts if len(d.text) >= MIN_CHUNK_CHARS]


# ---------- ৩. Embed + ৪. Store ----------
def _fingerprint():
    return "|".join(str(x) for x in (
        PIPELINE_VERSION, settings.EMBEDDING_MODEL, settings.EMBEDDING_DIMENSIONS,
        settings.RAG_CHUNK_SIZE, settings.RAG_CHUNK_OVERLAP,
    ))


def _title_from(text, path):
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()[:200]
    return path.stem.replace("_", " ").replace("-", " ").title()


def ingest_source(source_name, *, force=False):
    """একটা ফাইল প্রক্রিয়া করে। ফেরত: (document, outcome)।

    outcome: created | updated | unchanged | reactivated | failed
    """
    path = resolve_source(source_name)
    digest = hashlib.sha256(
        path.read_bytes() + _fingerprint().encode()
    ).hexdigest()

    doc, created = KnowledgeDocument.objects.get_or_create(
        source_name=source_name, defaults={"title": path.stem}
    )

    if not force and doc.status == KnowledgeDocument.Status.READY and doc.content_hash == digest:
        if doc.is_active:
            return doc, "unchanged"
        doc.is_active = True
        doc.save(update_fields=["is_active"])
        return doc, "reactivated"

    doc.status = KnowledgeDocument.Status.PROCESSING
    doc.error_message = ""
    doc.save(update_fields=["status", "error_message"])

    try:
        text = read_text(path)
        drafts = split_text(text, markdown=path.suffix.lower() in MARKDOWN_SUFFIXES)
        if not drafts:
            raise ValueError("No readable text found (scanned PDF or empty file?)")

        vectors = embed_documents([d.embedding_input for d in drafts])   # নেটওয়ার্ক: DB-র বাইরে

        with transaction.atomic():                                       # সব-অথবা-কিছুই-না
            doc.chunks.all().delete()
            KnowledgeChunk.objects.bulk_create(
                [
                    KnowledgeChunk(
                        document=doc,
                        chunk_index=i,
                        heading_path=draft.heading_path,
                        content=draft.text,
                        embedding=vector,
                        embedding_model=settings.EMBEDDING_MODEL,
                    )
                    for i, (draft, vector) in enumerate(zip(drafts, vectors))
                ],
                batch_size=100,
            )
            doc.title = _title_from(text, path)
            doc.status = KnowledgeDocument.Status.READY
            doc.content_hash = digest
            doc.chunk_count = len(drafts)
            doc.error_message = ""
            doc.is_active = True
            doc.processed_at = timezone.now()
            doc.save()
    except Exception as exc:
        logger.exception("Ingestion failed for %s", source_name)
        doc.status = KnowledgeDocument.Status.FAILED
        doc.error_message = f"{type(exc).__name__}: {exc}"[:300]
        doc.save(update_fields=["status", "error_message"])
        return doc, "failed"

    return doc, "created" if created else "updated"


def deactivate_missing(present_source_names):
    """ফোল্ডার থেকে মুছে ফেলা ডকুমেন্টের জ্ঞান আর পরিবেশন না করা।"""
    return (
        KnowledgeDocument.objects.filter(is_active=True)
        .exclude(source_name__in=present_source_names)
        .update(is_active=False)
    )