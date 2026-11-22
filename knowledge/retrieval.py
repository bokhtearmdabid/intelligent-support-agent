from dataclasses import dataclass

from django.conf import settings
from pgvector.django import CosineDistance

from .embeddings import embed_query
from .models import KnowledgeChunk, KnowledgeDocument


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    source_name: str
    document_title: str
    heading_path: str
    content: str
    distance: float


def search_knowledge(query, *, k=None, max_distance=None):
    query = (query or "").strip()
    if not query:
        return []                                   # খালি প্রশ্নে API ডাকার মানে নেই

    k = settings.RAG_TOP_K if k is None else k
    max_distance = settings.RAG_MAX_DISTANCE if max_distance is None else max_distance

    vector = embed_query(query)
    rows = (
        KnowledgeChunk.objects
        .filter(
            document__status=KnowledgeDocument.Status.READY,
            document__is_active=True,
        )
        .select_related("document")
        .annotate(distance=CosineDistance("embedding", vector))
        .order_by("distance")[:k]
    )
    return [
        RetrievedChunk(
            chunk_id=c.id,
            source_name=c.document.source_name,
            document_title=c.document.title,
            heading_path=c.heading_path,
            content=c.content,
            distance=float(c.distance),
        )
        for c in rows
        if c.distance <= max_distance
    ]