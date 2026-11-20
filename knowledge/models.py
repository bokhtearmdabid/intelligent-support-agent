from django.conf import settings
from django.db import models
from pgvector.django import HnswIndex, VectorField


class KnowledgeDocument(models.Model):
    class DocType(models.TextChoices):
        POLICY = "policy", "Policy"
        FAQ = "faq", "FAQ"
        MANUAL = "manual", "Product manual"
        GUIDELINE = "guideline", "Guideline"

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    source_name = models.CharField(max_length=255, unique=True)   # knowledge_base/ থেকে আপেক্ষিক পথ
    title = models.CharField(max_length=200)
    doc_type = models.CharField(
        max_length=20, choices=DocType.choices, default=DocType.POLICY
    )
    status = models.CharField(
        max_length=12, choices=Status.choices, default=Status.PENDING
    )
    content_hash = models.CharField(max_length=64, blank=True)
    chunk_count = models.PositiveIntegerField(default=0)
    error_message = models.CharField(max_length=300, blank=True)
    is_active = models.BooleanField(default=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["title"]

    def __str__(self):
        return self.title


class KnowledgeChunk(models.Model):
    document = models.ForeignKey(
        KnowledgeDocument, on_delete=models.CASCADE, related_name="chunks"
    )
    chunk_index = models.PositiveIntegerField()
    heading_path = models.CharField(max_length=300, blank=True)
    content = models.TextField()
    embedding = VectorField(dimensions=settings.EMBEDDING_DIMENSIONS)
    embedding_model = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["document_id", "chunk_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["document", "chunk_index"], name="unique_chunk_per_document"
            ),
        ]
        indexes = [
            HnswIndex(
                name="chunk_embedding_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]

    def __str__(self):
        return f"{self.document_id}#{self.chunk_index}"