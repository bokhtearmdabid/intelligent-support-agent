import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from django.conf import settings
from django.test import SimpleTestCase, TestCase, override_settings
from pgvector.django import CosineDistance

from knowledge import embeddings
from knowledge.embeddings import EmbeddingError
from knowledge.ingestion import (
    deactivate_missing,
    ingest_source,
    resolve_source,
    split_text,
)
from knowledge.models import KnowledgeChunk, KnowledgeDocument

DIM = settings.EMBEDDING_DIMENSIONS
EMBED = "knowledge.ingestion.embed_documents"

SAMPLE = (
    "# Returns\n\n"
    "## Damaged items\n"
    "If an item arrives damaged you can report it within 7 days of delivery.\n\n"
    "## Refund time\n"
    "Refunds to bKash take 3 to 5 working days after approval.\n"
)


def fake_embed(texts):
    return [[1.0] + [0.0] * (DIM - 1) for _ in texts]


class SplitTests(SimpleTestCase):
    def test_heading_path_is_kept(self):
        drafts = split_text(SAMPLE, markdown=True)
        self.assertEqual(
            [d.heading_path for d in drafts],
            ["Returns > Damaged items", "Returns > Refund time"],
        )
        self.assertTrue(drafts[0].embedding_input.startswith("Returns > Damaged items\n"))

    def test_long_section_is_split_within_limit(self):
        long_text = "# Big\n\n" + " ".join(
            f"Sentence number {i} explains something about delivery." for i in range(120)
        )
        drafts = split_text(long_text, markdown=True)
        self.assertGreater(len(drafts), 1)
        self.assertLessEqual(max(len(d.text) for d in drafts), settings.RAG_CHUNK_SIZE)

    def test_tiny_fragments_are_dropped(self):
        self.assertEqual(split_text("# Title\n\nHi\n", markdown=True), [])

    def test_plain_text_has_no_heading_path(self):
        drafts = split_text("A plain paragraph that is long enough to keep.", markdown=False)
        self.assertEqual(drafts[0].heading_path, "")


class IngestionTests(TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        override = override_settings(KNOWLEDGE_DIR=self.dir)
        override.enable()
        self.addCleanup(override.disable)

    def write(self, name, text):
        (self.dir / name).write_text(text, encoding="utf-8")

    def test_ingest_creates_document_and_chunks(self):
        self.write("returns.md", SAMPLE)
        with patch(EMBED, side_effect=fake_embed):
            doc, outcome = ingest_source("returns.md")
        self.assertEqual(outcome, "created")
        self.assertEqual(doc.status, KnowledgeDocument.Status.READY)
        self.assertEqual(doc.title, "Returns")
        self.assertEqual(doc.chunks.count(), 2)
        self.assertEqual(doc.chunk_count, 2)
        chunk = doc.chunks.first()
        self.assertEqual(chunk.embedding_model, settings.EMBEDDING_MODEL)
        self.assertEqual(len(chunk.embedding), DIM)

    def test_second_run_makes_no_api_call(self):
        self.write("returns.md", SAMPLE)
        with patch(EMBED, side_effect=fake_embed):
            ingest_source("returns.md")
        with patch(EMBED) as embed:
            _, outcome = ingest_source("returns.md")
        self.assertEqual(outcome, "unchanged")
        embed.assert_not_called()

    def test_force_reprocesses(self):
        self.write("returns.md", SAMPLE)
        with patch(EMBED, side_effect=fake_embed):
            ingest_source("returns.md")
        with patch(EMBED, side_effect=fake_embed) as embed:
            _, outcome = ingest_source("returns.md", force=True)
        self.assertEqual(outcome, "updated")
        embed.assert_called_once()

    def test_changed_file_replaces_old_chunks(self):
        self.write("returns.md", SAMPLE)
        with patch(EMBED, side_effect=fake_embed):
            ingest_source("returns.md")
        self.write("returns.md", "# Returns\n\n## New rule\nReturns are now accepted within 10 days.\n")
        with patch(EMBED, side_effect=fake_embed):
            _, outcome = ingest_source("returns.md")
        self.assertEqual(outcome, "updated")
        contents = list(KnowledgeChunk.objects.values_list("content", flat=True))
        self.assertEqual(len(contents), 1)
        self.assertIn("10 days", contents[0])
        self.assertFalse(any("7 days" in c for c in contents))

    def test_embedding_failure_keeps_old_chunks_and_marks_failed(self):
        self.write("returns.md", SAMPLE)
        with patch(EMBED, side_effect=fake_embed):
            ingest_source("returns.md")
        self.write("returns.md", SAMPLE + "\n## Extra\nAnother section with enough text to keep.\n")
        with patch(EMBED, side_effect=EmbeddingError("boom")):
            doc, outcome = ingest_source("returns.md")
        self.assertEqual(outcome, "failed")
        self.assertEqual(doc.status, KnowledgeDocument.Status.FAILED)
        self.assertIn("boom", doc.error_message)
        self.assertEqual(doc.chunks.count(), 2)          # পুরনোগুলো অক্ষত: আংশিক অবস্থা নেই

    def test_empty_file_fails_without_chunks(self):
        self.write("empty.md", "   \n")
        with patch(EMBED, side_effect=fake_embed):
            doc, outcome = ingest_source("empty.md")
        self.assertEqual(outcome, "failed")
        self.assertEqual(doc.chunks.count(), 0)

    def test_path_traversal_is_blocked(self):
        for bad in ("../secret.txt", "../../.env", "/etc/passwd"):
            with self.assertRaises(ValueError, msg=bad):
                resolve_source(bad)

    def test_deactivate_missing(self):
        self.write("a.md", SAMPLE)
        self.write("b.md", SAMPLE)
        with patch(EMBED, side_effect=fake_embed):
            ingest_source("a.md")
            ingest_source("b.md")
        count = deactivate_missing(["a.md"])
        self.assertEqual(count, 1)
        self.assertTrue(KnowledgeDocument.objects.get(source_name="a.md").is_active)
        self.assertFalse(KnowledgeDocument.objects.get(source_name="b.md").is_active)

    def test_deactivated_document_is_reactivated_without_api_call(self):
        self.write("a.md", SAMPLE)
        with patch(EMBED, side_effect=fake_embed):
            ingest_source("a.md")
        deactivate_missing([])
        with patch(EMBED) as embed:
            _, outcome = ingest_source("a.md")
        self.assertEqual(outcome, "reactivated")
        embed.assert_not_called()


class VectorSearchTests(TestCase):
    """pgvector সত্যিই কাজ করছে কি না: Step 7-এর ভিত্তি।"""

    def test_cosine_distance_orders_by_similarity(self):
        doc = KnowledgeDocument.objects.create(source_name="x.md", title="X")
        near = [1.0] + [0.0] * (DIM - 1)
        far = [0.0, 1.0] + [0.0] * (DIM - 2)
        KnowledgeChunk.objects.create(document=doc, chunk_index=0, content="far",
                                      embedding=far, embedding_model="t")
        KnowledgeChunk.objects.create(document=doc, chunk_index=1, content="near",
                                      embedding=near, embedding_model="t")
        results = list(
            KnowledgeChunk.objects.annotate(d=CosineDistance("embedding", near)).order_by("d")
        )
        self.assertEqual([c.content for c in results], ["near", "far"])
        self.assertAlmostEqual(results[0].d, 0.0, places=5)
        self.assertAlmostEqual(results[1].d, 1.0, places=5)


class EmbeddingGuardTests(SimpleTestCase):
    def _client_returning(self, vectors):
        client = Mock()
        client.embed_documents.return_value = vectors
        return patch.object(embeddings, "_client", return_value=client)

    def test_wrong_dimension_is_rejected(self):
        with self._client_returning([[0.1] * (DIM - 1)]):
            with self.assertRaises(EmbeddingError):
                embeddings.embed_documents(["text"])

    def test_vectors_are_normalized(self):
        with self._client_returning([[3.0] + [4.0] + [0.0] * (DIM - 2)]):
            (vector,) = embeddings.embed_documents(["text"])
        self.assertAlmostEqual(sum(x * x for x in vector), 1.0, places=6)

    def test_api_failure_becomes_embedding_error(self):
        client = Mock()
        client.embed_documents.side_effect = TimeoutError("slow")
        with patch.object(embeddings, "_client", return_value=client):
            with self.assertRaises(EmbeddingError):
                embeddings.embed_documents(["text"])