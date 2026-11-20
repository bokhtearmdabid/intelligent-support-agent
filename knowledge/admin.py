from django.contrib import admin, messages

from .ingestion import ingest_source, resolve_source
from .models import KnowledgeChunk, KnowledgeDocument


class ChunkInline(admin.TabularInline):
    model = KnowledgeChunk
    fields = ("chunk_index", "heading_path", "content")   # embedding ইচ্ছাকৃতভাবে বাদ (৭৬৮টা সংখ্যা)
    readonly_fields = fields
    extra = 0
    can_delete = False

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(KnowledgeDocument)
class KnowledgeDocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "source_name", "doc_type", "status",
                    "chunk_count", "is_active", "processed_at")
    list_filter = ("status", "doc_type", "is_active")
    readonly_fields = ("source_name", "status", "content_hash", "chunk_count",
                       "error_message", "processed_at", "created_at")
    inlines = [ChunkInline]
    actions = ["reingest_selected"]

    def has_add_permission(self, request):
        return False      # ডকুমেন্ট আসে ফোল্ডার থেকে, হাতে নয়

    @admin.action(description="Re-ingest selected documents")
    def reingest_selected(self, request, queryset):
        for doc in queryset:
            try:
                _, outcome = ingest_source(doc.source_name, force=True)
                level = messages.ERROR if outcome == "failed" else messages.SUCCESS
                self.message_user(request, f"{doc.source_name}: {outcome}", level=level)
            except (ValueError, FileNotFoundError) as exc:
                self.message_user(request, f"{doc.source_name}: {exc}", level=messages.ERROR)