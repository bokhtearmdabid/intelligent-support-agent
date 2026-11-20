from django.contrib import admin

from .models import ChatMessage, SupportConversation


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    can_delete = False
    readonly_fields = ("role", "content", "model_name", "prompt_version",
                       "input_tokens", "output_tokens", "created_at")

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(SupportConversation)
class SupportConversationAdmin(admin.ModelAdmin):
    list_display = ("public_id", "customer", "title", "status", "updated_at")
    readonly_fields = ("public_id", "created_at", "updated_at")
    inlines = [ChatMessageInline]