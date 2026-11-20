from django.conf import settings
from rest_framework import serializers

from .models import ChatMessage, SupportConversation


class ChatRequestSerializer(serializers.Serializer):
    conversation_id = serializers.UUIDField(required=False)
    message = serializers.CharField(
        max_length=settings.CHAT_MESSAGE_MAX_CHARS, trim_whitespace=True
    )


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ("role", "content", "created_at")
        read_only_fields = fields


class ConversationListSerializer(serializers.ModelSerializer):
    conversation_id = serializers.UUIDField(source="public_id", read_only=True)

    class Meta:
        model = SupportConversation
        fields = ("conversation_id", "title", "status", "updated_at")
        read_only_fields = fields


class ConversationDetailSerializer(ConversationListSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta(ConversationListSerializer.Meta):
        fields = ConversationListSerializer.Meta.fields + ("messages",)
        read_only_fields = fields