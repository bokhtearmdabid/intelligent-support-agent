from django.conf import settings
from rest_framework.exceptions import NotFound
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework.viewsets import ReadOnlyModelViewSet

from .llm import LLMUnavailable, generate_reply
from .models import SupportConversation
from .serializers import (
    ChatRequestSerializer,
    ConversationDetailSerializer,
    ConversationListSerializer,
)
from .services import (
    conversations_for_customer,
    get_conversation_for_customer,
    recent_history,
    save_exchange,
)


class ChatView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "chat"

    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        message = serializer.validated_data["message"]
        conversation_id = serializer.validated_data.get("conversation_id")

        conversation = None
        history = []
        if conversation_id:
            try:
                conversation = get_conversation_for_customer(
                    request.user, conversation_id
                )
            except SupportConversation.DoesNotExist:
                raise NotFound()   # অন্যেরটা আর অস্তিত্বহীন: একই উত্তর
            history = recent_history(conversation, settings.CHAT_HISTORY_LIMIT)

        # ধীর কাজ: কোনো DB transaction খোলা নেই
        try:
            result = generate_reply(history, message)
        except LLMUnavailable:
            return Response(
                {"detail": "The assistant is temporarily unavailable. "
                           "Please try again in a moment."},
                status=503,
            )

        conversation = save_exchange(request.user, conversation, message, result)
        return Response(
            {"conversation_id": conversation.public_id, "reply": result.text},
            status=201,
        )


class ConversationViewSet(ReadOnlyModelViewSet):
    permission_classes = [IsAuthenticated]
    lookup_field = "public_id"

    def get_queryset(self):
        qs = conversations_for_customer(self.request.user)
        if self.action == "retrieve":
            qs = qs.prefetch_related("messages")
        return qs

    def get_serializer_class(self):
        if self.action == "retrieve":
            return ConversationDetailSerializer
        return ConversationListSerializer