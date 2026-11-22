from django.db import transaction

from .models import ChatMessage, SupportConversation


def conversations_for_customer(user):
    if not user.is_authenticated:
        return SupportConversation.objects.none()
    return SupportConversation.objects.filter(customer=user)


def get_conversation_for_customer(user, public_id):
    return conversations_for_customer(user).get(public_id=public_id)


def recent_history(conversation, limit):
    """শেষ `limit`টা মেসেজ, পুরনো থেকে নতুন ক্রমে: [(role, content), ...]"""
    latest = list(conversation.messages.order_by("-created_at", "-id")[:limit])
    history = [(m.role, m.content) for m in reversed(latest)]
    # উইন্ডো assistant মেসেজ দিয়ে শুরু হলে সেটা বাদ (অনেক provider user দিয়ে শুরু চায়)
    while history and history[0][0] != ChatMessage.Role.USER:
        history.pop(0)
    return history


@transaction.atomic
def save_exchange(user, conversation, user_text, result, *, sources=(), retrieval_status=""):
    if conversation is None:
        conversation = SupportConversation.objects.create(
            customer=user, title=user_text[:80]
        )
    ChatMessage.objects.create(
        conversation=conversation,
        role=ChatMessage.Role.USER,
        content=user_text,
    )
    ChatMessage.objects.create(
        conversation=conversation,
        role=ChatMessage.Role.ASSISTANT,
        content=result.text,
        model_name=result.model_name,
        prompt_version=result.prompt_version,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        sources=list(sources),
        retrieval_status=retrieval_status,
    )
    conversation.save(update_fields=["updated_at"])
    return conversation