from datetime import timedelta

from django.utils import timezone

from .models import Order, Payment

# knowledge_base/returns_and_refunds.md-এর সাথে মিল থাকতে হবে (test এটা পাহারা দেয়)
RETURN_WINDOW_DAYS = 7
CHANGE_OF_MIND_WINDOW_DAYS = 3

NOTE = (
    "This only checks the time window. It does not mean a refund or replacement "
    "is approved: the support team reviews every request and decides. "
    "Item condition and warranty claims are not covered by this check."
)


def _window(delivered_at, days, now):
    ends = delivered_at + timedelta(days=days)
    return {
        "window_days": days,
        "window_ends": timezone.localtime(ends).date().isoformat(),
        "within_window": now <= ends,        # শেষ মুহূর্ত পর্যন্ত বৈধ
    }


def assess_return_eligibility(order, now=None):
    """শুধু সময়সীমার হিসাব। LLM নয়, সাধারণ Python।"""
    now = now or timezone.now()
    payment_received = any(
        p.status == Payment.Status.PAID for p in order.payments.all()
    )
    result = {
        "order_number": order.order_number,
        "order_status": order.status,
        "payment_received": payment_received,
    }

    if order.status == Order.Status.CANCELLED:
        result.update(reason_code="order_cancelled", can_submit_return_request=False)
    elif order.status != Order.Status.DELIVERED:
        result.update(reason_code="not_delivered_yet", can_submit_return_request=False)
    elif order.delivered_at is None:
        result.update(reason_code="delivery_date_missing", can_submit_return_request=False)
    else:
        damaged = _window(order.delivered_at, RETURN_WINDOW_DAYS, now)
        change_of_mind = _window(order.delivered_at, CHANGE_OF_MIND_WINDOW_DAYS, now)
        result.update(
            delivered_at=order.delivered_at.isoformat(),
            days_since_delivery=(now - order.delivered_at).days,
            damaged_defective_or_wrong_item=damaged,
            change_of_mind_unopened=change_of_mind,
            reason_code="within_window" if damaged["within_window"] else "window_expired",
            can_submit_return_request=damaged["within_window"],
        )

    result["note"] = NOTE
    return result