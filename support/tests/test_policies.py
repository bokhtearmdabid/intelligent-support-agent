from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.test import TestCase
from django.utils import timezone

from accounts.models import Customer
from shop.models import Order, Payment
from shop.policies import (
    CHANGE_OF_MIND_WINDOW_DAYS,
    RETURN_WINDOW_DAYS,
    assess_return_eligibility,
)

NOW = timezone.now()


class EligibilityTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = Customer.objects.create_user("p@test.com", "Pass!12345")

    def order(self, status=Order.Status.DELIVERED, delivered_ago=None, pay=Payment.Status.PAID):
        delivered_at = NOW - delivered_ago if delivered_ago is not None else None
        order = Order.objects.create(
            customer=self.user, status=status, shipping_address="x",
            delivered_at=delivered_at,
        )
        Payment.objects.create(order=order, method="bkash", status=pay, amount=100)
        return order

    def test_recently_delivered_is_eligible_in_both_windows(self):
        result = assess_return_eligibility(self.order(delivered_ago=timedelta(days=2)), NOW)
        self.assertTrue(result["can_submit_return_request"])
        self.assertEqual(result["reason_code"], "within_window")
        self.assertTrue(result["damaged_defective_or_wrong_item"]["within_window"])
        self.assertTrue(result["change_of_mind_unopened"]["within_window"])
        self.assertTrue(result["payment_received"])

    def test_change_of_mind_window_is_shorter(self):
        result = assess_return_eligibility(self.order(delivered_ago=timedelta(days=4)), NOW)
        self.assertTrue(result["damaged_defective_or_wrong_item"]["within_window"])
        self.assertFalse(result["change_of_mind_unopened"]["within_window"])
        self.assertTrue(result["can_submit_return_request"])

    def test_expired_window(self):
        result = assess_return_eligibility(self.order(delivered_ago=timedelta(days=40)), NOW)
        self.assertFalse(result["can_submit_return_request"])
        self.assertEqual(result["reason_code"], "window_expired")
        self.assertEqual(result["days_since_delivery"], 40)

    def test_boundary_exactly_at_window_end_is_still_eligible(self):
        order = self.order(delivered_ago=timedelta(days=RETURN_WINDOW_DAYS))
        self.assertTrue(assess_return_eligibility(order, NOW)["can_submit_return_request"])

    def test_one_second_after_window_end_is_not_eligible(self):
        order = self.order(delivered_ago=timedelta(days=RETURN_WINDOW_DAYS, seconds=1))
        self.assertFalse(assess_return_eligibility(order, NOW)["can_submit_return_request"])

    def test_not_delivered_and_cancelled(self):
        shipped = assess_return_eligibility(self.order(status=Order.Status.SHIPPED), NOW)
        self.assertEqual(shipped["reason_code"], "not_delivered_yet")
        self.assertFalse(shipped["can_submit_return_request"])
        cancelled = assess_return_eligibility(self.order(status=Order.Status.CANCELLED), NOW)
        self.assertEqual(cancelled["reason_code"], "order_cancelled")

    def test_delivered_without_date_is_flagged_not_guessed(self):
        result = assess_return_eligibility(self.order(delivered_ago=None), NOW)
        self.assertEqual(result["reason_code"], "delivery_date_missing")
        self.assertFalse(result["can_submit_return_request"])

    def test_unpaid_order_reports_no_payment(self):
        order = self.order(delivered_ago=timedelta(days=1), pay=Payment.Status.PENDING)
        self.assertFalse(assess_return_eligibility(order, NOW)["payment_received"])

    def test_result_says_it_is_not_an_approval(self):
        result = assess_return_eligibility(self.order(delivered_ago=timedelta(days=1)), NOW)
        self.assertIn("does not mean a refund", result["note"])
        self.assertNotIn("approved", result["reason_code"])


class PolicyDriftTests(TestCase):
    def test_code_constants_match_the_policy_document(self):
        """কোডের সংখ্যা আর ডকুমেন্টের সংখ্যা আলাদা হয়ে গেলে এই test ফেল করবে।"""
        text = (Path(settings.KNOWLEDGE_DIR) / "returns_and_refunds.md").read_text(encoding="utf-8")
        self.assertIn(f"within {RETURN_WINDOW_DAYS} days of the delivery date", text)
        self.assertIn(f"within {CHANGE_OF_MIND_WINDOW_DAYS} days of delivery", text)