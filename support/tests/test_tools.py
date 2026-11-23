import json
from datetime import timedelta

from django.contrib.auth.models import AnonymousUser
from django.test import TestCase
from django.utils import timezone

from accounts.models import Customer
from shop.models import Order, OrderItem, Payment, Product
from support.tools import build_tools


def call(tools, name, **args):
    tool = next(t for t in tools if t.name == name)
    return json.loads(tool.invoke(args))


def make_order(customer, product, number, *, status="pending", delivered_days_ago=None, txn=None):
    delivered_at = (
        timezone.now() - timedelta(days=delivered_days_ago)
        if delivered_days_ago is not None else None
    )
    order = Order.objects.create(
        customer=customer, order_number=number, status=status,
        shipping_address="SECRET ADDRESS 12, Dhaka", carrier="Pathao",
        tracking_number="TRK-123", delivered_at=delivered_at,
    )
    OrderItem.objects.create(order=order, product=product, quantity=2, unit_price=product.price)
    Payment.objects.create(order=order, method="bkash", status="paid",
                           amount=200, transaction_id=txn)
    return order


class ToolTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.rahim = Customer.objects.create_user("rahim@test.com", "Pass!12345")
        cls.karim = Customer.objects.create_user("karim@test.com", "Pass!12345")
        cls.product = Product.objects.create(sku="T-1", name="Item", price=100, stock=5)
        cls.r1 = make_order(cls.rahim, cls.product, "ORD-RAHIM001",
                            status="delivered", delivered_days_ago=3, txn="BKASH-SECRET-1234")
        cls.r2 = make_order(cls.rahim, cls.product, "ORD-RAHIM002", status="shipped")
        cls.k1 = make_order(cls.karim, cls.product, "ORD-KARIM001", txn="NAGAD-SECRET-9999")

    def setUp(self):
        self.tools = build_tools(self.rahim)

    def test_tool_names(self):
        self.assertEqual(
            {t.name for t in self.tools},
            {"list_my_orders", "get_order_status", "get_order_details", "check_refund_eligibility"},
        )

    def test_no_tool_accepts_a_customer_argument(self):
        for tool in self.tools:
            fields = set(tool.args_schema.model_fields)
            self.assertFalse(fields & {"customer", "customer_id", "user", "user_id", "email"},
                             tool.name)

    def test_list_returns_only_own_orders(self):
        result = call(self.tools, "list_my_orders")
        numbers = {o["order_number"] for o in result["orders"]}
        self.assertEqual(numbers, {"ORD-RAHIM001", "ORD-RAHIM002"})
        raw = self.tools[0].invoke({})
        self.assertNotIn("ORD-KARIM001", raw)
        self.assertNotIn("karim@test.com", raw)

    def test_status_of_own_order(self):
        result = call(self.tools, "get_order_status", order_number="ORD-RAHIM002")
        self.assertEqual(result["status"], "shipped")
        self.assertEqual(result["carrier"], "Pathao")
        self.assertEqual(result["payment_status"], "paid")

    def test_other_customers_order_is_not_found_and_identical_to_missing(self):
        foreign = call(self.tools, "get_order_status", order_number="ORD-KARIM001")
        missing = call(self.tools, "get_order_status", order_number="ORD-NOSUCH01")
        self.assertEqual(foreign["error"], "order_not_found")
        self.assertEqual(foreign, missing)          # অস্তিত্ব ফাঁস নেই

    def test_every_tool_refuses_other_customers_order(self):
        for name in ("get_order_status", "get_order_details", "check_refund_eligibility"):
            result = call(self.tools, name, order_number="ORD-KARIM001")
            self.assertEqual(result["error"], "order_not_found", name)

    def test_extra_arguments_cannot_select_another_customer(self):
        tool = next(t for t in self.tools if t.name == "get_order_status")
        raw = tool.invoke({
            "order_number": "ORD-KARIM001",
            "customer_id": self.karim.id,
            "customer_email": "karim@test.com",
        })
        self.assertEqual(json.loads(raw)["error"], "order_not_found")

    def test_bad_order_number_format_is_rejected_before_database(self):
        for bad in ("ORD-1", "'; DROP TABLE shop_order;--", "", "ORD-RAHIM001; x"):
            result = call(self.tools, "get_order_status", order_number=bad)
            self.assertEqual(result["error"], "invalid_order_number", bad)

    def test_lowercase_and_spaces_are_accepted(self):
        result = call(self.tools, "get_order_status", order_number="  ord-rahim001 ")
        self.assertEqual(result["order_number"], "ORD-RAHIM001")

    def test_details_hide_address_email_and_transaction_id(self):
        raw = next(t for t in self.tools if t.name == "get_order_details").invoke(
            {"order_number": "ORD-RAHIM001"})
        for secret in ("SECRET ADDRESS", "rahim@test.com", "BKASH-SECRET", "1234"):
            self.assertNotIn(secret, raw)
        payment = json.loads(raw)["payments"][0]
        self.assertEqual(set(payment), {"method", "status", "amount", "paid_at"})

    def test_refund_check_uses_real_order_data(self):
        result = call(self.tools, "check_refund_eligibility", order_number="ORD-RAHIM001")
        self.assertTrue(result["can_submit_return_request"])
        self.assertEqual(result["days_since_delivery"], 3)
        not_delivered = call(self.tools, "check_refund_eligibility", order_number="ORD-RAHIM002")
        self.assertEqual(not_delivered["reason_code"], "not_delivered_yet")

    def test_anonymous_user_sees_nothing(self):
        anon = build_tools(AnonymousUser())
        self.assertEqual(call(anon, "list_my_orders")["count"], 0)
        self.assertEqual(
            call(anon, "get_order_status", order_number="ORD-RAHIM001")["error"],
            "order_not_found",
        )

    def test_list_limit_is_clamped(self):
        for i in range(12):
            make_order(self.rahim, self.product, f"ORD-BULK{i:04d}")
        self.assertEqual(call(self.tools, "list_my_orders", limit=999)["count"], 10)
        self.assertEqual(call(self.tools, "list_my_orders", limit=-5)["count"], 1)

    def test_tools_are_bound_to_their_own_user(self):
        karim_tools = build_tools(self.karim)
        result = call(karim_tools, "list_my_orders")
        self.assertEqual({o["order_number"] for o in result["orders"]}, {"ORD-KARIM001"})