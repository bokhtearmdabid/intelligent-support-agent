from decimal import Decimal

from django.core.cache import cache
from django.db import connection
from django.test.utils import CaptureQueriesContext
from rest_framework.test import APITestCase

from accounts.models import Customer
from shop.models import Order, OrderItem, Payment, Product


def make_order(customer, product, txn=None):
    order = Order.objects.create(customer=customer, shipping_address="Test address")
    OrderItem.objects.create(
        order=order, product=product, quantity=2, unit_price=product.price
    )
    Payment.objects.create(
        order=order,
        method=Payment.Method.BKASH,
        status=Payment.Status.PAID,
        amount=Decimal("200.00"),
        transaction_id=txn,
    )
    return order


class OrderApiTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.rahim = Customer.objects.create_user("rahim@test.com", "Pass!12345")
        cls.karim = Customer.objects.create_user("karim@test.com", "Pass!12345")
        cls.product = Product.objects.create(
            sku="T-1", name="Test Item", price=Decimal("100.00"), stock=5
        )
        cls.rahim_order = make_order(cls.rahim, cls.product, "BKASH-SECRET-1234")
        cls.karim_order = make_order(cls.karim, cls.product, "NAGAD-SECRET-9999")

    def setUp(self):
        cache.clear()  # throttle-এর গণনা এক test থেকে অন্যে যেন না গড়ায়

    # ---------- Authentication ----------
    def test_anonymous_cannot_list_orders(self):
        response = self.client.get("/api/orders/")
        self.assertEqual(response.status_code, 403)

    # ---------- Ownership ----------
    def test_list_returns_only_own_orders(self):
        self.client.force_authenticate(self.rahim)
        response = self.client.get("/api/orders/")
        self.assertEqual(response.status_code, 200)
        numbers = [o["order_number"] for o in response.data["results"]]
        self.assertEqual(numbers, [self.rahim_order.order_number])

    def test_own_order_detail(self):
        self.client.force_authenticate(self.rahim)
        response = self.client.get(f"/api/orders/{self.rahim_order.order_number}/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["total_amount"], "200.00")
        self.assertEqual(len(response.data["items"]), 1)

    def test_other_customers_order_returns_404(self):
        self.client.force_authenticate(self.rahim)
        response = self.client.get(f"/api/orders/{self.karim_order.order_number}/")
        self.assertEqual(response.status_code, 404)

    def test_other_customers_order_status_returns_404(self):
        self.client.force_authenticate(self.rahim)
        response = self.client.get(
            f"/api/orders/{self.karim_order.order_number}/status/"
        )
        self.assertEqual(response.status_code, 404)

    def test_foreign_and_nonexistent_orders_look_identical(self):
        """অন্যের order আর অস্তিত্বহীন order-এর response একই: কোনো তথ্য ফাঁস নেই।"""
        self.client.force_authenticate(self.rahim)
        foreign = self.client.get(f"/api/orders/{self.karim_order.order_number}/")
        missing = self.client.get("/api/orders/ORD-DOESNOTEXIST/")
        self.assertEqual(foreign.status_code, missing.status_code)
        self.assertEqual(foreign.data, missing.data)

    # ---------- Data exposure ----------
    def test_sensitive_data_not_exposed(self):
        self.client.force_authenticate(self.rahim)
        response = self.client.get(f"/api/orders/{self.rahim_order.order_number}/")
        body = response.content.decode()
        self.assertNotIn("rahim@test.com", body)
        self.assertNotIn("BKASH-SECRET-1234", body)
        masked = response.data["payments"][0]["transaction_id"]
        self.assertTrue(masked.endswith("1234"))
        self.assertTrue(masked.startswith("*"))

    def test_status_endpoint_exposes_only_whitelisted_fields(self):
        self.client.force_authenticate(self.rahim)
        response = self.client.get(
            f"/api/orders/{self.rahim_order.order_number}/status/"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.data.keys()),
            {
                "order_number", "status", "status_display", "carrier",
                "tracking_number", "estimated_delivery", "shipped_at",
                "delivered_at", "payment_status",
            },
        )

    # ---------- Read-only ----------
    def test_write_methods_not_allowed(self):
        self.client.force_authenticate(self.rahim)
        url = f"/api/orders/{self.rahim_order.order_number}/"
        for method in (
            self.client.post, self.client.put,
            self.client.patch, self.client.delete,
        ):
            self.assertEqual(method(url).status_code, 405)

    # ---------- Performance ----------
    def test_list_query_count_does_not_grow_with_orders(self):
        """N+1 সমস্যা নেই: order বাড়লেও query সংখ্যা একই।"""
        self.client.force_authenticate(self.rahim)
        with CaptureQueriesContext(connection) as before:
            self.client.get("/api/orders/")
        for _ in range(5):
            make_order(self.rahim, self.product)
        with CaptureQueriesContext(connection) as after:
            self.client.get("/api/orders/")
        self.assertEqual(len(before), len(after))


class ProductApiTests(APITestCase):
    def setUp(self):
        cache.clear()

    def test_public_list_hides_inactive_products_and_exact_stock(self):
        Product.objects.create(sku="A", name="Active", price=Decimal("10"), stock=7)
        Product.objects.create(
            sku="B", name="Hidden", price=Decimal("10"), stock=7, is_active=False
        )
        response = self.client.get("/api/products/")
        self.assertEqual(response.status_code, 200)
        results = response.data["results"]
        self.assertEqual([p["sku"] for p in results], ["A"])
        self.assertNotIn("stock", results[0])
        self.assertTrue(results[0]["in_stock"])