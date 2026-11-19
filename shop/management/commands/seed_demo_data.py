from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from accounts.models import Customer
from shop.models import Order, OrderItem, Payment, Product

DEMO_PASSWORD = "Demo@12345"   # শুধু ডেভেলপমেন্টের জন্য


class Command(BaseCommand):
    help = "Create demo customers, products, orders and payments."

    @transaction.atomic
    def handle(self, *args, **options):
        if Product.objects.exists():
            self.stdout.write(self.style.WARNING(
                "Products already exist. Skipping seed (run on a fresh DB)."
            ))
            return

        rahim = self._customer("rahim@example.com", "Rahim", "Uddin")
        karim = self._customer("karim@example.com", "Karim", "Hossain")
        self._customer(
            "support@example.com", "Sadia", "Support",
            role=Customer.Role.SUPPORT_AGENT, is_staff=True,
        )

        earbuds = self._product("EAR-001", "Wireless Earbuds", 2490, 50)
        watch = self._product("WAT-001", "Smart Watch", 4990, 20)
        speaker = self._product("SPK-001", "Bluetooth Speaker", 3290, 30)
        case = self._product("CAS-001", "Phone Case", 450, 200)
        bank = self._product("PWR-001", "Power Bank 10000mAh", 2190, 40)
        keyboard = self._product("KEY-001", "Mechanical Keyboard", 5500, 15)

        now = timezone.now()
        address = "House 12, Road 5, Dhanmondi, Dhaka"

        # Rahim: ৩ দিন আগে ডেলিভারি, রিফান্ড-উপযোগী সময়ের ভেতরে
        self._order(
            rahim, "ORD-DEMO0001", Order.Status.DELIVERED, address,
            [(earbuds, 1), (case, 2)], days_ago=6,
            delivered_at=now - timedelta(days=3),
            carrier="Pathao", tracking_number="PTH-100001",
            method=Payment.Method.BKASH, pay_status=Payment.Status.PAID,
            txn="BKASH-TXN-0001",
        )
        # Rahim: পথে আছে, COD (পেমেন্ট এখনো pending)
        self._order(
            rahim, "ORD-DEMO0002", Order.Status.SHIPPED, address,
            [(watch, 1)], days_ago=2,
            shipped_at=now - timedelta(days=1),
            estimated_delivery=(now + timedelta(days=2)).date(),
            carrier="Steadfast", tracking_number="STD-200002",
            method=Payment.Method.COD, pay_status=Payment.Status.PENDING,
        )
        # Rahim: ৪০ দিন আগের ডেলিভারি, রিফান্ডের সময়সীমার বাইরে
        self._order(
            rahim, "ORD-DEMO0003", Order.Status.DELIVERED, address,
            [(speaker, 1)], days_ago=45,
            delivered_at=now - timedelta(days=40),
            carrier="Pathao", tracking_number="PTH-100003",
            method=Payment.Method.CARD, pay_status=Payment.Status.PAID,
            txn="CARD-TXN-0003",
        )
        # Karim: আলাদা মালিকের order (ownership পরীক্ষার জন্য)
        self._order(
            karim, "ORD-DEMO0004", Order.Status.CONFIRMED,
            "Flat 3B, Agrabad, Chattogram",
            [(bank, 1), (keyboard, 1)], days_ago=1,
            method=Payment.Method.NAGAD, pay_status=Payment.Status.PAID,
            txn="NAGAD-TXN-0004",
        )
        # Karim: ব্যর্থ পেমেন্টসহ pending order
        self._order(
            karim, "ORD-DEMO0005", Order.Status.PENDING,
            "Flat 3B, Agrabad, Chattogram",
            [(case, 1)], days_ago=0,
            method=Payment.Method.CARD, pay_status=Payment.Status.FAILED,
        )

        self.stdout.write(self.style.SUCCESS(
            f"Seed complete. Demo password for new users: {DEMO_PASSWORD}"
        ))

    # ---------- helpers ----------
    def _customer(self, email, first, last, **extra):
        user, created = Customer.objects.get_or_create(
            email=email,
            defaults={"first_name": first, "last_name": last, **extra},
        )
        if created:
            user.set_password(DEMO_PASSWORD)
            user.save()
        return user

    def _product(self, sku, name, price, stock):
        return Product.objects.create(
            sku=sku, name=name, price=price, stock=stock,
            description=f"{name} - demo product.",
        )

    def _order(self, customer, number, status, address, items, days_ago,
               method, pay_status, txn=None, **extra):
        order = Order.objects.create(
            customer=customer, order_number=number, status=status,
            shipping_address=address, **extra,
        )
        for product, qty in items:
            OrderItem.objects.create(
                order=order, product=product,
                quantity=qty, unit_price=product.price,
            )
        paid_at = timezone.now() if pay_status == Payment.Status.PAID else None
        Payment.objects.create(
            order=order, method=method, status=pay_status,
            amount=order.total_amount, transaction_id=txn, paid_at=paid_at,
        )
        # created_at auto_now_add, তাই update() দিয়ে পুরনো তারিখ বসাচ্ছি
        Order.objects.filter(pk=order.pk).update(
            created_at=timezone.now() - timedelta(days=days_ago)
        )
        return order