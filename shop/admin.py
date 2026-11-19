from django.contrib import admin

from .models import Order, OrderItem, Payment, Product


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("sku", "name", "price", "stock", "is_active")
    search_fields = ("sku", "name")
    list_filter = ("is_active",)


class OrderItemInline(admin.TabularInline):
    model = OrderItem
    extra = 0


class PaymentInline(admin.TabularInline):
    model = Payment
    extra = 0


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ("order_number", "customer", "status", "created_at")
    list_filter = ("status",)
    search_fields = ("order_number", "customer__email")
    readonly_fields = ("order_number", "created_at", "updated_at")
    inlines = [OrderItemInline, PaymentInline]