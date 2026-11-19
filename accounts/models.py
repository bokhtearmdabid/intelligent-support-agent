from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models


class CustomerManager(BaseUserManager):
    use_in_migrations = True

    def _create_user(self, email, password, **extra_fields):
        if not email:
            raise ValueError("Email is required")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)   # password hash হয়ে সেভ হয়
        user.save(using=self._db)
        return user

    def create_user(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", False)
        extra_fields.setdefault("is_superuser", False)
        return self._create_user(email, password, **extra_fields)

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("role", Customer.Role.ADMIN)
        return self._create_user(email, password, **extra_fields)


class Customer(AbstractUser):
    class Role(models.TextChoices):
        CUSTOMER = "customer", "Customer"
        SUPPORT_AGENT = "support_agent", "Support Agent"
        ADMIN = "admin", "Admin"

    username = None                                  # username বাদ
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, blank=True)
    role = models.CharField(
        max_length=20, choices=Role.choices, default=Role.CUSTOMER
    )
    created_at = models.DateTimeField(auto_now_add=True)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    objects = CustomerManager()

    def __str__(self):
        return self.email
