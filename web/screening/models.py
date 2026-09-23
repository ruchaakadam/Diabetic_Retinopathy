from django.db import models
from django.contrib.auth.models import User


class Screening(models.Model):

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="screenings"
    )

    image = models.ImageField(
        upload_to="screenings/"
    )

    screening_date = models.DateTimeField(
        auto_now_add=True
    )

    quality_score = models.FloatField(
        null=True,
        blank=True
    )

    quality_status = models.CharField(
        max_length=30,
        blank=True,
        null=True
    )

    referable_probability = models.FloatField(
        null=True,
        blank=True
    )

    decision = models.CharField(
        max_length=100,
        blank=True,
        null=True
    )

    classification_source = models.CharField(
        max_length=100,
        blank=True,
        null=True
    )

    recommendation = models.TextField(
        blank=True,
        null=True
    )

    gradcam = models.CharField(
        max_length=500,
        blank=True,
        null=True
    )

    report = models.CharField(
        max_length=500,
        blank=True,
        null=True
    )

    def __str__(self):
        return (
            f"{self.user.username} - "
            f"{self.screening_date.strftime('%Y-%m-%d %H:%M')}"
        )