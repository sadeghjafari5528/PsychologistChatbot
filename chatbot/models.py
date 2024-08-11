from django.db import models
from django.contrib.auth.models import User

class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    emotion = models.JSONField()
    disorder = models.JSONField()
    validation = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    gad_7 = models.BooleanField(default=False)
    phq_9 = models.BooleanField(default=False)

    def __str__(self):
        return f'{self.user.username}: {self.message}'

class Question(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    is_validator = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    gad_7_count = models.IntegerField(default=0)
    phq_9_count = models.IntegerField(default=0)
    gad_7_completed = models.BooleanField(default=False)
    phq_9_completed = models.BooleanField(default=False)

    def __str__(self):
        return f'{self.user.username}: {self.created_at}'


class Questionnaire(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    question = models.TextField()
    answer = models.TextField(blank=True)
    gad_7_number = models.IntegerField(default=0)
    phq_9_number = models.IntegerField(default=0)
    is_gad_7 = models.BooleanField(default=False)
    is_phq_9 = models.BooleanField(default=False)
    question = models.TextField()
    answer = models.TextField()
    def __str__(self):
        return f'{self.user.username}: {self.created_at}'
