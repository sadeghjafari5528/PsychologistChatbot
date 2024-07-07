from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    emotion_label = models.TextField()
    emotion_prob = models.FloatField()
    disorder_label = models.TextField()
    disorder_prob = models.FloatField()
    validation_label = models.TextField()
    validation_prob = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username}: {self.message}'