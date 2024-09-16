from django.db import models
from django import forms
from django.forms import widgets

# Create your models here.
class SigFile(models.Model):
    file = models.FileField(upload_to = 'sigfiles/')
    uploaded_at = models.DateTimeField(auto_now_add=True)