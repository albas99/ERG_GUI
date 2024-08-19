from django.db import models

# Create your models here.

class SigFile(models.Model):
    description = models.CharField(max_length=1000, blank = True)
    file = models.FileField(upload_to='files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)