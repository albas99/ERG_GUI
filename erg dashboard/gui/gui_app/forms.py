from django import forms
from .models import SigFile
import os

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = SigFile
        fields = ('description', 'file')
