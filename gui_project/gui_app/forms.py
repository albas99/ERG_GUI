from django import forms
from django.forms import ModelForm, Form
from gui_app.models import SigFile

class SigFileForm(ModelForm):
    class Meta:
        model = SigFile
        fields = ['file']
        
        widgets = {
            'file': forms.FileInput(attrs = {
                'required': 'required',
                'accept': '.csv'
            }),
        }