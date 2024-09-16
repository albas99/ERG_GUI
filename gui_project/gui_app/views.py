from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import SigFileForm
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
import os
import pandas as pd
import json

# Create your views here.
def index(request):
    return HttpResponse("Hello, world!")
    
def simple_upload(request):
    return render(request, 'simple_upload.html')
    # if request.method == 'POST' and request.FILES['sigfile']:
    #     myfile = request.FILES['sigfile']
    #     fs = FileSystemStorage()
    #     filename = fs.save(myfile.name, myfile)
    #     myfile_url = fs.url(filename)
    #     print(filename)
    #     print(myfile_url)
    #     return render(request, 'simple_upload.html', {'myfile_url': myfile_url})
    # else:
    #     return render(request, 'simple_upload.html')
        
def model_upload(request):
    # return render(request, 'model_upload.html')
    if request.method == 'POST':
        form = SigFileForm(request.POST, request.FILES)
        if form.is_valid():
            sigfile = request.FILES['file']
            sigfile_name = sigfile.name
            sigfile_size = sigfile.size
            instance = form.save()
            sigfile_url = instance.file.url
            choices = populate_dropdown(sigfile_url)
            messages.success(request, 'File uploaded successfully!')
            return render(request, 'model_upload.html', {'form': form, 'choices': choices})
        else:
            return render(request, 'model_upload.html', {'form': form})
    else:
        form = SigFileForm()
        return render(request, 'model_upload.html', {'form': form})
        
def populate_dropdown(file_url):
    file_path = os.path.join(settings.MEDIA_ROOT, file_url.lstrip('/media/'))
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep = ',', index_col = 'Time,ms')
        choices =  df.columns.values.tolist()
        return choices
    else:
        print(f'File does not exist: {file_path}')
        return []
        
@csrf_exempt
def get_selection(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        selected_value = data['selected_value']
        return JsonResponse({'message': f'You selected: {selected_value}'})
    else:
        return JsonResponse({'message': ''})