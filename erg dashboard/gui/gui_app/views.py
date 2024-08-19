from django.shortcuts import redirect, render
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.views.generic.edit import FormView
from django.contrib import messages
from .forms import UploadFileForm
import csv, io
import pandas as pd

# Create your views here.

def index(request):
    return render(request, "index.html")
    # return HttpResponse("Hello, World!")


def upload_file(request):
    file_data = {}
    if request.method =="POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            # df = pd.read_csv(uploaded_file.file, sep = ',', index_col = 'Time,ms')
            # signal_ids = df.columns.tolist()
            # choice_list = [(id, id) for id in signal_ids]
            # handle_uploaded_file(request.FILES["file"])
            form.save()
            messages.success(request, "File uploaded successfully")
            # return redirect("index")
            # return render(request, 'index.html', {'choices': choice_list})
            file_data.update({'file_url': uploaded_file.model.url})
            return JsonResponse(file_data)
        messages.error(request, "Failed to upload file")
    form = UploadFileForm()
    return render(request, "index.html", {"form": form})

def process_column(request):
    if request.method == 'POST':
        column_name = request.POST.get('column_name')
        file_url = request.POST.get('file_url')
        df = pd.read_csv(file_url)
        column_data = df[column_name]
        return JsonResponse({'column_data': column_data.tolist()})
    else:
        return JsonResponse({'error': 'Invalid request method'})