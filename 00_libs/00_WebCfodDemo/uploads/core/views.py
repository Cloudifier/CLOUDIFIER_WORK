from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from uploads.core.models import Document
from uploads.core.forms import DocumentForm

from uploads.core.obj_detect_api.cfod import FastObjectDetector
from uploads import urls

import numpy as np
from scipy import misc
import cv2
import magic
import datetime
import ffmpy
import os

SCORE_THRESHOLD  = 0.5
FRAME_W, FRAME_H = 1280, 720

def home(request):
    documents = Document.objects.all()
    return render(request, 'core/home.html', { 'documents': documents })

def convert_avi_to_mp4(avi_file_path, output_name):
    ff = ffmpy.FFmpeg(inputs={avi_file_path: None},
                      outputs={output_name: None})
    ff.run()
    
def process_image(image_full_path):
    global FRAME_W
    global FRAME_H
    global SCORE_THRESHOLD

    config_path = os.path.join('uploads\core\obj_detect_api', 'config.txt')
    cfod = FastObjectDetector(score_threshold = SCORE_THRESHOLD, new_faces_file = True,
                             config_file = config_path)
    cfod.prepare(image_shape = (FRAME_H, FRAME_W))

    image_array     = misc.imread(image_full_path)
    image_array     = cv2.resize(image_array, (FRAME_W, FRAME_H))
    print(image_array.shape)
    processed_array = cfod.predict_img(image_array)
    new_full_path   = os.path.join('media', "processed_" + os.path.basename(image_full_path))
    misc.imsave(new_full_path, processed_array)

    return new_full_path, processed_array.shape[1] + 0.013 * processed_array.shape[1]

def process_video(video_full_path):
    global FRAME_W
    global FRAME_H
    global SCORE_THRESHOLD

    config_path = os.path.join('uploads\core\obj_detect_api', 'config.txt')
    cfod = FastObjectDetector(score_threshold = SCORE_THRESHOLD, new_faces_file = True, 
                             config_file = config_path)
    cfod.prepare(image_shape = (FRAME_H, FRAME_W))

    video      = cv2.VideoCapture(video_full_path)
    fourcc     = int(video.get(cv2.CAP_PROP_FOURCC))
    fourcc     = cv2.VideoWriter_fourcc(*'XVID')
    fps        = video.get(cv2.CAP_PROP_FPS)
    frame_size = (FRAME_W, FRAME_H)
    
    video_full_path , _ = os.path.splitext(video_full_path) 
    new_full_path = os.path.join('media/', "processed_" + os.path.basename(video_full_path) + ".avi")
    new_video = cv2.VideoWriter(new_full_path, fourcc , fps, frame_size)

    while(video.isOpened()):
        ret, frame = video.read()
        if ret:
            
            frame = cv2.resize(frame, (FRAME_W, FRAME_H)) 
            frame = cfod.predict_img(frame)
        
            new_video.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    new_video.release()
    cv2.destroyAllWindows()

    return new_full_path, frame_size[0]

def loader(request):
    return render(request, "core/loader.html")

def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        
        item_full_path = os.path.join('media/', filename)
        if "image" in magic.from_file(item_full_path).lower(): 
            new_item_path, width = process_image(item_full_path)
            new_full_path = new_item_path
            is_video = False 
        else:
            new_item_path, width = process_video(item_full_path)
            new_full_path, _ = os.path.splitext(new_item_path)
            new_full_path += ".mp4"
            is_video = True
            convert_avi_to_mp4(new_item_path, new_full_path)

        processed_file_url = fs.url(os.path.basename(new_full_path))
        return render(request, 'core/new.html', {
            'processed_file_url': processed_file_url, 'is_video': is_video, "width": width
        })

    return render(request, 'core/new.html')


def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        for filename, file in request.FILES.items():
            name = request.FILES[filename].name
            break

        if form.is_valid():
            form.save()

            item_full_path = os.path.join('media/', name)
            if "image" in magic.from_file(item_full_path).lower(): 
                new_item_path, width = process_image(item_full_path)
                new_full_path = new_item_path
                is_video = False 
            else:
                new_item_path, width = process_video(item_full_path)
                new_full_path, _ = os.path.splitext(new_item_path)
                new_full_path += ".mp4"
                is_video = True
                convert_avi_to_mp4(new_item_path, new_full_path)

            processed_file_url = fs.url(os.path.basename(new_full_path))
            return render(request, 'core/model_form_upload.html', {'form': form, 
                'processed_file_url': processed_file_url, 'is_video': is_video, "width": width})
        else:
            form = DocumentForm()
            
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })
