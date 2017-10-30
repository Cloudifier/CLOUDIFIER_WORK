# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 09:39:10 2017

@author: High Tech Systems and Software

@project: Cloudifier.NET

@sub-project: Objecte Detection and Localization

@description: 
  Tensorflow based (Object Detection API)
  Architecture and specifications developed by Cloudifier team
  Code implemented and tested by HTSS

@copyright: CLOUDIFIER SRL

"""



import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from matplotlib import patches

# tf object detection module
#from utils import label_map_util
#from utils import visualization_utils as vis_util


def LoadModelMaybe(model_name, download_url, frozen_file = 'frozen_inference_graph.pb'):
  print("Preparing frozen detection model...")
  model_file = model_name  + '.tar.gz'
  frozen_model_file = os.path.join(model_name,frozen_file)
  if not os.path.isfile(frozen_model_file):
    print("Downloading model...",flush=True)
    opener = urllib.request.URLopener()
    opener.retrieve(download_url + model_file, model_file)
    print("Done downloading model.",flush=True)
    print("Extracting ...", flush = True)
    tar_file = tarfile.open(model_file)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())    
    print("Done extraction.")
  else:
    print("Model allready downloaded.")
  return
  
# Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#object detection api model
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# images to test
PATH_TO_TEST_IMAGES_DIR = '_test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection','data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

##
## Download Model
##
LoadModelMaybe(MODEL_NAME, DOWNLOAD_BASE)
    

##
## Load a (frozen) Tensorflow model into memory
##
print("Preparing graph...", flush = True)
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')    
print("Done preparing graph.",flush = True)    
boxes_list = []
score_list = []
class_list = []
numdt_list = []    
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      print("Running session...", flush = True)
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      print("Done running session.", flush = True)
      boxes_list.append(boxes)
      score_list.append(scores)
      class_list.append(classes)
      numdt_list.append(num_detections)
      
      # Visualization of the results of a detection.
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image_np,
#         np.squeeze(boxes),
#         np.squeeze(classes).astype(np.int32),
#         np.squeeze(scores),
#         category_index,
#         use_normalized_coordinates=True,
#         line_thickness=8)
      
      # Create figure and axes
      fig,ax = plt.subplots(1, figsize=IMAGE_SIZE)
      
      # Display the image
      ax.imshow(image_np)
      img_h = image_np.shape[0]
      img_w = image_np.shape[1]
      
      score_threshold = 0.5
      for i in range(boxes.shape[1]):
        if scores[0,i]>score_threshold:
          coords = boxes[0,i,:]
          ymin = coords[0] * img_h
          xmin = coords[1] * img_w
          ymax = coords[2] * img_h
          xmax = coords[3] * img_w
          # Create a Rectangle patch
          rect = patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,linewidth=1,edgecolor='r',facecolor='none')
        
          # Add the patch to the Axes
          ax.add_patch(rect)
          ax.text(xmin, ymax, 'CLS:{:.0f} ({:.1f}%)'.format(classes[0,i],scores[0,i]*100)
                  ,horizontalalignment='left',verticalalignment='bottom', color='red',
                  weight = 'bold')
      
      plt.show()
      
      #fig = plt.figure(figsize=IMAGE_SIZE)
      #plt.imshow()
      

