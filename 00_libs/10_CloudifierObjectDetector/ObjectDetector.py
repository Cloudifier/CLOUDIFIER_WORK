# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 09:39:10 2017

@author: Cloudifier SRL, High Tech Systems and Software

@project: Cloudifier.NET

@sub-project: Cloudifier Object Detection based on Tensor Flow (CODTF)

@description: 
  Tensorflow based (Object Detection API)
  Architecture and specifications developed by Cloudifier team
  Code implemented and tested by HTSS/Cloudifier

@copyright: CLOUDIFIER SRL

@todo:
  - finish multi-scene inference
  - segmentation
  

"""

import tensorflow as tf
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import json

import pandas as pd


from matplotlib import pyplot as plt
from PIL import Image

from matplotlib import patches



import time

def load_module(module_name, file_name):
  """
  loads modules from _pyutils Google Drive repository
  usage:
    module = load_module("logger", "logger.py")
    logger = module.Logger()
  """
  from importlib.machinery import SourceFileLoader
  home_dir = os.path.expanduser("~")
  valid_paths = [
                 os.path.join(home_dir, "Google Drive"),
                 os.path.join(home_dir, "GoogleDrive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                 os.path.join("C:/", "GoogleDrive"),
                 os.path.join("C:/", "Google Drive"),
                 os.path.join("D:/", "GoogleDrive"),
                 os.path.join("D:/", "Google Drive"),
                 ]

  drive_path = None
  for path in valid_paths:
    if os.path.isdir(path):
      drive_path = path
      break

  if drive_path is None:
    raise Exception("Couldn't find google drive folder!")

  utils_path = os.path.join(drive_path, "_pyutils")
  print("Loading [{}] package...".format(os.path.join(utils_path,file_name)),flush = True)
  logger_lib = SourceFileLoader(module_name, os.path.join(utils_path, file_name)).load_module()
  print("Done loading [{}] package.".format(os.path.join(utils_path,file_name)),flush = True)

  return logger_lib


class ObjectDetector:
  def __init__(self, config_file = 'config.txt'):
    self.detection_threshold = 0.5
    self.IMAGE_SIZE = (12,8)
    self.config_file = config_file
    self.LoadConfig()
    print("Base [{}]".format(self._base), flush = True)
    log_module = load_module("logger", "logger.py")    
    self.log = log_module.Logger(lib_name = "CODTF", base_folder = self._base)
    
    self.SetupRepository()
    
    self.LoadModelsConfig()
    
    self.SetupAllModels()
    
    (self.DEFAULT_DETECTOR_FILE, 
     self.CLASSES_FILE, 
     self.DEFAULT_MODEL_NAME) = self.LoadModel(self.DEFAULT_MODEL)
    
    if self.CLASSES_FILE != '':
      self.CLASSES = self.LoadClasses()
    else:
      self.CLASSES = None
    
    self.df_time = pd.DataFrame(columns = ["MODEL","IMAGE_SHAPE",
                                           "TIME", "AVG_ACC","NR_IDs"])
      
    self.SetActiveGraph(self.DEFAULT_DETECTOR_FILE, self.DEFAULT_MODEL_NAME)
    
    self.log.VerboseLog("Default detector [{}] READY.".format(self.DEFAULT_MODEL_NAME))
    return
  
  
  
  
  def SetActiveGraph(self, detector_graph_file, model_name):
    self.LOADED_GRAPH = self.LoadDetectionGraph(detector_graph_file)
    self.LOADED_GRAPH_FILE = detector_graph_file
    self.LOADED_MODEL = model_name
    return
  
  
  
  def SetupAllModels(self):
    for model in self.MODELS:
      self.LoadModel(model)
    return
  
  
  def _log_model_time(self, model, shape, secs, acc, preds):
    i = self.df_time.shape[0]
    s_shape = "{}".format(shape)
    self.df_time.loc[i] = [model, s_shape, secs, acc, preds]
    return
  
  def ShowTimings(self):
    old = pd.options.display.width
    pd.options.display.width =  1000
    self.log.VerboseLog("Results:\n{}".format(self.df_time))
    pd.options.display.width =  old
    return
  
  def AnalyzeSingleImage(self, image_file):   
    l = self.log
    with self.LOADED_GRAPH.as_default():
      with tf.Session(graph=self.LOADED_GRAPH) as sess:
        image = Image.open(image_file)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = self.load_image_into_numpy_array(image)
        img_h = image_np.shape[0]
        img_w = image_np.shape[1]
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.LOADED_GRAPH.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.LOADED_GRAPH.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.LOADED_GRAPH.get_tensor_by_name('detection_scores:0')
        classes = self.LOADED_GRAPH.get_tensor_by_name('detection_classes:0')
        num_detections = self.LOADED_GRAPH.get_tensor_by_name('num_detections:0')
        # Actual detection.
        l.VerboseLog("Running session for one image...")
        start_time = time.time()
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        end_time = time.time()
        l.VerboseLog("[{}] Done running one image in {:.2f}sec.".format(
            self.LOADED_MODEL,
            end_time - start_time))
        
        
    nr_detections = 0
    result_boxes = []
    result_scores = []
    result_classes = []
    accuracy_list = []
    self.LAST_DETECTION_CLASSES = []
    for i in range(boxes.shape[1]):
      score = scores[0,i]
      if score >= self.detection_threshold:
        accuracy_list.append(score)
        coords = boxes[0,i,:]
        ymin = coords[0] * img_h
        xmin = coords[1] * img_w
        ymax = coords[2] * img_h
        xmax = coords[3] * img_w   
        det_class = classes[0,i]
        nr_detections += 1
        result_boxes.append((xmin,ymin,xmax,ymax))
        result_scores.append(score)
        result_classes.append(det_class)
        self.LAST_DETECTION_CLASSES.append(self.CLASSES[int(det_class)])
    avg_acc = 0
    if len(accuracy_list)>0:
      avg_acc = np.mean(accuracy_list)
    self._log_model_time(self.LOADED_MODEL, (img_w,img_h), 
                         end_time - start_time, avg_acc, len(accuracy_list))
    return result_boxes,result_scores,result_classes
  



  def AnalyzeImages(self, image_file_list):   
    l = self.log
    l.VerboseLog("Processing {} images...".format(len(image_file_list)))
    with self.LOADED_GRAPH.as_default():
      with tf.Session(graph=self.LOADED_GRAPH) as sess:
        batch_start_time = time.time()
        for image_file in image_file_list:
          image = Image.open(image_file)
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          image_np = self.load_image_into_numpy_array(image)
          img_h = image_np.shape[0]
          img_w = image_np.shape[1]
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = self.LOADED_GRAPH.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = self.LOADED_GRAPH.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = self.LOADED_GRAPH.get_tensor_by_name('detection_scores:0')
          classes = self.LOADED_GRAPH.get_tensor_by_name('detection_classes:0')
          num_detections = self.LOADED_GRAPH.get_tensor_by_name('num_detections:0')
          # Actual detection.
          l.VerboseLog("Running session...") 
          start_time = time.time()
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          end_time = time.time()        
          l.VerboseLog("Done running session in {:.2f}s.".format(end_time-start_time))
        batch_end_time = time.time()
        # end batch
      # end session
    # end graph
    
    l.VerboseLog("Done batch images in {:.2f}s.".format(batch_end_time-batch_start_time))
    
    
    NowProcessListsOfResults() 
    
    nr_detections = 0
    result_boxes = []
    result_scores = []
    result_classes = []
    for i in range(boxes.shape[1]):
      if scores[0,i] >= self.detection_threshold:
        coords = boxes[0,i,:]
        ymin = coords[0] * img_h
        xmin = coords[1] * img_w
        ymax = coords[2] * img_h
        xmax = coords[3] * img_w   
        det_class = classes[0,i]
        det_score = scores[0,i]
        nr_detections += 1
        result_boxes.append((xmin,ymin,xmax,ymax))
        result_scores.append(det_score)
        result_classes.append(det_class)        
    return result_boxes,result_scores,result_classes
  
  
  def ShowLastDetectionClasses(self):
    self.log.VerboseLog("[{}] detected classes: {}".format(self.LOADED_MODEL,
                        self.LAST_DETECTION_CLASSES))
    #for cls in self.LAST_DETECTION_CLASSES:
    #  self.log.VerboseLog("   - "+cls)
    return
      
  
  
  def DrawImage(self, image_file, boxes, scores, classes):
    image = Image.open(image_file)
    image_np = self.load_image_into_numpy_array(image)
    fig,ax = plt.subplots(1, figsize=self.IMAGE_SIZE)
    ax.imshow(image_np)

    for box,score,cls in zip(boxes,scores,classes):
      xmin,ymin,xmax,ymax = box
      rect = patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,linewidth=1,edgecolor='r',facecolor='none')
      ax.add_patch(rect)
      if self.CLASSES != None:
        ax.text(xmin, ymax, 'CLS:{} ({:.1f}%)'.format(self.CLASSES[int(cls)],score*100)
                    ,horizontalalignment='left',verticalalignment='bottom', color='red',
                    weight = 'bold')    
      else:
        ax.text(xmin, ymax, 'CLS:{} ({:.1f}%)'.format(int(cls),score*100)
                    ,horizontalalignment='left',verticalalignment='bottom', color='red',
                    weight = 'bold')    
    self.log.VerboseLog("Rendering full scene [{}]...".format(self.LOADED_MODEL))    
    plt.show()
    return
  
  
  
  
  
  def FullSingleSceneInference(self, image_file, USE_MODEL = 'DEFAULT'):
    """
     Performs full scene inference on a single image
     will use default model or any model from configuration file if specified
    """  
    if USE_MODEL != 'DEFAULT':
      detector_file, classes, model_name = self.LoadModel(USE_MODEL)
      self.SetActiveGraph(detector_graph_file =  detector_file,
                          model_name = model_name)
      
    b,s,c = self.AnalyzeSingleImage(image_file = image_file)
    self.DrawImage(image_file = image_file, boxes = b,
                   scores = s, classes = c
                   )
    return
  
  
  
  
  def LoadClasses(self, cfile = ''):  
    if cfile == '':
      cfile = self.CLASSES_FILE
    with open(cfile) as f:
      lines = f.read().splitlines()
    labels = list(lines)
    for i,line in enumerate(lines):
      labels[i] = line.split(": ")[1]
    return labels
  
    
  
  
  
  def LoadDetectionGraph(self, graph_file):
    l = self.log    
    l.VerboseLog("Preparing graph [...{}]...".format(graph_file[-60:]))
    start_time = time.time()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(graph_file, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')    
    end_time = time.time()
    l.VerboseLog("Done preparing graph in {:.2f}s.".format(end_time-start_time))        
    return detection_graph
  
  
  
  
  def LoadModelsConfig(self):
    self.MODELS = self.CONFIG["MODELS"]
    self.DEFAULT_MODEL = self.CONFIG["DEFAULT_MODEL"]
    self.log.VerboseLog("Default model: {}".format(self.MODELS[self.DEFAULT_MODEL]["MODEL_NAME"]))
    return
  
  
  
  def LoadModel(self, model):
    model_name = self.MODELS[model]["MODEL_NAME"]
    model_url = self.MODELS[model]["MODEL_URL"]
    model_file = self.MODELS[model]["MODEL_FILE"]
    model_graph = self.MODELS[model]["DETECTION_GRAPH"]
    if "CLASSES_FILE" in self.MODELS[model].keys():
      classes_file = os.path.join(self._classes,self.MODELS[model]["CLASSES_FILE"])
    else:
      classes_file = ""
    graph_file = os.path.join(self._repo,model_name,model_graph)
    self._LoadModelMaybe(model_name = model_name, 
                         model_file = model_file, 
                         download_url = model_url,
                         frozen_file = model_graph)
    return graph_file, classes_file, model_name
  
  
  
  def LoadConfig(self):
    cfg_file = open(self.config_file)
    self.CONFIG = json.load(cfg_file)
    if not ("BASE_FOLDER" in self.CONFIG.keys()):
      raise Exception("Invalid config file. BASE_FOLDER not found")
    else:
      self._base = self.CONFIG["BASE_FOLDER"]    
      if not ("APP_FOLDER" in self.CONFIG.keys()):
        raise Exception("Invalid config file. APP_FOLDER not found")
      self._app_folder = self.CONFIG["APP_FOLDER"]    
      if self._base.upper() in ["GOOGLE DRIVE", "GOOGLEDRIVE"]:
        home_dir = os.path.expanduser("~")
        valid_paths = [
                       os.path.join(home_dir, "Google Drive"),
                       os.path.join(home_dir, "GoogleDrive"),
                       os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                       os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                       os.path.join("C:/", "GoogleDrive"),
                       os.path.join("C:/", "Google Drive"),
                       os.path.join("D:/", "GoogleDrive"),
                       os.path.join("D:/", "Google Drive"),
                       ]      
        drive_path = None
        for path in valid_paths:
          if os.path.isdir(path):
            drive_path = path
            break      
        if drive_path is None:
          raise Exception("Couldn't find google drive folder!")
        if self._app_folder[0] in ['/','\\']:
          self._app_folder = self._app_folder[1:]
        self._base = os.path.join(drive_path, self._app_folder)
      else:
        self._base = os.path.join(self._base, self._app_folder)
    return
  
  
  
  def SetupRepository(self, rep_folder = '_models_repository', classes_folder = '_classes'):
    self._repo = os.path.join(self._base, rep_folder)
    self._classes = os.path.join(self._base, classes_folder)
    if not os.path.isdir(self._repo):
      self.log.VerboseLog("Creating folder [{}]".format(self._repo))
      os.makedirs(self._repo)
    if not os.path.isdir(self._classes):
      self.log.VerboseLog("Creating folder [{}]".format(self._classes))
      os.makedirs(self._classes)
    return
    
  
  
  def _LoadModelMaybe(self, model_name, model_file, download_url, 
                     frozen_file = 'frozen_inference_graph.pb'):
    log = self.log
    log.VerboseLog("Preparing frozen detection model [{}]...".format(model_name))
    archive_file = os.path.join(self._repo, model_file)
    frozen_model_file = os.path.join(self._repo,model_name,frozen_file)
    if not os.path.isfile(frozen_model_file):
      full_url = download_url + model_file
      log.VerboseLog("Downloading model from {}...".format(full_url))
      opener = urllib.request.URLopener()
      opener.retrieve(full_url, archive_file)
      log.VerboseLog("Done downloading {}.".format(archive_file))
      log.VerboseLog("Extracting ...")
      tar_file = tarfile.open(archive_file)
      for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if frozen_file in file_name:
          tar_file.extract(file, self._repo)    
      log.VerboseLog("Done extraction.")
    else:
      log.VerboseLog("Model allready downloaded.")
    return
  
  def load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  
  
  
  
  
if __name__=='__main__':
  

  codtf = ObjectDetector()
  for model in codtf.MODELS:
    codtf.FullSingleSceneInference('c3.jpg', USE_MODEL=model)
    codtf.ShowLastDetectionClasses()
  codtf.ShowTimings()