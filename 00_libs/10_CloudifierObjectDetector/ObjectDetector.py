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

from cdo_label_engine import ObjectLabelEngine, _DATASETS



__version__   = "2.1.ODAPI v2"
__author__    = "Cloudifier"
__copyright__ = "(C) Cloudifier SRL"
__project__   = "Cloudifier"  
__credits__   = "Parts used from open-source project OmniDJ by Knowledge Investment Group"


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

_T_INPUT = 'image_tensor:0'
_T_CLASSES = 'detection_classes:0'
_T_BOXES = 'detection_boxes:0'
_T_SCORES = 'detection_scores:0'
_T_N_DET = 'num_detections:0'

      
class ObjectDetector:
  def __init__(self, config_file = 'config.txt', DEBUG = True,
               no_default = False):
    self.DEBUG = DEBUG
    self.detection_threshold = 0.5
    self.IMAGE_SIZE = (12,8)
    log_module = load_module("logger", "logger.py")    
    self.logger = log_module.Logger(lib_name = "CODv2",
                                    config_file = config_file)
    self.log = self.logger

    self.LoadConfig()
    
    self.default_tagger = ObjectLabelEngine(logger = self.logger)
    
    self.SetupRepository()
    
    self.LoadModelsConfig()
    
    self.SetupAllModels()
    
    (self.DEFAULT_DETECTOR_FILE, 
     self.CLASSES_FILE, 
     self.CLASSES_MAP,
     self.DEFAULT_MODEL_NAME,
     self.MODEL_TYPE) = self.DownloadModel(self.DEFAULT_MODEL)
      
    self.USE_OBJECT_DETECTION_API = True # false for YOLO or other...
    
    
    self.df_time = pd.DataFrame(columns = ["MODEL","IMAGE_SHAPE",
                                           "AVG_TIME", "AVG_ACC","NR_IDs", "PASSES"])
    
    if not no_default:
      self.SetActiveGraph(self.DEFAULT_DETECTOR_FILE, self.DEFAULT_MODEL_NAME)    
      self.log.VerboseLog("Default detector [{}] READY.".format(self.DEFAULT_MODEL_NAME))
    return
  
  def ReloadClasses(self):
    self.default_tagger.load_config(dataset = self.MODEL_TYPE,
                                   labels_file = self.CLASSES_FILE,
                                   labelmap_file = self.CLASSES_MAP, 
                                   )
    return
    
  
  
  def SetActiveGraph(self, detector_graph_file, model_name,
                     classes_file = None, map_file = None, 
                     model_type = None):
    self.logger.VerboseLog("Loading [{}]".format(model_name))
    self.LOADED_GRAPH = self.LoadDetectionGraph(detector_graph_file)
    self.LOADED_GRAPH_FILE = detector_graph_file
    self.LOADED_MODEL = model_name
    if classes_file is not None: self.CLASSES_FILE = classes_file
    if map_file is not None: self.CLASSES_MAP = map_file
    if model_type is not None: self.MODEL_TYPE = model_type
    self.ReloadClasses()
    return
  
  
  
  def SetupAllModels(self):
    for model in self.MODELS:
      self.DownloadModel(model)
    return
  
  
  def _log_model_time(self, model, shape, secs, acc, preds, n_passes):
    i = self.df_time.shape[0]
    s_shape = "{}".format(shape)
    self.df_time.loc[i] = [model, s_shape, secs, acc, preds, n_passes]
    return
  
  def ShowTimings(self, by_value = 'AVG_TIME'):
    old = pd.options.display.width
    pd.options.display.width =  1000
    self.log.VerboseLog("Results:\n{}".format(self.df_time.sort_values(by = by_value)))
    pd.options.display.width =  old
    return
  
  
  def tf_AnalizeSingleImage(self, image_file, n_passes = 1):
    """
     n_passes:  how many inference runs for a given image for speed testing purposes
                works only in mode DEBUG = True
    """
    l = self.log
    if self.DEBUG and n_passes==1:
      n_passes = 3
    else:
      n_passes = 1
    #with self.LOADED_GRAPH.as_default():
    
    #DO NOT CREATE NEW SESSION FOR EACH INFERENCE
    
    with tf.Session(graph=self.LOADED_GRAPH) as sess:
      image = Image.open(image_file)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = self.load_image_into_numpy_array(image)
      img_h = image_np.shape[0]
      img_w = image_np.shape[1]
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = self.LOADED_GRAPH.get_tensor_by_name(_T_INPUT)
      # Each box represents a part of the image where a particular object was detected.
      boxes = self.LOADED_GRAPH.get_tensor_by_name(_T_BOXES)
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = self.LOADED_GRAPH.get_tensor_by_name(_T_SCORES)
      classes = self.LOADED_GRAPH.get_tensor_by_name(_T_CLASSES)
      num_detections = self.LOADED_GRAPH.get_tensor_by_name(_T_N_DET)
      timings = []
      # Actual detection.
      for n_pass in range(n_passes):
        l.VerboseLog("  Running inference pass {}...".format(n_pass+1))
        self.logger.start_timer(self.LOADED_MODEL)
        res_list = sess.run([boxes, scores, classes, num_detections],
                            feed_dict={ image_tensor : image_np_expanded})
        (np_boxes, np_scores, np_classes, np_num_detections) = res_list
        total_time = self.logger.end_timer(self.LOADED_MODEL)
        timings.append(total_time)
        l.VerboseLog("   [{}] Done inference in {:.2f}sec. found {} dets".format(
            self.LOADED_MODEL,
           total_time, np_num_detections))
    
    if self.DEBUG:
      self.log.show_timings()

    avg_time = self.logger.get_timing(self.LOADED_MODEL) #np.mean(timings)        
    nr_detections = 0
    result_boxes = []
    result_scores = []
    result_classes = []
    accuracy_list = []
    self.LAST_DETECTION_CLASSES = []
    for i in range(np_boxes.shape[1]):
      score = np_scores[0,i]
      if score >= self.detection_threshold:
        accuracy_list.append(score)
        coords = np_boxes[0,i,:]
        ymin = coords[0] * img_h
        xmin = coords[1] * img_w
        ymax = coords[2] * img_h
        xmax = coords[3] * img_w   
        det_class = np_classes[0,i]
        nr_detections += 1
        result_boxes.append((xmin,ymin,xmax,ymax))
        result_scores.append(score)
        result_classes.append(det_class)
        self.LAST_DETECTION_CLASSES.append(self.default_tagger.GetLabel(int(det_class)))
    avg_acc = 0
    if len(accuracy_list)>0:
      avg_acc = np.mean(accuracy_list)
    self._log_model_time(self.LOADED_MODEL, (img_w,img_h), 
                         avg_time, avg_acc, len(accuracy_list),
                         n_passes)
    return result_boxes,result_scores,result_classes
  
  
  
  def AnalyzeSingleImage(self, image_file):   
    if self.USE_OBJECT_DETECTION_API:
      result_boxes,result_scores,result_classes = self.tf_AnalizeSingleImage(image_file)
    return result_boxes,result_scores,result_classes
  



  def AnalyzeImages(self, image_file_list):   
    return

  
  
  def ShowLastDetectionClasses(self):
    self.log.VerboseLog("[{}] detected {} classes: {}".format(self.LOADED_MODEL,
                        len(self.LAST_DETECTION_CLASSES), self.LAST_DETECTION_CLASSES))
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
      if self.default_tagger != None:
        ax.text(xmin, ymax, 'CLS:{} ({:.1f}%)'.format(self.default_tagger.GetLabel(int(cls)),score*100)
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
      (detector_file, classes_file, map_file, 
       model_name, model_type) = self.DownloadModel(USE_MODEL)
      self.SetActiveGraph(detector_graph_file =  detector_file,
                          model_name = model_name,
                          classes_file = classes_file,
                          map_file = map_file,
                          model_type = model_type
                          )
      
    b,s,c = self.AnalyzeSingleImage(image_file = image_file)
    self.DrawImage(image_file = image_file, boxes = b,
                   scores = s, classes = c
                   )
    return  
    
  
  
  
  def LoadDetectionGraph(self, graph_file):
    l = self.log    
    t =""
    if graph_file == self.DEFAULT_DETECTOR_FILE:
      t = "DEFAULT"
    l.VerboseLog("Loading {} graph [...{}]...".format(t,graph_file[-25:]))
    start_time = time.time()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(graph_file, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')    
    end_time = time.time()
    l.VerboseLog(" Done preparing graph in {:.2f}s.".format(end_time-start_time))        
    return detection_graph
  
  
  
  
  def LoadModelsConfig(self):
    self.MODELS = self.CONFIG["MODELS"]
    self.DEFAULT_MODEL = self.CONFIG["DEFAULT_MODEL"]
    self.log.VerboseLog("Default model: {}".format(self.MODELS[self.DEFAULT_MODEL]["MODEL_NAME"]))
    return
  
  
  
  def DownloadModel(self, model):
    

    model_name = self.MODELS[model]["MODEL_NAME"]
    model_url = self.MODELS[model]["MODEL_URL"]
    model_file = self.MODELS[model]["MODEL_FILE"]
    model_graph = self.MODELS[model]["DETECTION_GRAPH"]
    classes_file = self.MODELS[model]["CLASSES_FILE"]
    
    if "MAP_FILE" in self.MODELS[model].keys():
      map_file = self.MODELS[model]["MAP_FILE"]
    else:
      map_file = None
      
    if "MODEL_TYPE" in self.MODELS[model].keys():
      model_type = self.MODELS[model]["MODEL_TYPE"]
    else:
      model_type = "COCO"
      
    model_base_name, ext = os.path.splitext(os.path.splitext(model_file)[0])
    graph_file = os.path.join(self._repo,model_base_name,model_graph)
    self._DownloadModelMaybe(model_name = model_name, 
                             model_file = model_file, 
                             download_url = model_url,
                             frozen_file = model_graph)
    return graph_file, classes_file, map_file, model_name, model_type
  
    
  def LoadConfig(self):
    self.CONFIG = self.logger.config_data
    self._app_folder = self.logger.GetBaseFolder()
    return
  
    
  def SetupRepository(self, rep_folder = '_models_repository', classes_folder = '_classes'):
    self._repo = self.logger.CheckFolder(rep_folder)
    self._classes = self.logger.CheckFolder(classes_folder)
    return
    
  
  
  def _DownloadModelMaybe(self, model_name, model_file, download_url, 
                          frozen_file = 'frozen_inference_graph.pb'):
    log = self.log
    log.VerboseLog("Prep-ing  [{}]...".format(model_name))
    model_base_name, ext = os.path.splitext(os.path.splitext(model_file)[0])
    archive_file = os.path.join(self._repo, model_file)
    frozen_model_file = os.path.join(self._repo,model_base_name,frozen_file)
    if not os.path.isfile(frozen_model_file):
      full_url = download_url + model_file
      log.VerboseLog(" Downloading  from {}...".format(full_url[:25]))
      opener = urllib.request.URLopener()
      opener.retrieve(full_url, archive_file)
      log.VerboseLog(" Done download ...{}.".format(archive_file[-30:]))
      log.VerboseLog(" Extracting ...")
      tar_file = tarfile.open(archive_file)
      for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if frozen_file in file_name:
          log.VerboseLog("   Extracting {}".format(file_name))
          tar_file.extract(file, self._repo)    
      log.VerboseLog(" Done extraction.")
    else:
      log.VerboseLog(" Model allready downloaded.")
    return
  
  def load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  
  
  
  
  
if __name__=='__main__':
  
  model_types_to_test = ['OID','COCO']

  codtf = ObjectDetector(no_default = True)
  for model in codtf.MODELS:
#    if "MODEL_TYPE" in codtf.MODELS[model].keys():
#      model_type = codtf.MODELS[model]["MODEL_TYPE"]
#      if model_type in model_types_to_test:
        codtf.FullSingleSceneInference('test6.jpg', USE_MODEL=model)
        codtf.ShowLastDetectionClasses()
  codtf.ShowTimings()