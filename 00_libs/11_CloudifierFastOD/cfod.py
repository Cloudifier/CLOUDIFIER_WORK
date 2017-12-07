# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 07:07:50 2017

history:

"""

import os

from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np

import PIL
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

from scipy.misc import imresize




from omni_camera_utils import VideoCameraStream, np_rect, np_circle
from omni_face_eng import FaceEngine, is_shape, FacialLandmarks


__version__   = "1.1.yolo_dlib"
__author__    = "Cloudifier"
__copyright__ = "(C) Cloudifier SRL"
__project__   = "Cloudifier"  
__credits__   = "Parts used from open-source project OmniDJ by Knowledge Investment Group"

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




class FastObjectDetector:
  def __init__(self, config_file = "config.txt", max_boxes=10, score_threshold=.6, iou_threshold=.5,
               use_PIL = False, PERSON_CLASS = 0, add_new_session = False, session = None):

    self.DEBUG = True
    self.__version__ = __version__
    
    self.use_PIL = use_PIL
    self.image_shape = None
    module = load_module("logger", "logger.py")
    self.logger = module.Logger(lib_name = "CFOD", lib_ver = self.__version__,
                                config_file = config_file,
                                DEBUG = self.DEBUG,
                                HTML = True)
    self.config_data = self.logger.config_data
    
    self.logger.VerboseLog("Init session...")
    if session != None:
      self.sess = session
      K.set_session(self.sess)
    else:
      if add_new_session:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)
      else:
        K.clear_session()
        self.sess = K.get_session()
    self.logger.VerboseLog("Done init session.", show_time = True)
    
    
    self.PERSON_CLASS = PERSON_CLASS

    
    classes_file = os.path.join(self.logger._data_dir, self.config_data["CLASSES"])
    anchors_file = os.path.join(self.logger._data_dir, self.config_data["ANCHORS"])
    self.logger.VerboseLog("Load classes: {}".format(classes_file))
    self.class_names = read_classes(classes_file)
    self.logger.VerboseLog("Load anchors: {}".format(anchors_file))
    self.anchors = read_anchors(anchors_file)
    
    
    self.shape_large_clf = None
    self.shape_small_clf = None
    if "DLIB_FACE_MODEL" in self.config_data.keys():
      lpredictor_file = os.path.join(self.logger._data_dir, self.config_data["DLIB_FACE_MODEL"])

    if "DLIB_FACE_MODEL_SMALL" in self.config_data.keys():
      spredictor_file = os.path.join(self.logger._data_dir, self.config_data["DLIB_FACE_MODEL_SMALL"])
      
    if "DLIB_FACE_NET" in self.config_data.keys():
      net_model_file = os.path.join(self.logger._data_dir, self.config_data["DLIB_FACE_NET"])

    self.logger.VerboseLog("Loading face detector engine ...")
    self.face_engine = FaceEngine(path_small_shape_model = spredictor_file, 
                                  path_large_shape_model = lpredictor_file, 
                                  path_faceid_model = net_model_file,
                                  logger = self.logger,
                                  DEBUG = self.DEBUG,
                                  score_threshold = 0.619)
    self.logger.VerboseLog("Done loading face detector engine.")
   

    model_file = os.path.join(self.logger._data_dir, self.config_data["MODEL"])
    self.logger.VerboseLog("Loading model [{}]...".format(model_file))
    self.yolo_model = load_model(model_file)
    self.learning_phase = K.learning_phase()
    self.logger.VerboseLog("Done loading model.", show_time = True)
    self.logger.VerboseLog(self.logger.GetKerasModelDesc(self.yolo_model))
    self.logger.Log(self.logger.GetKerasModelDesc(self.yolo_model))


    

    
    self.max_boxes = max_boxes
    self.score_threshold = score_threshold
    self.iou_threshold = iou_threshold
    
    self.prepared = False
    self.model_shape = (self.config_data["MODEL_SIZE"],self.config_data["MODEL_SIZE"])
    
    self.logger.VerboseLog("Done engine init.")
    
    return
    
  def _DEBUG_INFO(self, skey, img = None):
    if self.DEBUG:
      self.logger.VerboseLog("Received callback command [{}]".format(skey))
    if skey == "I":
      pers_id = 10
      df = self.face_engine.get_id_vs_all(pers_id)
      self.logger.VerboseLog("DEBUG FACE {} VS ALL:\n{}".format(pers_id,df))
    elif skey == "S":
      self.logger.OutputImage(img)
    return
  
  def npimg_preprocess(self, np_img, new_size):
    img = imresize(np_img,new_size).astype('float32')
    img /= 255
    return np.expand_dims(img, 0)

  def prepare(self, image_shape = (720., 1280.)):
    """
    final preparation of computation graph
    """
    image_shape = (float(image_shape[0]),float(image_shape[1]))
    self.image_shape = image_shape
    self.yolo_outputs = yolo_head(self.yolo_model.output, 
                                  self.anchors, 
                                  len(self.class_names))    
    self.scores, self.boxes, self.classes = self.y9k_eval(yolo_outputs = self.yolo_outputs, 
                                                          image_shape = self.image_shape,
                                                          max_boxes = self.max_boxes, 
                                                          score_threshold = self.score_threshold, 
                                                          iou_threshold = self.iou_threshold)
    self.prepared = True
    return self.prepared
  
  def _person_callback(self, box, np_img):
    """
    call this function when class is self.PERSON_CLASS 
    """
    top,left,bottom,right = box
    left = max(0, int(left))
    top = max(0, int(top))
    right = min(np_img.shape[1]-1, int(right))
    bottom = min(np_img.shape[0]-1, int(bottom))
    np_pers_img = np_img[top:bottom,left:right,:].copy() # .copy() bug workaround
    
    # 1st of all detect person face if available !
    
    
    DRAW_FACE_RECT = True
    FACE_THUMBSIZE = 75
    FACE_LANDMARKS = True # 0: no, 1: full 2: only 5 feats
    DISPLAY_FACE = True
    FACE_ID = True
    
    self.logger.start_timer(" FaceGetInfo")
    res = self.face_engine.GetFaceInfo(np_pers_img, get_shape = FACE_LANDMARKS,
                                       get_id_name = FACE_ID)
    self.logger.end_timer(" FaceGetInfo")
    found_box, np_facial_shape, _, fid, sname = res
    
    
    FACE_TOP_OFFSET = -30
    FACE_BOTTOM_OFFSET = 10
    FACE_LEFT_OFFSET = -10
    FACE_RIGHT_OFFSET = 10
  
    
    if found_box != None:
      person_id = sname+" [{}]".format(fid)
      L,T,R,B = found_box
      T += FACE_TOP_OFFSET
      L += FACE_LEFT_OFFSET
      R += FACE_RIGHT_OFFSET
      B += FACE_BOTTOM_OFFSET
      np_exact_face = np_pers_img[T:B,L:R,:]
      face_left = left + L
      face_top = top + T
      face_right = left + R
      face_bottom = top + B
      
      ###
      ### now do extra processing 
      ###
      self.logger.start_timer(" FaceDrawingTime")
      if DISPLAY_FACE:
        if (np_exact_face.shape[0]>FACE_THUMBSIZE) and (np_exact_face.shape[1]>FACE_THUMBSIZE):
          np_resized = scipy.misc.imresize(np_exact_face, (FACE_THUMBSIZE,FACE_THUMBSIZE))
          np_img[:FACE_THUMBSIZE,-FACE_THUMBSIZE:,:] = np_resized
      
      if FACE_LANDMARKS:
        for i in range(np_facial_shape.shape[0]):
          x = int(np_facial_shape[i,0]) + left
          y = int(np_facial_shape[i,1]) + top
          clr = (0,0,255)
          if is_shape(FacialLandmarks.FL_LEYE,i) :
            clr = (255,255,255)
          if is_shape(FacialLandmarks.FL_REYE,i) :
            clr = (0,255,255)
          np_img = np_circle(np_img, (x, y), 1, clr, -1)

      if DRAW_FACE_RECT:
        np_img = np_rect(face_left, face_top, face_right, face_bottom, np_img,
                         color = (0,255,0), text = person_id)
      self.logger.end_timer(" FaceDrawingTime")
      ### 
      ### done extra processing
      ###
    
    return np_img
  
  def process_boxes(self, scores, boxes, classes, np_img, show_label = False):
    
    for i,box in enumerate(boxes):
      cls = self.class_names[classes[i]]
      score = scores[i]
      top, left, bottom, right = box
      clr = (255,255,255)
      if classes[i] == self.PERSON_CLASS:
        clr = (0,0,255) # red
        np_img = self._person_callback(box,np_img)
      label = None
      if show_label:
        label = "{} [{:.1f}%]".format(cls, score * 100)
      np_img = np_rect(left,top,right,bottom,np_img, 
                       color = clr, text = label,
                       use_cv2 = False, thickness = 2)    # cv2 = True for fast drawing
    return np_img

  def predict_img(self, img):
    """
     accepts a single image in HWC format - image can be any size
     
     returns touple of 4 items:
       out_scores: scores for each box
       out_boxes: each box
       out_classes: box classes
       np_image: numpy HWC image
      
    """
    _result = None
    if self.prepared:
      self.logger.start_timer("Predict")
      image_data = self.npimg_preprocess(img, self.model_shape)
      # Run the session 
      # model uses BatchNorm thus need to pass {K.learning_phase(): 0}
      self.logger.start_timer("YOLO Inference")
      out_scores, out_boxes, out_classes = self.sess.run(
          [self.scores, self.boxes, self.classes], 
           feed_dict = {self.yolo_model.input:image_data,
                        self.learning_phase:0})
      _DEBUG_t_inference = self.logger.end_timer("YOLO Inference")
      # Print predictions info
      if self.DEBUG:
        self.logger.VerboseLog('Found {} boxes for current frame: {} '.format(
            len(out_boxes), out_classes, _DEBUG_t_inference))
      
      if self.use_PIL:
        self.logger.start_timer("PIL_YOLO_Draw")
        # Generate colors for drawing bounding boxes.
        colors = generate_colors(self.class_names)
        # Draw bounding boxes on the image file
        image = PIL.Image.fromarray(img)
        draw_boxes(image, out_scores, out_boxes, out_classes, self.class_names, colors,
                   logger = self.logger)      
        _result =  np.array(image)
        self.logger.end_timer("PIL_YOLO_Draw")
      else:
        self.logger.start_timer("ProcessBoxes")
        img = self.process_boxes(out_scores, out_boxes, out_classes, 
                                 np_img = img, show_label = True)
        _DEBUG_t_process = self.logger.end_timer("ProcessBoxes")
        _result = img


      _DEBUG_t_total = self.logger.end_timer("Predict")        
      if self.DEBUG:
        self.logger.VerboseLog(" Total frame time {:.3f}s inference: {:.3f}s process: {:.3f}s".format(
            _DEBUG_t_total, _DEBUG_t_inference, _DEBUG_t_process))
      
    else:
      self.logger.VerboseLog("ERROR: Predict called without preparation")
      
    return _result
    
  def predict_file(self, image_file):
    """
     accepts a single image file
    """    
    _result = None
    if self.prepared:
      self.logger.VerboseLog("Predicting ...")
      # Preprocess your image
      image, image_data = preprocess_image(image_file, model_image_size = self.model_shape)
  
      # Run the session 
      # model uses BatchNorm thus need to pass {K.learning_phase(): 0}
      out_scores, out_boxes, out_classes = self.sess.run([self.scores, 
                                                          self.boxes, 
                                                          self.classes], 
                                                         feed_dict = {self.yolo_model.input:image_data,
                                                                      K.learning_phase():0})
 
      # Print predictions info
      self.logger.VerboseLog('Found {} boxes for {}'.format(len(out_boxes), image_file), show_time = True)
      # Generate colors for drawing bounding boxes.
      colors = generate_colors(self.class_names)
      # Draw bounding boxes on the image file
      draw_boxes(image, out_scores, out_boxes, out_classes, self.class_names, colors, 
                 logger = self.logger)
      # Save the predicted bounding box on the image
      image.save(os.path.join(self.logger._outp_dir, image_file), quality=90)
      # Display the results 
      output_image = scipy.misc.imread(os.path.join(self.logger._outp_dir, image_file))
      imshow(output_image)
      self.logger.VerboseLog('Done preparing output image', show_time = True)
      
      _result =  (out_scores, out_boxes, out_classes)
      
    else:
      self.logger.VerboseLog("ERROR: Predict called without preparation")
      
    return _result

  def y9k_filter_boxes(self, box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    
    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs
    
    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis = -1)
    box_class_scores = K.max(box_scores, axis = -1)
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = box_class_scores >= threshold
    
    # Step 4: Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)
    
    return scores, boxes, classes
    
  def y9k_non_max_suppression(self, scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have 
             been scaled to the image size 
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. 
    Note also that this function will transpose the shapes of scores, boxes, classes. 
    This is made for convenience.
    """
    # max nr of classes param tensor to be used in tf.image.non_max_suppression()    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32', name='max_box_tensor')     
    # next initialize variable max_boxes_tensor
    #self.logger.VerboseLog("{} == {} is {}".format(self.sess,K.get_session(),
    #                       self.sess == K.get_session()))
    self.logger.VerboseLog("{} == {} is {}".format(self.sess,K.get_session(),
                           self.sess == K.get_session()))
    self.sess.run(tf.variables_initializer([max_boxes_tensor])) 
    
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes,scores, 
                                               max_boxes_tensor,
                                               iou_threshold = iou_threshold)
    
    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores  = K.gather(scores, nms_indices)
    boxes   = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes  
  
  def y9k_eval(self, yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes 
    along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), 
    contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook 
                   we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], 
                    then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
       
    # Retrieve outputs of the YOLO model (≈1 line)
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
  
    # Convert boxes to be ready for filtering functions 
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
  
    # perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = self.y9k_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape) #, self.K)
  
    # perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes =  self.y9k_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, iou_threshold = iou_threshold)
    
    return scores, boxes, classes  
  
  def show_fr_stats(self):
    self.logger.VerboseLog("FR STATS:\n{}".format(self.face_engine.get_stats()))
    self.logger.show_timers()
    ## maximum across all sessions and .run calls so far
    #max_bytes = self.sess.run(tf.contrib.memory_stats.MaxBytesInUse())
    ## current usage
    #bytes_use = self.sess.run(tf.contrib.memory_stats.BytesInUse())
    #self.logger.VerboseLog("MaxBytesInUse: {}".format(max_bytes))
    #self.logger.VerboseLog("BytesInUse: {}".format(bytes_use))
    return
  
  
  def shutdown(self):
    self.logger.VerboseLog("Shutdown.")
    #del self.yolo_model
    #self.sess.close()
    #del self.sess
    #del self.face_engine
    return
  

if __name__ == '__main__':

  cfod = FastObjectDetector(score_threshold = 0.5)
  vstrm = VideoCameraStream(logger = cfod.logger)  
  if vstrm.video != None:
    video_frame_shape = (vstrm.H,vstrm.W) 
    cfod.prepare(image_shape = video_frame_shape)
    vstrm.play(process_func = cfod.predict_img, info_func = cfod._DEBUG_INFO)
    vstrm.shutdown()
    if cfod.DEBUG:
      cfod.show_fr_stats()
    cfod.shutdown()
