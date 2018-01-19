# -*- coding: utf-8 -*-
"""
History:
  
  2017-10-01 First version based on YOLO v1
  2017-11-15 Second version based on YOLO2K
  2017-11-30 Added OmniFR features 
  2017-12-10 Added support for production implementation
  2017-12-11 Finalized production grade computational graph 
  2017-12-18 Added support for TF ODAPI graphs

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


__version__   = "1.3.ytf_fr2"
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
  def __init__(self, config_file = "config.txt", 
               max_boxes=10, 
               score_threshold=.6, 
               iou_threshold=.5,
               use_PIL = False, 
               PERSON_CLASS = 0, add_new_session = False, session = None):

    self.DEBUG = True
    self.prepared = False
    self.__version__ = __version__
    
    self.use_PIL = use_PIL
    self.image_shape = None
    self.current_session = None
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
    self.logger.VerboseLog("Load classes: ...{}".format(classes_file[-30:]))
    self.class_names = read_classes(classes_file)
    self.logger.VerboseLog("Load anchors: ...{}".format(anchors_file[-30:]))
    self.anchors = read_anchors(anchors_file)
    
    fr_method = self.config_data["FR_METHOD"]
    fr_result = self.config_data["FR_OUTPUT_FILE"]
    fr_out_file = fr_method + "_" +  fr_result
    
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
                                  fr_method = fr_method,
                                  logger = self.logger,
                                  DEBUG = self.DEBUG,
                                  score_threshold = 0.619,
                                  output_file = fr_out_file)
    self.logger.VerboseLog("Done loading face detector engine.")
    
    self.used_method = self.config_data["USED_METHOD"] 
    self.valid_methods = self.config_data["ALL_METHODS"]
    assert self.used_method in self.valid_methods, "Configuratio issue: {} not in methods {}".format(
        self.used_method, self.valid_methods)
   
    self.log("CFOD used method: {}".format(self.used_method))

    if self.used_method=='YOLO':
      self._load_yolo()
    elif self.used_method=='TFODAPI':
      self._load_tfodapi()
      
    
    self.max_boxes = max_boxes
    self.score_threshold = score_threshold
    self.iou_threshold = iou_threshold
    
    
    self.logger.VerboseLog("Done engine init.")
    
    return
  
  
  def _load_tfodapi(self):
    self.pb_file = os.path.join(self.logger._data_dir, self.config_data["TFODAPI_MODEL"])
    assert os.path.isfile(self.pb_file), "Model graph file {} not found".format(self.pb_file)
    self.classes_tensor_name = self.config_data["TFODAPI_CLASSES_TENSOR_NAME"]
    self.scores_tensor_name = self.config_data["TFODAPI_SCORES_TENSOR_NAME"]
    self.boxes_tensor_name = self.config_data["TFODAPI_BOXES_TENSOR_NAME"]
    self.input_tensor_name = self.config_data["TFODAPI_INPUT_TENSOR_NAME"]
    self.numdet_tensor_name = self.config_data["TFODAPI_NUMDET_TENSOR_NAME"]
    self.tfodapi_graph = self.logger.LoadTFGraph(self.pb_file)
    self.tfodapi_sess = tf.Session(graph = self.tfodapi_graph)
    self.tf_classes = self.tfodapi_sess.graph.get_tensor_by_name(
                            self.classes_tensor_name+":0")
    self.tf_scores = self.tfodapi_sess.graph.get_tensor_by_name(
                            self.scores_tensor_name+":0")
    self.tf_boxes= self.tfodapi_sess.graph.get_tensor_by_name(
                            self.boxes_tensor_name+":0")
    self.tf_numdet= self.tfodapi_sess.graph.get_tensor_by_name(
                            self.numdet_tensor_name+":0")
    self.tf_odapi_input = self.tfodapi_sess.graph.get_tensor_by_name(
                            self.input_tensor_name+":0")
    self.prepared = True      
    self.current_session = self.tfodapi_sess
    self.log("Done loading TFGRaph")
    
    return
  
  
  def _load_yolo(self):
    model_file = os.path.join(self.logger._data_dir, self.config_data["YOLO_MODEL"])
    self.logger.VerboseLog("Loading scene inference model [{}]...".format(model_file[-30:]))
    
    self.classes_tensor_name = self.config_data["YOLO_CLASSES_TENSOR_NAME"]
    self.scores_tensor_name = self.config_data["YOLO_SCORES_TENSOR_NAME"]
    self.boxes_tensor_name = self.config_data["YOLO_BOXES_TENSOR_NAME"]
    self.input_tensor_name = self.config_data["YOLO_INPUT_TENSOR_NAME"]
    
    filename, file_extension = os.path.splitext(model_file)
    self.pb_file = filename+'.pb'   
    #FIRST TRY TO LOAD .pb GRAPH THEN TURN TO MODEL
    if os.path.isfile(self.pb_file):
      self.log("TFGraph found. Loading [...{}]".format(self.pb_file[-30:]))
      self.yolo_graph = self.logger.LoadTFGraph(self.pb_file)
      self.yolo_sess = tf.Session(graph = self.yolo_graph)
      
      self.learning_phase = self.yolo_sess.graph.get_tensor_by_name("keras_learning_phase:0")
      self.tf_classes = self.yolo_sess.graph.get_tensor_by_name(
                              self.classes_tensor_name+":0")
      self.tf_scores = self.yolo_sess.graph.get_tensor_by_name(
                              self.scores_tensor_name+":0")
      self.tf_boxes= self.yolo_sess.graph.get_tensor_by_name(
                              self.boxes_tensor_name+":0")
      self.tf_yolo_input = self.yolo_sess.graph.get_tensor_by_name(
                              self.input_tensor_name+":0")
      ## graph_h, graph_w not used at inference but kept nevertheless
      self.graph_out_height = self.config_data["GRAPH_H"]
      self.graph_out_width = self.config_data["GRAPH_W"]
      ##
      #IF .pb LOADED THEN CONSIDER MODEL PREPARED !
      self.prepared = True      
      self.current_session = self.yolo_sess
      self.log("Done loading TFGRaph")
    else:  
      self.yolo_model = load_model(model_file)
      #self.learning_phase = self.sess.graph.get_tensor_by_name("keras_learning_phase:0")
      #K.learning_phase()
      self.logger.VerboseLog("Done loading model.", show_time = True)
      self.logger.VerboseLog(self.logger.GetKerasModelDesc(self.yolo_model))
      self.logger.Log(self.logger.GetKerasModelDesc(self.yolo_model))
      
    self.model_shape = (self.config_data["YOLO_MODEL_SIZE"],self.config_data["YOLO_MODEL_SIZE"])      
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

  
  def _person_callback(self, box, np_img):
    """
    callback function used when class is self.PERSON_CLASS 
    """
    top,left,bottom,right = box
    left = max(0, int(left))
    top = max(0, int(top))
    right = min(np_img.shape[1]-1, int(right))
    bottom = min(np_img.shape[0]-1, int(bottom))
    np_pers_img = np_img[top:bottom,left:right,:].copy() # .copy() bug workaround
    
    # 1st of all detect person face if available !
    
    
    DRAW_FACE_RECT = True
    FACE_LANDMARKS = True # 0: no, 1: full 2: only 5 feats
    DISPLAY_FACE = True
    FACE_ID = True
    
    self.logger.start_timer(" FaceGetInfo")
    res = self.face_engine.GetFaceInfo(np_pers_img, get_shape = FACE_LANDMARKS,
                                       get_id_name = FACE_ID,
                                       get_thumb = True)
    self.logger.end_timer(" FaceGetInfo")
    found_box, np_facial_shape, _, fid, sname, np_resized, dist = res
    
    
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
      if self.DEBUG:
        self.last_face = np_exact_face
        self.last_person = np_pers_img
        self.last_align = np_resized
      face_left = left + L
      face_top = top + T
      face_right = left + R
      face_bottom = top + B
      
      ###
      ### now do extra processing 
      ###
      self.logger.start_timer(" FaceDrawingTime")
      if DISPLAY_FACE:
        #if ((np_exact_face.shape[0] > FACE_THUMBSIZE) and 
        #    (np_exact_face.shape[1] > FACE_THUMBSIZE)):   
        #  np_resized = scipy.misc.imresize(np_exact_face, (FACE_THUMBSIZE,FACE_THUMBSIZE))
        resized_h, resized_w = np_resized.shape[0:2]
        np_img[:resized_h, -resized_w:, :] = np_resized
      
      if FACE_LANDMARKS:
        np_img = self.face_engine.draw_facial_shape(np_img, np_facial_shape, 
                                                    left = left, top = top)

      if DRAW_FACE_RECT:
        txt = person_id + " D:{:.2f} S:{}".format(dist,
                             (face_bottom-face_top, face_right-face_left),)
        np_img = np_rect(face_left, face_top, face_right, face_bottom, np_img,
                         color = (0,255,0), text = txt)
      self.logger.end_timer(" FaceDrawingTime")
      ### 
      ### done extra processing
      ###
    return np_img
  
  def process_boxes(self, scores, boxes, classes, np_img, show_label = False):
    """
    process each box box within inferred scene
    """
    for i,box in enumerate(boxes):
      cls_idx = min(classes[i],len(self.class_names)-1)
      cls = self.class_names[cls_idx]
      score = scores[i]
      top, left, bottom, right = box
      H = int(bottom-top)
      W = int(right-left)
      clr = (255,255,255)
      
      if classes[i] == self.PERSON_CLASS:
        clr = (0,0,255) # red
        np_img = self._person_callback(box,np_img)
      label = None
      
      if show_label:
        label = "{} [{:.1f}%] {}".format(cls, score * 100, (H,W))
        
      np_img = np_rect(left,top,right,bottom,np_img, 
                       color = clr, text = label,
                       use_cv2 = False, thickness = 2)    # cv2 = True for fast drawing
    return np_img

  
  def predict_img(self, np_img):
    result = None
    if self.used_method == 'YOLO':
      result = self._yolo_predict_img(np_img)
    elif self.used_method == 'TFODAPI':
      result  = self._tfodapi_predict_img(np_img)
    return result


  def _tfodapi_predict_img(self, np_img):
    """
     TF ODAPI graph based inference
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
      
      assert len(np_img.shape) == 3, "Input image in tf_pred must pe HWC"

      np_img_input = np.expand_dims(np_img, axis = 0)
      
      img_h = np_img_input.shape[1]
      img_w = np_img_input.shape[2]
      # Run the session 
      # model uses BatchNorm thus need to pass {K.learning_phase(): 0}
      self.logger.start_timer("TFODAPI Inference")
      np_scores, np_boxes, np_classes, np_numdet = self.tfodapi_sess.run(
          [self.tf_scores, self.tf_boxes, self.tf_classes, self.tf_numdet], 
           feed_dict = {self.tf_odapi_input : np_img_input,
                        })
      _DEBUG_t_inference = self.logger.end_timer("TFODAPI Inference")
      
      self.logger.start_timer("ProcessBoxes")

      result_boxes = []
      result_scores = []
      result_classes = []
      for i in range(np_boxes.shape[1]):
        score = np_scores[0,i]
        if score >= self.score_threshold:
          coords = np_boxes[0,i,:]
          top = coords[0] * img_h
          left = coords[1] * img_w
          bottom = coords[2] * img_h
          right = coords[3] * img_w   
          det_class = np_classes[0,i] - 1 # zero-offset due to COCO dict...
          result_boxes.append((top,left,bottom,right))
          result_scores.append(score)
          result_classes.append(int(det_class))

      img = self.process_boxes(result_scores, result_boxes, result_classes, 
                               np_img = np_img, show_label = True)
      _DEBUG_t_process = self.logger.end_timer("ProcessBoxes")
      _result = img

      # Print predictions info
      if self.DEBUG:
        self.logger.VerboseLog('Found {} boxes for current frame: {} in {:.2f}s'.format(
            len(result_boxes), result_classes, _DEBUG_t_inference))

      _DEBUG_t_total = self.logger.end_timer("Predict")        
      if self.DEBUG:
        self.logger.VerboseLog(" Total frame time {:.3f}s inference: {:.3f}s process: {:.3f}s".format(
            _DEBUG_t_total, _DEBUG_t_inference, _DEBUG_t_process))
      
    else:
      self.logger.VerboseLog("ERROR: Predict called without preparation")
      
    return _result



  def _yolo_predict_img(self, img):
    """
     yolo graph based inference
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
      out_scores, out_boxes, out_classes = self.yolo_sess.run(
          [self.tf_scores, self.tf_boxes, self.tf_classes], 
           feed_dict = {self.tf_yolo_input : image_data,
                        self.learning_phase : 0})
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
    self.log("Preparing final Y2k evaluation graph for {} input".format(image_shape))
       
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
    
    scores = tf.identity(scores, name = self.scores_tensor_name)
    boxes = tf.identity(boxes, name = self.boxes_tensor_name)
    classes = tf.identity(classes, name = self.classes_tensor_name)
    
    return scores, boxes, classes  


  def prepare(self, image_shape = (720., 1280.)):
    """
    final preparation of computation graph 
    will save production grade full DAG as .pb file
    """
    if self.prepared: # return if model allready prepared (probabily loaded .pb)      
      return True
    self.log("Preparing outputs ...")
    image_shape = (float(image_shape[0]),float(image_shape[1]))
    self.image_shape = image_shape
    self.yolo_outputs = yolo_head(self.yolo_model.output, 
                                  self.anchors, 
                                  len(self.class_names))    
    self.log("Preparing evaluation ...")
    self.tf_scores, self.tf_boxes, self.tf_classes = self.y9k_eval(yolo_outputs = self.yolo_outputs, 
                                                          image_shape = self.image_shape,
                                                          max_boxes = self.max_boxes, 
                                                          score_threshold = self.score_threshold, 
                                                          iou_threshold = self.iou_threshold)
    self.tf_yolo_input = self.yolo_model.input
    self.input_tensor_name = self.tf_yolo_input.name
    self.log("Graph prepared. Saving .pb ...")
    self.yolo_sess = K.get_session()
    self.logger.SaveGraph(session = self.yolo_sess,
                          tensor_list = [self.tf_scores, 
                                         self.tf_boxes, 
                                         self.tf_classes],
                          pb_file = self.pb_file,
                          input_names = [self.input_tensor_name],)
                          #output_names = [self.scores_tensor_name,
                          #                self.boxes_tensor_name,
                          #                self.classes_tensor_name])   
    self.log("Done Saving .pb file.")
    self.prepared = True
    self.current_session = self.yolo_sess
    return self.prepared

  
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
    self.current_session.close()
    return
  
  def log(self, s, show_time = False):
    self.logger.VerboseLog(s, show_time = show_time,)
    return
  
  
  def on_click(self, x, y):
    return
  

if __name__ == '__main__':

  cfod = FastObjectDetector(score_threshold = 0.5)
  vstrm = VideoCameraStream(logger = cfod.logger,
                            process_func = cfod.predict_img, 
                            info_func = cfod._DEBUG_INFO,
                            onclick_func = cfod.on_click)
  if vstrm.video != None:
    video_frame_shape = (vstrm.H,vstrm.W) 
    cfod.prepare(image_shape = video_frame_shape)
    vstrm.play()
    vstrm.shutdown()
    if cfod.DEBUG:
      cfod.show_fr_stats()
    cfod.shutdown()
