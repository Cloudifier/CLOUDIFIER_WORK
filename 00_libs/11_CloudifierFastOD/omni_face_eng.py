# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 00:08:05 2017

History:
  

"""
from sklearn.metrics.pairwise import pairwise_distances
from collections import OrderedDict
import cv2
import dlib
import pandas as pd
import numpy as np
import os

__version__ = "0.1.dl20.1"
__author__  = "Andrei Ionut Damian"
__copyright__ = "(C) Knowledge Investment Group"
__project__ = "OmniDJ"

class FacialLandmarks:
  FL_MOUTH = "MOUTH"
  FL_REYEB = "RIGHT EYEBROW"
  FL_LEYEB = "LEFT EYEBROW"
  FL_REYE  = "RIGHT EYE"
  FL_LEYE  = "LEFT EYE"
  FL_NOSE  = "NOSE"
  FL_JAW   = "JAW"
  FL_SET = [FL_MOUTH, 
            FL_REYEB, FL_LEYEB,
            FL_REYE, FL_LEYE,
            FL_NOSE,
            FL_JAW]

FACIAL_LANDMARKS = OrderedDict([
	(FacialLandmarks.FL_MOUTH, (48, 68)),
	(FacialLandmarks.FL_REYEB, (17, 22)),
	(FacialLandmarks.FL_LEYEB, (22, 27)),
	(FacialLandmarks.FL_REYE, (36, 42)),
	(FacialLandmarks.FL_LEYE, (42, 48)),
	(FacialLandmarks.FL_NOSE, (27, 35)),
	(FacialLandmarks.FL_JAW, (0, 17))
])

def is_shape(name, nr):
  assert name in FacialLandmarks.FL_SET
  return (nr>=FACIAL_LANDMARKS[name][0]) and (nr<FACIAL_LANDMARKS[name][1])

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
  # create two copies of the input image -- one for the
  # overlay and one for the final output image
  overlay = image.copy()
  output = image.copy()
  
  # if the colors list is None, initialize it with a unique
  # color for each facial landmark region
  if colors is None:
    colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
              (168, 100, 168), (158, 163, 32),
              (163, 38, 32), (180, 42, 220)]
    # loop over the facial landmark regions individually
  for (i, name) in enumerate(FACIAL_LANDMARKS.keys()):
    # grab the (x, y)-coordinates associated with the
    # face landmark
    (j, k) = FACIAL_LANDMARKS[name]
    pts = shape[j:k]
    # check if are supposed to draw the jawline
    if name == "JAW":
      # since the jawline is a non-enclosed facial region,
      # just draw lines between the (x, y)-coordinates
      for l in range(1, len(pts)):
        ptA = tuple(pts[l - 1])
        ptB = tuple(pts[l])
        cv2.line(overlay, ptA, ptB, colors[i], 2)
 
    # otherwise, compute the convex hull of the facial
    # landmark coordinates points and display it
    else:
      hull = cv2.convexHull(pts)
      cv2.drawContours(overlay, [hull], -1, colors[i], -1)    
  # apply the transparent overlay
  cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
  
  # return the output image
  return output      


class FaceEngine:
  def __init__(self, path_small_shape_model, path_large_shape_model, path_faceid_model,
               logger,
               method = 'dlib', score_threshold = 0.6, 
               DEBUG = False):
    """
     loads DLib models for 5, 68 feature detection together with CNN model for 
       128 face embeddings
     need to pass Logger object
    """
    self.method = method
    self.score_threshold = score_threshold
    self.__version__ = __version__
    self.DEBUG = DEBUG
    self.logger = logger
    self.logger.VerboseLog("Initializing FaceEngine v.{}".format(self.__version__))
    self.config_data = self.logger.config_data
    self.shape_large_model = None
    self.shape_small_model = None
    self.faceid_model = None
    self.NR_EMBEDDINGS = 128
    self.ID_FIELD = "ID"
    self.NAME_FIELD = "NAME"
    
    self.data_file = os.path.join(self.logger._data_dir, "faces.csv")
    self.feats_names = []
    for i in range(self.NR_EMBEDDINGS):
      self.feats_names.append("F_{}".format(i+1))
      
    if os.path.isfile(self.data_file):
      self.df_faces = pd.read_csv(self.data_file, index_col = False)
      self.columns = list(self.df_faces.columns)
    else:
      self.columns = [self.ID_FIELD, self.NAME_FIELD]
      for i in range(128):
        self.columns.append(self.feats_names[i])        
      self.df_faces = pd.DataFrame(columns = self.columns)
    
    if path_large_shape_model != None:
      self.logger.VerboseLog("Loading dlib LARGE 68 shape predictor model [{}]...".format(
          path_large_shape_model))
      self._shape_large_model = dlib.shape_predictor(path_large_shape_model)
      self.logger.VerboseLog("Done loading dlib LARGE 68 shape predictor model", show_time = True)
      
    if path_small_shape_model != None:
      self.logger.VerboseLog("Loading dlib small 5 shape predictor model [{}]...".format(
          path_small_shape_model))
      self._shape_small_model = dlib.shape_predictor(path_small_shape_model)
      self.logger.VerboseLog("Done loading dlib small 5 shape predictor model.", show_time = True)

    self.logger.VerboseLog("Loading face detector ...")    
    self._face_detector = dlib.get_frontal_face_detector()
    self.logger.VerboseLog("Done loading face detector.", show_time = True)    
    
    self.logger.VerboseLog("Loading face recognition model ...")
    if hasattr(dlib, "face_recognition_model_v1"):
      self._face_recog = dlib.face_recognition_model_v1(path_faceid_model)
    else:
      self._face_recog = None
      ver = 0
      self.logger.VerboseLog("Dlib face recognition model NOT available v{}.".format(ver), show_time = True)
    self.logger.VerboseLog("Done loading face recognition model.", show_time = True)
    
    return
  
  def GetFaceInfo(self, np_image, get_shape = True, get_id_name = True):
    """
     returns a tuple (BOX, SHAPE, EMBED, ID, NAME) containing facial info such as:
         BOX - LTRB tuple if face detected in frame or None otherwise
         SHAPE - facial landmarks if get_shape
         EMBED - embedding vector if get_embed
         ID - db user ID if get_id_name
         NAME -  db user name if get_id_name
    """
    _found_box = None
    _shape = None
    _embed = None
    _id = None
    _name = "???"
    self.logger.start_timer("  FaceDetect")
    fbox, _found_box = self.face_detect(np_image)    
    self.logger.end_timer("  FaceDetect")
    if get_shape and (fbox != None):
      self.logger.start_timer("  FaceLandmarks")
      landmarks, _shape= self.face_landmarks(np_image, fbox)
      self.logger.start_timer("  FaceLandmarks")
      if landmarks != None:
        if get_id_name:
          self.logger.start_timer("  FaceID")
          _id, _name, _embed  = self.face_id_maybe_save(np_image, landmarks)
          self.logger.end_timer("  FaceID")
    return (_found_box, _shape, _embed, _id, _name)
  
  def get_stats(self):
    mat = self.df_faces[self.feats_names]
    dists = pairwise_distances(mat)
    df = pd.DataFrame(dists)
    df.columns = self.df_faces[self.NAME_FIELD]
    df.set_index(self.df_faces[self.NAME_FIELD], inplace = True)
    return df
  
  def _get_distances(self, embed, embed_matrix):
    result = None
    if embed_matrix.shape[0] > 0:
      dists = (embed_matrix - embed)**2
      dists = np.sum(dists, axis = 1)
      dists = np.sqrt(dists)
      result = dists
    return result
       
  
  def get_id_vs_all(self, pers_id):
    embed = self.df_faces[self.df_faces[self.ID_FIELD] == pers_id][self.feats_names].values.ravel()
    other_df = self.df_faces[self.df_faces[self.ID_FIELD] != pers_id]
    other_df_short = other_df[[self.ID_FIELD, self.NAME_FIELD]].copy()
    other_embeds = other_df[self.feats_names].values    
    other_df_short.loc[:,'DIST'] = list(self._get_distances(embed, other_embeds))
    return other_df_short
       
  
  def _get_current_matrix(self):
    np_matrix = self.df_faces[self.feats_names].values
    return np_matrix
  
  def _save_data(self):
    self.df_faces.to_csv(self.data_file, index = False)
    return
  
  def _find_closest_embedding(self, embed):
    """
     given (NR_EMBEDDINGS,) vector finds closest embedding and returns ID
    """
    result = -1
    np_embeds = self._get_current_matrix()
    if np_embeds.shape[0]>0:
      dists = (np_embeds - embed)**2
      dists = np.sum(dists, axis = 1)
      dists = np.sqrt(dists)
      min_dist = np.min(dists)
      if min_dist <= self.score_threshold:
        result = np.argmin(dists)      
        if self.DEBUG:
          self.logger.VerboseLog(" OmniFaceEngine: Person ID [Idx:{} Dist:{:3f}]".format(
              result, min_dist))
    return result
  
  
  def _create_identity(self, embed):
    """
    receives embed and creates new identity in data store
    returns ID and Name
    """
    pers_id = self.df_faces.shape[0] + 10
    pers_name = "PERSOANA_#{}".format(pers_id)
    rec = {}
    rec[self.ID_FIELD] = pers_id
    rec[self.NAME_FIELD] = pers_name
    for i,col in enumerate(self.feats_names):
      rec[col] = embed[i]
      
    self.last_rec = rec
    self.df_faces = self.df_faces.append(rec, ignore_index = True)
    self._save_data()
    if self.DEBUG:
      self.logger.VerboseLog(" OmniFaceEngine: Created new identity {}".format(
          pers_name))
    return pers_id, pers_name


  def _get_name_by_id(self, idpers, use_index = False):
    if use_index:
      sname = self.df_faces.loc[idpers,self.NAME_FIELD]
    else:
      sname = self.df_faces[self.df_faces[self.ID_FIELD]==idpers].loc[0,self.NAME_FIELD]
    return sname

    
  def _get_id_by_index(self, idx):
    return self.df_faces.loc[idx,self.ID_FIELD]
  
  
  def __dl_face_embed(self, np_img, dl_shape):
    return self._face_recog.compute_face_descriptor(np_img, dl_shape)
  
  def tf_face_embed(self, np_img):
    _result = None
    return _result


  def get_face_embed(self, np_img, dl_shape = None):
    self.logger.start_timer("   FaceEMBED")
    _result = None
    if self.method == 'dlib':
      assert dl_shape != None
      _result = self.__dl_face_embed(np_img, dl_shape = dl_shape)
    elif self.method =='model':
      _result = self.tf_face_embed(np_img)
    self.logger.end_timer("   FaceEMBED")
      
    return _result
      
  
  def _get_info(self, embed):
    """
    given generated embedding get ID and Name of that person
    returns -1, "" if not found
    """
    idx = self._find_closest_embedding(embed)
    idpers = -1
    sname = ""
    if idx != -1:
      sname = self._get_name_by_id(idx, use_index = True)
      idpers = self._get_id_by_index(idx)
    return idpers,sname
  
  def face_id_maybe_save(self, np_img, landmarks_shape):
    """
    tries to ID face. Will return ID, Name, Embed if found or new info if NOT found
    also saves new IDs in own face datastore
    must pass np_img (H,W,C) and landmarks_shape (from face_landmarks)
    """
    result = (None, None, None)
    if self._face_recog != None:
      # get embed
      embed = self.get_face_embed(np_img, dl_shape = landmarks_shape)
      # try to find if avail
      pers_id, pers_name = self._get_info(embed)
      if pers_id == -1:
        # now create new identity
        pers_id, pers_name = self._create_identity(embed)
      result = (pers_id, pers_name, embed)
    return result
  
  def face_detect(self, np_img):
    """
     face detector - will return 1st bounding box both in dlib format and tuple format
     will return None if nothing found
    """
    boxes = self._face_detector(np_img)
    result = (None, None)
    if len(boxes)>0:
      box = boxes[0]
      LTRB = (box.left(),box.top(),box.right(),box.bottom())
      result = (box, LTRB)
    return result
    
  def face_landmarks(self, np_img, dlib_box, large_landmarks = True):
    """
     face landmarks generator - will return numpy array of [points,2] or None if
     nothing found
    """
    result = (None,None)

    if large_landmarks:
      func = self._shape_large_model
      nr_land = 68
    else:
      func = self._shape_small_model
      nr_land = 5
    
    landmarks = func(np_img, dlib_box)
    np_landmarks = np.zeros((nr_land,2))
    for i in range(nr_land):
      np_landmarks[i] = (landmarks.part(i).x, landmarks.part(i).y)
    
    result = landmarks, np_landmarks
    
    return result