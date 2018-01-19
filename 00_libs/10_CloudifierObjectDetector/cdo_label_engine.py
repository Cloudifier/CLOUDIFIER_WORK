# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 13:55:19 2017

@author: Andrei
"""

import sys
import tensorflow as tf


__version__   = "2.1.ODAPI v2"
__author__    = "Cloudifier"
__copyright__ = "(C) Cloudifier SRL"
__project__   = "Cloudifier"  
__credits__   = "Parts used from open-source project OmniDJ by Knowledge Investment Group"


_DATASETS = ['COCO','OID']
_COCO = _DATASETS[0]
_OID = _DATASETS[1]

class ObjectLabelEngine:
  def __init__(self, logger, label_folder = None, dataset = 'coco', 
               labels_file = None, labelmap_file = None,
               num_classes = None):

    dataset = dataset.upper()
    assert dataset in _DATASETS, "Unknown dataset {} ".format(dataset)
    
    self.logger = logger
    
    self.load_config(dataset, label_folder, labels_file = labels_file,
                     labelmap_file = labelmap_file, num_classes = num_classes)
    
    return    
  
  
  def load_config(self, dataset, label_folder = None, labels_file = None, 
                  labelmap_file = None, num_classes = None):
    self.loaded = False
    self.logger.VerboseLog("Loading config for dataset {} [{}]".format(
        dataset, labels_file))
    self.dataset = dataset
    if label_folder is None:
      label_folder = "_classes"

    self.label_folder = label_folder
    self.labels_file = labels_file
    self.labelmap_file = labelmap_file
    self.num_classes = num_classes
    
    if self.dataset == _COCO:
      self._prepare_coco()
    elif self.dataset ==_OID:
      self._prepare_oid()
      
    assert self.loaded, "Labels configuration not loaded !"  
    self.logger.VerboseLog("Done loading config for {} [{}]".format(
        dataset, labels_file))
    
    return
  
  
  def GetLabel(self, idx):
    result ="None"
    if self.dataset == _COCO:
      result = self._coco_GetLabelByIndex(idx)
    elif self.dataset ==_OID:
      result = self._oid_GetLabelByIndex(idx)
    return result
  
  def _coco_LoadClasses(self, cfile = ''):  
    assert cfile != '', "Must provide coco labels file" 
    self.logger.VerboseLog("Loading COCO labels from ...{}".format(cfile[-30:]))
    with open(cfile) as f:
      lines = f.read().splitlines()
    labels = list(lines)
    for i,line in enumerate(lines):
      labels[i] = line.split(": ")[1]
    return labels

  
  def _prepare_coco(self):
    dict_file = self.labels_file
    num_classes = self.num_classes
    if num_classes is None: num_classes = 183
    if dict_file is None: dict_file = "coco_ro.txt"
    
    labels_fn = self.logger.GetFileFromFolder(self.label_folder,dict_file)
    self._coco_labels = self._coco_LoadClasses(labels_fn)
    self.loaded = True
    return
  
  def _coco_GetLabelByIndex(self, idx):
    return self._coco_labels[idx]
  
  def _prepare_oid(self):
    dict_file = self.labels_file
    map_file = self.labelmap_file
    num_classes = self.num_classes
    if num_classes is None: num_classes = 6012
    if dict_file is None: dict_file = "dict.csv"
    if map_file is None: map_file = "labelmap.txt"
    
    dict_fn = self.logger.GetFileFromFolder(self.label_folder,dict_file)
    map_fn = self.logger.GetFileFromFolder(self.label_folder,map_file)
    

    (self._oid_labelmap, 
     self._oid_label_dict) = self._oid_LoadLabelMaps(num_classes, 
                                                     map_fn, 
                                                     dict_fn)
    self.loaded = True
    return
    

  def _oid_LoadLabelMaps(self, num_classes, labelmap_path, dict_path):
    """Load index->mid and mid->display name maps.
    Args:
      labelmap_path: path to the file with the list of mids, describing predictions.
      dict_path: path to the dict.csv that translates from mids to display names.
    Returns:
      labelmap: an index to mid list
      label_dict: mid to display name dictionary
    """
    self.logger.VerboseLog("Loading OID label map...")
    labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path).readlines()]
    if len(labelmap) < num_classes:
      self.logger.VerboseLog(
          "Label map loaded from ...{} contains {} lines while the number of classes is {}".format(
              labelmap_path[-30:], len(labelmap), num_classes))
      sys.exit(1)
    self.logger.VerboseLog("Done loading OID label map.", show_time = True)
    
    self.logger.VerboseLog("Loading OID dictionary ...{} ...".format(
        dict_path[-30:]))
    label_dict = {}
    for i, line in enumerate(tf.gfile.GFile(dict_path).readlines()):
      words = [word.strip(' "\n') for word in line.split(',', 1)]
      label_dict[words[0]] = words[1]
    self.logger.VerboseLog("Done OID dictionary.", show_time = True)
  
    return labelmap, label_dict
  
  
  def _oid_GetLabelByIndex(self,idx):
    """
    """
    mid = self._oid_labelmap[idx]
    display_name = self._oid_label_dict.get(mid, 'unknown')
    return display_name
