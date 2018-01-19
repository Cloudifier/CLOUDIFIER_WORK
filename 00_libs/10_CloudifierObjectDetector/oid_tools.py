# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 13:55:19 2017

@author: Andrei
"""

import sys

__version__   = "2.1.ODAPI v2"
__author__    = "Cloudifier"
__copyright__ = "(C) Cloudifier SRL"
__project__   = "Cloudifier"  
__credits__   = "Parts used from open-source project OmniDJ by Knowledge Investment Group"


_DATASETS = ['coco','oid']

class ObjectLabelEngine:
  def __init__(self, logger, label_folder, dataset = 'coco', file_name = None):
    assert dataset in _DATASETS, "Unknown dataset "+dataset
    
    self.logger = logger
    self.label_folder = label_folder
    self.dataset = dataset
    self.file_name = file_name
    
    if self.dataset == 'coco':
      self.prepare_coco()
    elif dataset =='oid':
      self.prepare_oid()
    
    return    
  
  def GetLabel(self, idx):
    result ="None"
    if self.dataset == 'coco':
      result = self._coco_GetLabelByIndex(idx)
    elif dataset =='oid':
      result = self._oid_GetLabelByIndex(idx)
    return result

  def _oid_LoadLabelMaps(self, tf, logger, num_classes, labelmap_path, dict_path):
    """Load index->mid and mid->display name maps.
    Args:
      labelmap_path: path to the file with the list of mids, describing predictions.
      dict_path: path to the dict.csv that translates from mids to display names.
    Returns:
      labelmap: an index to mid list
      label_dict: mid to display name dictionary
    """
    labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path).readlines()]
    if len(labelmap) != num_classes:
      logger.VerboseLog(
          "Label map loaded from {} contains {} lines while the number of classes is {}".format(
              labelmap_path, len(labelmap), num_classes))
      sys.exit(1)
  
    label_dict = {}
    for line in tf.gfile.GFile(dict_path).readlines():
      words = [word.strip(' "\n') for word in line.split(',', 1)]
      label_dict[words[0]] = words[1]
  
    return labelmap, label_dict
  
  
  def _oid_GetLabelByIndex(self,idx):
    """
    """
    mid = labelmap[idx]
    display_name = label_dict.get(mid, 'unknown')
    return display_name
