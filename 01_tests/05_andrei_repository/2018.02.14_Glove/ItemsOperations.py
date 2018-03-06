# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:08:30 2018

@author: AndreiS
"""

import pandas as pd
import numpy as np
from utils import LoadLogger, guess_encoding
from sklearn.metrics import pairwise_distances
import json
import h5py

DEBUG = False

class ItemsOperations():
    
    def __init__(self, config_file = "config.txt"):
        
        self.logger = LoadLogger(lib_name = "ItemsOp",
                                 TF_KERAS = False,
                                 config_file = "config.txt")
        self.config_file = config_file
        self._get_config_data()
        
    def _get_config_data(self):
        
        self.embeddings_file = self.logger.config_data['EMB_FILE']
        self.embeddings_start_idx = int(self.logger.config_data['EMB_START_IDX'])
        self.embeddings_end_idx = int(self.logger.config_data['EMB_END_IDX'])
        self.item_idx = int(self.logger.config_data['ITEM_IDX'])
        self.distance = self.logger.config_data['DISTANCE']
        self.pair_items = self.logger.config_data['PAIR_ITEMS']
        self.triplet_items = self.logger.config_data['TRIPLET_ITEMS']
        
        self._load_data()
        
    
    def get_items_to_test(self, items_file = None):
        
        if items_file is None:
            items_file = self.config_file
            
        with open(items_file) as fp:
            config_data = json.load(fp)
       
        self.pair_items = config_data['PAIR_ITEMS']
        self.triplet_items = config_data['TRIPLET_ITEMS']
        
        
    def _load_data(self):
        
        
        full_path = self.logger.GetDataFile(self.embeddings_file)
        if ".txt" in self.embeddings_file:
            with open(full_path, encoding = guess_encoding(full_path)) as embeddings_fp:
                file_content = embeddings_fp.readlines()
                file_content = [line.split(' ') for line in file_content]
                file_content = [ [elem[0]] + list(map(float, elem[1:])) for elem in file_content]
                embeddings_colnames = ['E_' + str(i) for i in range(len(file_content[0]) - 1)]
                self.embeddings_df = pd.DataFrame(file_content, columns = ['Item'] + embeddings_colnames)
        elif ".csv" in self.embeddings_file:
            self.embeddings_df  = pd.read_csv(full_path, encoding='ISO-8859-1')
        elif ".h5" in self.embeddings_file:
            model_cache = h5py.File(full_path, 'r')
            self.embeddings_df = model_cache['embeddings'].value
        else:
            self.logger.VerboseLog("Unknown embeddings file type")
            exit(-1)
            
        self.embeddings_np = np.array(self.embeddings_df.iloc[:, self.embeddings_start_idx:
                                      self.embeddings_end_idx + 1])
        self.items_np = self.embeddings_df.iloc[:, self.item_idx].values
        self.item_by_idx = dict(zip(self.items_np, range(0, len(self.items_np))))
        
    def _find_idx(self, item_name):
        return self.item_by_idx[item_name]
    
    def _top_K(self, item_name, dist_from_all, k):
        
        sorted_indexes = np.argsort(dist_from_all)
        self.logger.VerboseLog("Top {} closest items for {}".format(k, item_name))
        top_k_indexes = sorted_indexes[1 : (k + 1)]
        top_k_items = list(zip(np.take(self.items_np, top_k_indexes), dist_from_all[top_k_indexes]))
        top_k_items = [(item, "{:.3f}".format(dist)) for item, dist in top_k_items]
        self.logger.VerboseLog("{}".format(top_k_items))
        
        self.logger.VerboseLog("Top {} farest items for {}".format(k, item_name))
        bottom_k_indexes = sorted_indexes[-k :]
        bottom_k_items = list(zip(np.take(self.items_np, bottom_k_indexes), dist_from_all[bottom_k_indexes]))
        bottom_k_items = [(item, "{:.3f}".format(dist)) for item, dist in bottom_k_items]
        self.logger.VerboseLog("{}".format(bottom_k_items))
        
        
    def _cosine_top_K(self, item_name, item_emb, k):
        
        norm_embeddings = np.linalg.norm(self.embeddings_np, axis = 1)
        norm_item = np.linalg.norm(item_emb) 
        
        dist_from_all = 1 - (self.embeddings_np / norm_embeddings.reshape(-1, 1)).dot(
                item_emb / norm_item)
        #dist_from_all_norm = np.linalg.norm(dist_from_all, axis = 1)
        self._top_K(item_name, dist_from_all, k)

    def _euclidean_top_K(self, item_name, item_emb, k):
        
        dist_from_all = pairwise_distances(item_emb.reshape(1,-1), self.embeddings_np).flatten()
        self._top_K(item_name, dist_from_all, k)
        
    def _compute_euclidean_stats(self, items, k):
        
        if len(items) == 2:
            item1_name, item1_emb = items[0]
            item2_name, item2_emb = items[1]
            
            similarity = np.sqrt(np.sum( (item1_emb - item2_emb)**2 ))
        
            self.logger.VerboseLog("Euclidean distance between {} and {}: {}".format(
                item1_name, item2_name, similarity))
        
        for item_name, item_emb in items:
            self._euclidean_top_K(item_name, item_emb, k) 
    
    def _compute_cosine_stats(self, items, k):
        
        if len(items) == 2:
            item1_name, item1_emb = items[0]
            item2_name, item2_emb = items[1]
            item1_norm = np.linalg.norm(item1_emb)
            item2_norm = np.linalg.norm(item2_emb)
        
            similarity = (item1_emb / item1_norm).dot(item2_emb / item2_norm)
        
            self.logger.VerboseLog("Cosine distance between {} and {}: {}".format(
                item1_name, item2_name, similarity))
        
        for item_name, item_emb in items:
            self._cosine_top_K(item_name, item_emb, k) 
    
    def binary_operations(self, items_to_test):
        
        for item1, item2 in items_to_test:
            
            (item1_name, item1_idx), = item1.items()
            (item2_name, item2_idx), = item2.items()
            
            item1_idx = int(item1_idx)
            item2_idx = int(item2_idx)
            
            if item1_idx == -1:
                item1_idx = self._find_idx(item1_name)
            if item2_idx == -1:
                item2_idx = self._find_idx(item2_name)
            
            item1_emb = self.embeddings_np[item1_idx]
            item2_emb = self.embeddings_np[item2_idx]
            
            k = 10
            items = list(zip((item1_name, item2_name), (item1_emb, item2_emb)))
            self._compute_euclidean_stats(items, k)
            
            diff_item = item1_emb - item2_emb
            self._euclidean_top_K("difference between {} and {} using euclidean distance".format(
                    item1_name, item2_name), diff_item, k)
           
            avg_item = (item1_emb + item2_emb) / 2
            
            self._euclidean_top_K("average between {} and {} using euclidean distance".format(
                    item1_name, item2_name), avg_item, k)
            
            print()
            
            self._compute_cosine_stats(items, k)
            self._cosine_top_K("difference between {} and {} using cosine distance".format(
                    item1_name, item2_name), diff_item, k)
            self._cosine_top_K("average between {} and {} using cosine distance".format(
                    item1_name, item2_name), avg_item, k)
            
    def ternary_operations(self, items_to_test):
            
        for item1, item2, item3 in items_to_test:
            
            (item1_name, item1_idx), = item1.items()
            (item2_name, item2_idx), = item2.items()
            (item3_name, item3_idx), = item3.items()
                
            item1_idx = int(item1_idx)
            item2_idx = int(item2_idx)
            item3_idx = int(item3_idx)
        
            if item1_idx == -1:
                item1_idx = self._find_idx(item1_name)
            if item2_idx == -1:
                item2_idx = self._find_idx(item2_name)
            if item3_idx == -1:
                item3_idx = self._find_idx(item3_name)
                
            item1_emb = self.embeddings_np[item1_idx]
            item2_emb = self.embeddings_np[item2_idx]
            item3_emb = self.embeddings_np[item3_idx]
                
            k = 10
            items = list(zip((item1_name, item2_name, item3_name), (item1_emb, item2_emb, item3_emb)))
            self._compute_euclidean_stats(items, k)
                
            operations_item = item1_emb - item2_emb + item3_emb
                
            self._euclidean_top_K("{} - {} + {}  using euclidean distance".format(
                        item1_name, item2_name, item3_name), operations_item, k)
               
            avg_item = (item1_emb + item2_emb + item3_emb) / 3
                
            self._euclidean_top_K("average between {} and {} and {} using euclidean distance".format(
                        item1_name, item2_name, item3_name), avg_item, k)
                
            print()
                
            self._compute_cosine_stats(items, k)
            self._cosine_top_K("{} - {} + {} using cosine distance".format(
                        item1_name, item2_name, item3_name), operations_item, k)
            self._cosine_top_K("average between {} and {} and {} using cosine distance".format(
                        item1_name, item2_name, item3_name), avg_item, k)
            
            print()
            print("----------------------------------------------------------")
            print()
            
    
    def get_items_by_short_name(self, short_name):
        
        mask = [short_name.lower() in item.lower() for item in self.items_np]
        return self.items_np[mask]
        
            
if __name__ == '__main__':
    
    if not DEBUG:
        item_experiment = ItemsOperations()
        
    
    nurofen_items = item_experiment.get_items_by_short_name("nurofen")
    theraflu_items = item_experiment.get_items_by_short_name("theraflu")
    vitc_items = item_experiment.get_items_by_short_name("vitamina c ")
    
    
    
    
    
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
    