# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:27:52 2017

@author: LaurentiuP

@history:
"""

import os
import pandas as pd
import numpy as np
import itertools
import json


class BigDataProcesser:
  def __init__(self, logger=None, config_file='config.txt'):
    assert config_file != ""
    f = open(config_file)
    self.config_data = json.load(f)
    assert ("DF_CSV" in self.config_data.keys())
    assert ("CHUNKSIZE" in self.config_data.keys())

    self.logger = logger
    self._log("Initializing BigDataProcessor; df=[..{}] ...".format(self.config_data["DF_CSV"][-40:]))

    col_names_keys = ["COL" + str(i) for i in range(1, self.config_data["LAST_COL_ID"] + 1)]
    self.col_names = []
    for key in col_names_keys:
      self.col_names.append(self.config_data[key])
    self.reader = pd.read_csv(self.config_data["DF_CSV"],
                              names = self.col_names,
                              chunksize = self.config_data["CHUNKSIZE"])
    self._log("Initialized BigDataProcessor.")
    return

  def _log(self, str_msg, results = False, show_time = False):
    if self.logger != None:
      self.logger.VerboseLog(str_msg, results, show_time)
    else:
      print(str_msg, flush=True)
    return
  
  
  def GenerateMarketBaskets(self, *args):
    
    keep_columns = list(args)
    if keep_columns == []:
      keep_columns = self.col_names

    self._log("Generating market baskets. Keep cols={} ...".format(keep_columns))
    for i, batch in enumerate(self.reader):
      self._log("  Processing batch {} with {:,} entries ...".format(i+1, batch.shape[0]))
      df = pd.DataFrame(batch[keep_columns])

      #TODO generalizare!!!!
      df.loc[:, 'FTRAN'] = df['SITE_ID'].astype(str) + df['TRAN_ID'].astype(str)
      df.drop(['TRAN_ID', 'SITE_ID'], axis = 1, inplace = True)
      df["IDE"] = df["IDE"].apply(lambda x: x - 1)
      unique_trans_grouped = df.groupby('FTRAN').apply(lambda x: x['IDE'].values)
      unique_trans_grouped = np.array(unique_trans_grouped)
      self._log("  Batch processed.", show_time=True)
      yield unique_trans_grouped
  

  
if __name__ == '__main__':
  bp = BigDataProcesser()