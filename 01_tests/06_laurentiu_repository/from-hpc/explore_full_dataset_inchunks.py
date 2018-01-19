
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from time import time
from IPython.display import display, HTML
from tqdm import tqdm_notebook as tqdm


# In[2]:


class SimpleLogger:
    def __init__(self):
        self.last_time = time()
        self.c_time = time()
  
    def VerboseLog(self, _str, show_time = False):
        self.last_time = self.c_time
        self.c_time = time()
        if show_time:
            _str += " [{:.2f}s]".format(self.c_time-self.last_time)
        print(_str, flush = True)
  
    def log(self, _str, show_time = False):
        self.last_time = self.c_time
        self.c_time = time()
        if show_time:
            _str += " [{:.2f}s]".format(self.c_time-self.last_time)    
        print(_str, flush = True)
        
logger = SimpleLogger()


# In[6]:


def write_df(df, fname):
    import os
    if not os.path.isfile(fname):
        df.to_csv(fname)
    else:
        df.to_csv(fname, mode = 'a', header = False)
    
    return

def add_vect_to_mco(mco, vect, window = 2, max_count = 65000):
    for i, prod_id in enumerate(vect):
        nr_left_prods = min(vect[:i].shape[0], window)
        nr_right_prods = min(vect[i+1:].shape[0], window)
        for j in range(nr_left_prods):
            l_ind = i - (j + 1)
            l_prod = vect[l_ind]
            if l_prod != prod_id:
                current_mco_val = mco[prod_id, l_prod]
                mco[prod_id, l_prod] = max(current_mco_val + 1 / (j + 2), max_count)

        for j in range(nr_right_prods):
            r_ind = i + j + 1
            r_prod = vect[r_ind]
            if r_prod != prod_id:
                current_mco_val = mco[prod_id, r_prod]
                mco[prod_id, r_prod] = max(current_mco_val + 1 / (j + 2), max_count)

    return mco           
                
def add_batch_to_mco(mco, batch, prod_field_name):
    if batch.shape[0] == 1:
        return mco
    
    mco = add_vect_to_mco(mco, batch[prod_field_name].values)

    return mco
        
def process_chunk(df_chunk, tran_field, tran_det_field, mco, prod_field_name):
    all_unique_trans = df_chunk[tran_field].unique()
    tqdm_works = True
    for i in tqdm(range(all_unique_trans.shape[0])):
        tran = all_unique_trans[i]
        batch = df_chunk[df_chunk[tran_field] == tran]
        batch.sort_values(by = tran_det_field, inplace = True)
        mco = add_batch_to_mco(mco, batch, prod_field_name)
        if tqdm_works:
            one_percent = int(all_unique_trans.shape[0] * 0.01)
            if i % one_percent == 0:
                logger.log('  Processed {:.2f}%.'.format((i / all_unique_trans.shape[0]) * 100))

        
    return mco


# In[4]:


keep_cols = ['TRAN_ID', 'TRAN_DET_ID', 'SITE_ID', 'TIMESTAMP', 'NEW_ID']
all_cols = ['TRAN_ID', 'TRAN_DET_ID', 'SITE_ID', 'CUST_ID', 'ITEM_ID', 'QTY', 'AMOUNT', 'TIMESTAMP', 'NEW_ID']
chunksize = 5e6
nr_products = 28377

reader = pd.read_csv('full_dataset_transactions.csv',
                     names = all_cols,
                     chunksize = chunksize)

logger.log('Creating mco ...')
mco = np.zeros((nr_products + 10, nr_products + 10), dtype = np.float16)
logger.log('   Created mco.', True)


# In[ ]:


for i, batch in enumerate(reader):
    logger.log('Preprocessing batch {} with {:,} entries...'.format(i, batch.shape[0]))
    df = pd.DataFrame(batch[keep_cols])
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df.loc[:, 'FTRAN'] = df['SITE_ID'].astype(str) + df['TRAN_ID'].astype(str)
    df.drop(['TRAN_ID', 'SITE_ID'], axis = 1, inplace = True)
    logger.log('  Done preprocessing batch.', True)
    logger.log('Processing the batch ...')
    mco = process_chunk(df, 'FTRAN', 'TRAN_DET_ID', mco, 'NEW_ID')
    logger.log('  Done processing the batch.', True)
    
    if i == 5:
        break
        
mco.dump('mco.npy')

