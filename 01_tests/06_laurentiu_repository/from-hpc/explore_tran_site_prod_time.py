import pandas as pd
import numpy as np
from time import time

df_prod = pd.read_csv('ITEMS.csv', encoding='ISO-8859-1')
newids = np.array(df_prod['NEW_ID'].tolist()) - 1
newids = list(newids)
ids = df_prod['ITEM_ID'].tolist()
dict_id2newid = dict(zip(ids, newids))

print('Start reading tran_site_prod_time.csv ...')
start = time()
df_tran_site_prod_time = pd.read_csv('tran_site_prod_time.csv')
print('Finished reading tran_site_prod_time.csv! [{:.2f}s]'.format (time() - start))
df_tran_site_prod_time.columns = ['ITEM_ID', 'TIMESTAMP', 'TRAN_ID', 'SITE_ID']

print('Updating ITEM_IDs ...')
df_tran_site_prod_time['ITEM_ID'].update( df_tran_site_prod_time['ITEM_ID'].map(dict_id2newid) )   # series update is an inplace operation

print('Transforming TIMESTAMP column')
df_tran_site_prod_time['TIMESTAMP'] = pd.to_datetime(df_tran_site_prod_time['TIMESTAMP'])

print('Sorting by TIMESTAMP')
df_tran_site_prod_time = df_tran_site_prod_time.sort_values(by='TIMESTAMP')

print('Unique TRAN_ID = {}'.format(df_tran_site_prod_time.TRAN_ID.unique().shape))

print('Creating items baskets ...')
start = time()
X = df_tran_site_prod_time.groupby(['TRAN_ID', 'SITE_ID']).apply(lambda x: x['ITEM_ID'].values)
print('Finished creating items baskets! [{:.2f}s]'.format (time() - start))
