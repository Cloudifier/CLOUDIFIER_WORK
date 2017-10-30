import pandas as pd
import numpy as np
from time import time
import os

from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Div, OpenURL, TapTool
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.io import curdoc


'''
title = "[P2V] t-SNE visualization of {:,} products whose {}-d embeddings " \
          "are trained during {} epochs.".format(10000, 64, 5)
'''

###
prods_filename = os.path.join("D:/", "Google Drive/_hyperloop_data/recom_compl/PROD.csv")
print('Se incarca in memorie setul de date cu produse: {} ... '.format(prods_filename))
start = time()
df_prods = pd.read_csv(prods_filename, encoding='ISO-8859-1')
end = time()
print('S-a incarcat setul de date in {:.2f}s .. top 5 prods:\n\n{}'.format(end - start, df_prods.head(5)))

ids = np.array(df_prods['NEW_ID'].tolist()) - 1
ids = list(ids)
dictionary = dict(zip(ids, df_prods['PROD_NAME'].tolist())) # id: prod_name
reversed_dictionary = dict(zip(df_prods['PROD_NAME'].tolist(), ids)) # prod_name: id
#del df_prods
###

path = 'D:\\Google Drive\\_hyperloop_data\\recom_compl\\_mean_flatten_comp\\LOCALemb64_epochs15_momentum_flatten_v1'
norm_embeddings = np.load(os.path.join(path, 'checkpoints\\norm_embeddings_14.npy'))[:10000]
low_dim_embs = np.load(os.path.join(path, 'low_dim_embs\\low_dim_embs_14.npy'))

desc = Div(text=open(os.path.join(os.path.dirname(__file__), "description.html")).read(), width=800)
product = TextInput(title="Numele produsului contine...")

