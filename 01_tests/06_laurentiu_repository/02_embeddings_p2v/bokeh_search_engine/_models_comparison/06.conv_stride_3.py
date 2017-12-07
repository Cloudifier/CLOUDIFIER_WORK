from lib_server import Server
import os
from bokeh.io import curdoc

prods_filename = os.path.join("D:\\", "Google Drive\\_hyperloop_data\\recom_compl\\_data\\PROD.csv")
np_path = os.path.join("D:\\", "Google Drive\\_hyperloop_data\\recom_compl\\models_comparison")
server = Server(prods_filename, os.path.join(np_path, '06.conv_stride_3'))

server.create_plot('map', 800, 1500, 'cosine')
server.create_plot('map', 500, 1000, 'cosine')
server.create_plot('top_k', 650, 650, 'cosine')
server.create_plot('top_k', 650, 650, 'euclidean')

server.draw_products()  # initial load of the data

curdoc().add_root(server.create_layout())
curdoc().title = "CONV_STRIDE_3"