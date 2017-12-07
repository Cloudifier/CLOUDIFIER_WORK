from lib_server import Server
import os
from bokeh.io import curdoc

prods_filename = os.path.join("D:\\", "Google Drive\\_hyperloop_data\\recom_compl\\_data\\PROD.csv")
np_path = os.path.join("D:\\", "Google Drive\\_hyperloop_data\\recom_compl\\models_comparison")
server = Server(prods_filename, os.path.join(np_path, '07.full_softmax_v2\\_hot_season'))

server.create_plot('map', 500, 1000)
server.create_plot('map', 500, 700)
server.create_plot('top_k', 650, 650)

server.draw_products()  # initial load of the data

curdoc().add_root(server.create_layout())
curdoc().title = "Sezon cald"