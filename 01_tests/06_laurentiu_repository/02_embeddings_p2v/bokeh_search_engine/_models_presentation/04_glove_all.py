from lib_server_v2 import Server
from bokeh.io import curdoc

server = Server(config_file = 'config_04.txt', suffix = '04_glove_all')

server.CreatePlot('map', 500, 1000)
server.CreatePlot('map', 500, 700)
server.CreatePlot('topk', 550, 550)
server.CreatePlot('topk', 550, 550)
server.CreatePlot('topk', 550, 550)

server.DrawProducts()  # initial load of the data

curdoc().add_root(server.CreateLayout())
curdoc().title = "[GloVe] Toate tranzactiile"