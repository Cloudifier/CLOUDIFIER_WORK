from lib_server_v2 import Server
from bokeh.io import curdoc

server = Server(config_file = 'config_02.txt', suffix = '02_p2v_summer')

server.CreatePlot('map', 500, 1000)
server.CreatePlot('map', 500, 700)
server.CreatePlot('topk', 550, 550)
server.CreatePlot('topk', 550, 550)
server.CreatePlot('topk', 550, 550)

server.DrawProducts()  # initial load of the data

curdoc().add_root(server.CreateLayout())
curdoc().title = "[P2V] Tranzactii vara"