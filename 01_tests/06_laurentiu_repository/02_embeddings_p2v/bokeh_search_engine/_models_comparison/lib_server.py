import pandas as pd
import numpy as np
import os

from bokeh.plotting import figure
from bokeh.layouts import widgetbox, column, row
from bokeh.models import ColumnDataSource, HoverTool, Div,\
  BoxZoomTool, ResetTool, WheelZoomTool, PanTool, BoxAnnotation,\
  Arrow, VeeHead, Title, LabelSet
from bokeh.models.widgets import TextInput, PreText, TableColumn,DataTable, Panel,\
  Tabs, RadioButtonGroup, Button

dict_colors = {0: "red", 1: "purple", 2: "salmon", 3: "springgreen",
               4: "orange", 5: "yellow", 6: "blue", 7: "aqua", 8: "yellowgreen",
               9: "sienna", 10: "seagreen", 11: "hotpink", 12: "orchid",
               13: "turquoise", 14: "darkslategray", 15: "indianred"}


class DataCollector:
  def __init__(self, filename_prods, np_path):
    self.df_prods = pd.read_csv(filename_prods, encoding='ISO-8859-1')
    self.low_dim_embs = np.load(os.path.join(np_path, 'tsne_10000_products.npy'))
    self.y_kmeans = np.load(os.path.join(np_path, 'y_kmeans_10000_products.npy'))
    self.top_k_indexes_cosine = np.load(os.path.join(np_path, 'top_100_indexes_cosine.npy'))
    self.top_k_distances_cosine = np.load(os.path.join(np_path, 'top_100_distances_cosine.npy'))
    self.top_k_indexes_euclidean = np.load(os.path.join(np_path, 'top_100_indexes_euclidean.npy'))
    self.top_k_distances_euclidean = np.load(os.path.join(np_path, 'top_100_distances_euclidean.npy'))
    self.norm_embeddings = np.load(os.path.join(np_path, 'norm_embeddings.npy'))
    self.embeddings = np.load(os.path.join(np_path, 'embeddings.npy'))
    
    colors = list()
    for key in self.y_kmeans:
      colors.append(dict_colors[key])
    self.df_prods["COLOR"] = pd.Series(colors)
    self.df_prods["x"] = pd.Series(self.low_dim_embs[:, 0])
    self.df_prods["y"] = pd.Series(self.low_dim_embs[:, 1])
    self.df_prods = self.df_prods[:10000]

class Map:
  def __init__(self, plot_height, plot_width, metric):
    hover = HoverTool(tooltips=[
      ("Nume", "@name"),
      ("ID", "@old_id")
    ])
    
    self.metric = metric
    self.source = ColumnDataSource(data = dict(x = [], y = [], color = [], name = [], old_id = []))
    self.p = figure(plot_height = plot_height, plot_width = plot_width, title = "",
                    tools = [hover, BoxZoomTool(), WheelZoomTool(), ResetTool(), PanTool()])
    self.p.scatter(x="x", y="y", source=self.source, radius=0.25,
                   color="color", line_color=None)
    self.p.outline_line_width = 7
    self.p.outline_line_alpha = 0.3
    self.annotations = [None, None]

class TopKPlot:
  def __init__(self, plot_height, plot_width, metric):
    hover = HoverTool(tooltips=[
      ("Nume", "@name"),
      ("ID", "@old_id"),
      ("Distance", "@dist{0,0.000}")
    ])
    self.metric = metric
    self.source = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[], old_id=[], dist=[], short_name=[]))
    self.p = figure(#plot_height=plot_height, plot_width=plot_width,
                    title="Fereastra in care se vor afisa rezultatele cautarii",
                    tools=[hover, BoxZoomTool(), WheelZoomTool(), ResetTool(), PanTool()])
    self.p.scatter(x="x", y="y", source=self.source, size=8, color="color", line_color=None)
    self.p.outline_line_width = 7
    self.p.outline_line_alpha = 0.3
    
    self.bottom_title_search = Title(text="-", align="center")
    self.labels = LabelSet(x='x', y='y', text='short_name', level='glyph',
                           x_offset=-30, y_offset=5, source=self.source, render_mode='canvas',
                           text_font_size="5pt")
    self.labels.visible = False
    self.p.add_layout(self.bottom_title_search, "below")
    self.p.add_layout(self.labels)
    self.annotations = [None, None]
    
    self.button_group_labels = RadioButtonGroup(labels=["Ascundere nume", "Afisare nume"], active=0)
    self.button_group_labels.on_change('active', lambda attr, old, new: self.change_labels_visibility_callback())
    
    self.button_group_arrows = RadioButtonGroup(labels=["Ascundere indicatori", "Afisare indicatori"], active=1)
    self.button_group_arrows.on_change('active', lambda attr, old, new: self.change_arrows_visibility_callback())
    
    self.source_topk = ColumnDataSource(data=dict(ID=[], name=[], distance=[]))
    columns = [
        TableColumn(field="ID", title="ID Produs", width=75),
        TableColumn(field="name", title="Nume Produs"),
        TableColumn(field="distance", title="Distanta", width=75),
    ]
    self.table_topk = DataTable(source=self.source_topk, columns=columns, width=600)
    self.table_topk.height = 0
    
    self.source_topk.on_change('selected', self.select_neighbor_callback)
    
  def change_labels_visibility_callback(self):
    self.labels.visible = (self.button_group_labels.active == 1)
    
  def change_arrows_visibility_callback(self):
    for annotation in self.annotations:
      if annotation is not None:
          annotation.visible = (self.button_group_arrows.active == 1)
    
  def select_neighbor_callback(self, attr, old, new):
    if self.annotations[0] is not None:
      self.annotations[0].visible = False
    
    ID = new['1d']['indices'][0]
    prod_x = self.topk.iloc[ID]['x_new']
    prod_y = self.topk.iloc[ID]['y_new']
    
    arrow = Arrow(end=VeeHead(size=10), line_color='black', line_width=3,
               x_start=prod_x - 10, y_start=prod_y+10, x_end=prod_x, y_end=prod_y)
    
    self.annotations[0] = arrow
    self.p.renderers.extend([arrow])
    
  def populate_table(self, data):
    self.topk = data
    self.source_topk.data = dict(
        ID=data['ITEM_ID'],
        name=data['PROD_NAME'],
        distance=data['DIST'].apply(lambda x: round(x,4))
    )
    self.table_topk.height = 550

  def update(self, df, prod_id, found_id):
    if found_id:
      prod_x = df.iloc[0]['x_new']
      prod_y = df.iloc[0]['y_new']
      
      self.p.title.text = "Cei mai apropiati %d vecini de produsul cu ID %d (%s)" % (len(df)-1, prod_id, df.iloc[0]['PROD_NAME'])
      self.bottom_title_search.text = df.iloc[0]['PROD_NAME']
      self.p.outline_line_color = df.iloc[0]['COLOR']
      self.populate_table(df[1:][['ITEM_ID', 'PROD_NAME', 'DIST', 'x_new', 'y_new']])
      
      arrow = Arrow(end=VeeHead(size=10), line_color='orange', line_width=3,
             x_start=prod_x - 10, y_start=prod_y, x_end=prod_x, y_end=prod_y)
      
      self.annotations[1] = arrow
      self.p.renderers.extend([arrow])
      
      self.source.data = dict(
        x=df["x_new"],
        y=df["y_new"],
        color=df["COLOR"],
        name=df["PROD_NAME"],
        old_id=df["ITEM_ID"],
        dist=df["DIST"],
        short_name=df.PROD_NAME.str.slice(0,20)
      )
    else:
      self.p.title.text = "Produsul cu ID %d nu a fost gasit" % prod_id
      self.bottom_title_search.text = ""
      self.p.outline_line_color = "gainsboro"
      self.table_topk.height = 0

class Server:
  def __init__(self, filename_prods, np_path):
    self.data_collector = DataCollector(filename_prods, np_path)

    title_text = "<h1>Explorator interactiv al modelului de recomandari ce detecteaza complementaritatea produselor SSB</h1>"
    div_text = """
    <p align='justify'><b>Pentru interactiunea cu aplicatia, se va folosi TextBox-ul ce se afla mai sus pentru a va fi returnat un tabel cu toate ID-urile produselor al caror nume contin sirul introdus de la tastatura.
    Cautarea produsului pe harta se face prin selectarea liniei corespunzatoare din tabel.<br><br> Aplicatia afiseaza 2 harti: prima surprinde modul in care sunt grupate primele 10,000 produse, iar cea de-a doua va afisa top 100 produse apropiate de produsul selectat.
    </b></p>"""
    
    self.maps = []
    self.topk_plots = []
    self.title = Div(text=title_text, width=440)
    self.descript = Div(text=div_text, width=440)
    self.product_id = TextInput()
    self.product_name = TextInput(title="Introduceti numele produsului cautat")
    self.possibilities_descript = PreText(text='', width=460)

    self.show_topk_button = Button(label="Gasire complementaritate", button_type="success")
    self.show_topk_button.on_click(self.update_topk_plots)
    self.show_topk_button.disabled = True

    self.product_id.on_change('value', lambda attr, old, new: self.update_maps())
    self.product_name.on_change('value', lambda attr, old, new: self.show_possibilities())
    self.panels = []

    self.create_possibilities_fields()

  def create_plot(self, plot_type, plot_height, plot_width, metric='cosine'):
    if metric not in ['cosine', 'euclidean']:
      raise Exception('Please insert metric for this type of plot: cosine or euclidean')
    
    if plot_type not in ['map', 'top_k']:
      raise Exception("Error! plot_type not in ['map', 'top_k']")

    if plot_type is 'map':
      self.maps.append(Map(plot_height = plot_height, plot_width = plot_width, metric = metric))
    elif plot_type is 'top_k':
      self.topk_plots.append(TopKPlot(plot_height = plot_height, plot_width = plot_width, metric = metric))
    
  def create_possibilities_fields(self):
    self.df_possibilities = None
    self.source_possibilities = ColumnDataSource(data=dict(ID=[], name=[]))
    columns = [
        TableColumn(field="ID", title="ID Produs", width=75),
        TableColumn(field="name", title="Nume Produs"),
    ]
    self.table_possibilities = DataTable(source=self.source_possibilities, columns=columns, width=440, height=350)
    self.table_possibilities.height = 0
    
    self.source_possibilities.on_change('selected', self.select_data_entry_callback)
    
  def select_data_entry_callback(self, attr, old, new):
    ID = new['1d']['indices'][0]
    self.product_id.value = str(self.df_possibilities.iloc[ID].ITEM_ID)
  
  def create_layout(self):    
    layout_tab1 = row(column(self.title,
                             self.product_name,
                             widgetbox(self.show_topk_button),
                             self.possibilities_descript,
                             widgetbox(self.table_possibilities),
                             self.descript),
                      self.maps[1].p)
    
    layout_tab2 = column(self.maps[0].p)
     
    layout_tab3 = column(row(self.topk_plots[0].button_group_labels,
                             self.topk_plots[0].button_group_arrows),
                         row(self.topk_plots[0].p,
                             self.topk_plots[0].table_topk))
    
    
    self.panels.append(Panel(title = 'Cautare produs', child = layout_tab1))
    self.panels.append(Panel(title = 'Harta 01.01.2014-31.08.2017', child = layout_tab2))
    self.panels.append(Panel(title = 'Top prod. complementare', child = layout_tab3))
    
    return Tabs(tabs = self.panels)
  
  def draw_products(self):

    for m in self.maps:
      df_prods = self.data_collector.df_prods
      m.p.xaxis.axis_label = "t-SNE-x"
      m.p.yaxis.axis_label = "t-SNE-y"
      m.p.title.text = "[Prod2Vec] Harta produselor SSB creata pe baza complementaritatii lor"
      m.source.data = dict(
          x=df_prods["x"],
          y=df_prods["y"],
          color=df_prods["COLOR"],
          name=df_prods["PROD_NAME"],
          old_id=df_prods["ITEM_ID"]
      )
      
  def show_possibilities(self):
    prod_name_val = self.product_name.value.strip()
    df_prods = self.data_collector.df_prods
    self.df_possibilities = df_prods[df_prods.PROD_NAME.str.lower().str.contains(prod_name_val.lower())==True]
    nr_possibilities = len(self.df_possibilities)
    if nr_possibilities == 0:
      self.possibilities_descript.text = "Numele introdus nu corespunde niciunui produs"
      self.table_possibilities.height = 0
      self.show_topk_button.disabled = True
    else:
      self.possibilities_descript.text = "Rezultatul cautarii:"
      self.source_possibilities.data = dict(
          ID=self.df_possibilities['ITEM_ID'],
          name=self.df_possibilities['PROD_NAME']
      )
      self.table_possibilities.height = 250
      if nr_possibilities <= 6:
        self.table_possibilities.height = 150
      
      if nr_possibilities <= 3:
        self.table_possibilities.height = 100
      
      self.show_topk_button.disabled = False
  
  def update_maps(self):
    prod_id = self.product_id.value.strip()
    
    for m in self.maps:
      for annotation in m.annotations:
        if annotation is not None:
          annotation.visible = False
    
    if prod_id != "":
      for m in self.maps:
        data = self.data_collector
        selected = data.df_prods[data.df_prods.ITEM_ID == int(prod_id)]
        if len(selected) != 0:  
          prod_x = selected.iloc[0]['x']
          prod_y = selected.iloc[0]['y']
          arrow = Arrow(end=VeeHead(size=10), line_color='black', line_width=3,
                 x_start=prod_x - 8, y_start=prod_y+20, x_end=prod_x, y_end=prod_y)
          box = BoxAnnotation(plot=m.p, left=prod_x-12, right=prod_x+12,
                              top=prod_y-12, bottom=prod_y+12, fill_alpha=0.4,
                              fill_color='green')
          m.annotations[0] = box
          m.annotations[1] = arrow
          
          m.p.renderers.extend([box, arrow])
    
    
  def tsne(self, embeddings):
    from sklearn.manifold import TSNE    
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, random_state=42)    
    low_dim_embs = tsne.fit_transform(embeddings)    
    return low_dim_embs
  
  def select_products(self, season = None):
    data = self.data_collector
    
    #if metric is 'cosine':
    top_k_indexes = data.top_k_indexes_cosine
    top_k_distances = data.top_k_distances_cosine
    emb = np.array(data.norm_embeddings)
    #elif metric is 'euclidean':
    #  top_k_indexes = data.top_k_indexes_euclidean
    #  top_k_distances = data.top_k_distances_euclidean
    #  emb = np.array(data.embeddings)
    
    prod_id = self.product_id.value.strip()
    selected = pd.DataFrame(columns=list(data.df_prods.columns) + ['DIST'])
    found_id = False
    if prod_id != "":
      selected = data.df_prods[data.df_prods.ITEM_ID == int(prod_id)]
      if len(selected) == 0:
        selected = selected.assign(DIST=np.nan)
        return selected, int(prod_id), False
  
      new_id = selected.iloc[0]['NEW_ID'] - 1
      k_indexes = top_k_indexes[new_id]
      k_embeddings = emb[k_indexes]
      k_embeddings = np.vstack([k_embeddings, emb[new_id]])
      
      k_low_dim_embs = self.tsne(k_embeddings)
      k_distances = top_k_distances[new_id]
      
      k_distances = np.insert(k_distances, 0, 0)
      k_products = data.df_prods.iloc[k_indexes]
      selected = selected.append(k_products, ignore_index=True)
      selected['DIST'] = pd.Series(k_distances)
      selected['x_new'] = pd.Series(k_low_dim_embs[:, 0])
      selected['y_new'] = pd.Series(k_low_dim_embs[:, 1])
      #selected['x_new'] = selected['x']
      #selected['y_new'] = selected['y']
      
      found_id = True
  
    return selected, int(prod_id), found_id
  
  def update_topk_plots(self):
    df1, prod_id, found_id = self.select_products()
    
    for topk_plot in self.topk_plots:
      for annotation in topk_plot.annotations:
        if annotation is not None:
          annotation.visible = False
      
      topk_plot.update(df1, prod_id, found_id)

'''
prods_filename = os.path.join("D:\\", "Google Drive\\_hyperloop_data\\recom_compl\\PROD.csv")
np_path = os.path.join("D:\\", "Google Drive\\_hyperloop_data\\recom_compl\\_models_comparison")
server = Server(prods_filename, os.path.join(np_path, '01.flatten'))

server.create_plot('map', 800, 1500, 'cosine')
server.create_plot('map', 500, 1000, 'cosine')
server.create_plot('top_k', 650, 650, 'cosine')
server.create_plot('top_k', 650, 650, 'euclidean')

server.draw_products()  # initial load of the data

curdoc().add_root(server.create_layout())
curdoc().title = "Recom_compl"
'''