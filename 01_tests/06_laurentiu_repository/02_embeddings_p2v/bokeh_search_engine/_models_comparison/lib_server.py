import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE

from bokeh.plotting import figure
from bokeh.layouts import widgetbox, column, row
from bokeh.models import ColumnDataSource, HoverTool, Div,\
  BoxZoomTool, ResetTool, WheelZoomTool, PanTool, BoxAnnotation,\
  Arrow, VeeHead, Title, LabelSet
from bokeh.models.widgets import TextInput, PreText, TableColumn,DataTable, Panel,\
  Tabs, RadioButtonGroup, Button
from time import time


def load_module(module_name, file_name):
  """
  loads modules from _pyutils Google Drive repository
  usage:
    module = load_module("logger", "logger.py")
    logger = module.Logger()
  """
  from importlib.machinery import SourceFileLoader
  home_dir = os.path.expanduser("~")
  valid_paths = [
                 os.path.join(home_dir, "Google Drive"),
                 os.path.join(home_dir, "GoogleDrive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                 os.path.join("C:/", "GoogleDrive"),
                 os.path.join("C:/", "Google Drive"),
                 os.path.join("D:/", "GoogleDrive"),
                 os.path.join("D:/", "Google Drive"),
                 ]

  drive_path = None
  for path in valid_paths:
    if os.path.isdir(path):
      drive_path = path
      break

  if drive_path is None:
    raise Exception("Couldn't find google drive folder!")

  utils_path = os.path.join(drive_path, "_pyutils")
  print("Loading [{}] package...".format(os.path.join(utils_path,file_name)),flush = True)
  module_lib   = SourceFileLoader(module_name, os.path.join(utils_path, file_name)).load_module()
  print("Done loading [{}] package.".format(os.path.join(utils_path,file_name)),flush = True)

  return module_lib


dict_colors = {0: "red", 1: "purple", 2: "salmon", 3: "springgreen",
               4: "orange", 5: "yellow", 6: "blue", 7: "aqua", 8: "yellowgreen",
               9: "sienna", 10: "seagreen", 11: "hotpink", 12: "orchid",
               13: "turquoise", 14: "darkslategray", 15: "indianred",
               16: "darkred", 17: "darkkhaki", 18: "lime", 19: "mediumspringgreen",
               20: "darkgreen", 21: "navy", 22: "mediumslateblue",
               23: "teal", 24: "palevioletred"}

class DataCollector:
  def __init__(self, data_path, load_mco = True):
    try:
      self.df_prods = pd.read_csv(os.path.join(data_path, 'ITEMS.csv'), encoding='ISO-8859-1')
      self.low_dim_embs = np.load(os.path.join(data_path, 'tsne_10000_products.npy'))
      self.y_kmeans = np.load(os.path.join(data_path, 'y_kmeans_10000_products.npy'))
      self.top_k_indexes = np.load(os.path.join(data_path, 'top_100_indexes.npy'))
      self.top_k_distances = np.load(os.path.join(data_path, 'top_100_distances.npy'))
      self.norm_embeddings = np.load(os.path.join(data_path, 'norm_embeddings.npy'))
      if load_mco:
        self.mco = np.load(os.path.join(data_path, 'mco_f32.npy'))
    except OSError as e:
      print('Data not found in [...{}]'.format(data_path[-20:]))

    colors = list()
    for key in self.y_kmeans:
      colors.append(dict_colors[key])
    self.df_prods["COLOR"] = pd.Series(colors)
    self.df_prods["x"] = pd.Series(self.low_dim_embs[:, 0])
    self.df_prods["y"] = pd.Series(self.low_dim_embs[:, 1])
    self.df_prods = self.df_prods[:10000]


class Map:
  def __init__(self, plot_height, plot_width):
    hover = HoverTool(tooltips=[
      ("Nume", "@name"),
      ("ID", "@old_id")
    ])

    self.source = ColumnDataSource(data = dict(x = [], y = [], color = [], name = [], old_id = []))
    self.p = figure(plot_height = plot_height, plot_width = plot_width, title = "",
                    tools = [hover, BoxZoomTool(), WheelZoomTool(), ResetTool(), PanTool()])
    self.p.scatter(x="x", y="y", source=self.source, radius=0.25,
                   color="color", line_color=None)
    self.p.outline_line_width = 7
    self.p.outline_line_alpha = 0.3
    self.annotations = [None, None]


class TopKPlot:
  def __init__(self, plot_height, plot_width, has_co_occ_data):
    hover = HoverTool(tooltips=[
      ("Nume", "@name"),
      ("ID", "@old_id"),
      ("Distance", "@dist{0,0.000}")
    ])
    
    self.has_co_occ_data = has_co_occ_data
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
    
    # Create ColumnDataSource
    table_data_dict = dict(ID = [], name = [], distance = [])
    if self.has_co_occ_data:
      table_data_dict['co_occ_percentage'] = []
    self.source_topk = ColumnDataSource(data = table_data_dict)

    # Create Table Columns
    columns = [
        TableColumn(field = "ID", title = "ID Produs", width = 75),
        TableColumn(field = "name", title = "Nume Produs")
    ]
    if self.has_co_occ_data:
      columns.append(TableColumn(field = "co_occ_percentage", title = "Procent co-aparitie (%)", width = 75))
    columns.append(TableColumn(field = "distance", title = "Distanta", width = 75))

    self.table_topk = DataTable(source = self.source_topk, columns = columns, width = 600)
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
    
    table_data_dict = dict(ID = data['ITEM_ID'],
                           name = data['ITEM_NAME'],
                           distance = data['DIST'].apply(lambda x: round(x, 4)))
    if self.has_co_occ_data:
      table_data_dict['co_occ_percentage'] = data['CO_OCC'].apply(lambda x: round(x, 2))
    
    self.source_topk.data = table_data_dict
    self.table_topk.height = 250

  def update(self, df, prod_id, found_id):
    if found_id:
      prod_x = df.iloc[0]['x_new']
      prod_y = df.iloc[0]['y_new']
      
      self.p.title.text = "Cei mai apropiati %d vecini de produsul selectat - %s" % (len(df)-1, df.iloc[0]['ITEM_NAME'])
      self.bottom_title_search.text = df.iloc[0]['ITEM_NAME']
      self.p.outline_line_color = df.iloc[0]['COLOR']
      
      columns = ['ITEM_ID', 'ITEM_NAME', 'DIST', 'x_new', 'y_new']
      if self.has_co_occ_data:
        columns.append('CO_OCC')

      self.populate_table(df[1:][columns])
      
      arrow = Arrow(end=VeeHead(size=10), line_color='orange', line_width=3,
             x_start=prod_x - 10, y_start=prod_y, x_end=prod_x, y_end=prod_y)
      
      self.annotations[1] = arrow
      self.p.renderers.extend([arrow])
      
      self.source.data = dict(
        x = df["x_new"],
        y = df["y_new"],
        color = df["COLOR"],
        name = df["ITEM_NAME"],
        old_id = df["ITEM_ID"],
        dist = df["DIST"],
        short_name = df.ITEM_NAME.str.slice(0,20)
      )
    else:
      self.p.title.text = "Produsul cu ID %d nu a fost gasit" % prod_id
      self.bottom_title_search.text = ""
      self.p.outline_line_color = "gainsboro"
      self.table_topk.height = 0

class Server:
  def __init__(self, config_file = 'config.txt', suffix = ''):
    
    start = time()
    self.CONFIG = None
    logger_module = load_module('logger', 'logger.py')
    self.logger = logger_module.Logger(lib_name = "RECOMv3",
                                config_file = config_file,
                                log_suffix = suffix,
                                TF_KERAS = False,
                                HTML = True)
    
    self.CONFIG = self.logger.config_data
    self._base_folder = self.logger.GetBaseFolder()
    self._log('Initializing server ...')
    
    data_path = os.path.join(self._base_folder, self.CONFIG['DATA_FOLDER'])
    self.load_mco = (self.CONFIG["HAS_MCO"] == 1)
    self._log('  Loading data from [...{}]'.format(data_path[-20:]))
    self.data_collector = DataCollector(data_path, self.load_mco)
    self._log("  Finished loading data.", show_time = True)

    title_text = "<h1>Explorator interactiv al modelului de recomandari ce detecteaza complementaritatea produselor SSB</h1>"
    div_text = """
    <p align='justify'><b>Pentru interactiunea cu aplicatia, se va folosi TextBox-ul ce se afla mai sus pentru a va fi returnat 
    un tabel cu toate ID-urile produselor al caror nume contin sirul introdus de la tastatura.
    Cautarea produsului pe harta se face prin selectarea liniei corespunzatoare din tabel.<br><br> 
    Aplicatia afiseaza 2 harti: prima surprinde modul in care sunt grupate primele 10,000 produse, iar cea de-a doua va afisa 
    top 100 produse apropiate de produsul selectat.
    </b></p>"""
    self.choice_text = "<p align='justify'>Ati ales produsul cu ID <b>{}</b> (<b>{}</b>)</p>"

    self.maps = []
    self.topk_plots = []
    self.title = Div(text = title_text, width = 440)
    self.descript = Div(text = div_text, width = 440)
    self.choice_div = Div(text = "Nu ati ales inca niciun produs", width = 440)
    self.product_id = TextInput()
    self.product_name = TextInput(title = "Introduceti numele produsului cautat")
    self.possibilities_descript = PreText(text = '', width = 460)

    self.show_topk_button = Button(label = "Gasire complementaritate", button_type = "success")
    self.show_topk_button.on_click(self.update_topk_plots)
    self.show_topk_button.disabled = True
    
    self.show_topk_button_mco = Button(label = "Afisare comportament de cumparare",
                                       button_type = "success")
    self.show_topk_button_mco.on_click(self.populate_mco_table)
    self.show_topk_button_mco.disabled = True
    

    self.product_id.on_change('value', lambda attr, old, new: self.update_maps())
    self.product_name.on_change('value', lambda attr, old, new: self.show_possibilities())
    self.panels = []

    self.create_possibilities_fields()
    self.create_mco_table()
    self._log("Initialized the server. [{:.2f}s]".format(time() - start))

  def create_plot(self, plot_type, plot_height, plot_width):
    if plot_type not in ['map', 'top_k']:
      self._log("Error! plot_type not in ['map', 'top_k']")
      raise Exception("Error! plot_type not in ['map', 'top_k']")

    self._log('Creating {} plot; H,W=({}, {}) ...'.format(plot_type, plot_height, plot_width))

    if plot_type == 'map':
      self.maps.append(Map(plot_height = plot_height, plot_width = plot_width))
    elif plot_type == 'top_k':
      self.topk_plots.append(TopKPlot(plot_height = plot_height,
                                      plot_width = plot_width,
                                      has_co_occ_data = self.load_mco))

    self._log('Created {} plot.'.format(plot_type), show_time = True)


  def create_possibilities_fields(self):
    self._log('  Creating the table which will show all the possibilities of products ...')
    self.df_possibilities = None
    self.source_possibilities = ColumnDataSource(data=dict(ID=[], name=[]))
    columns = [
        TableColumn(field="ID", title="ID Produs", width=75),
        TableColumn(field="name", title="Nume Produs"),
    ]
    self.table_possibilities = DataTable(source=self.source_possibilities, columns=columns, width=440, height=350)
    self.table_possibilities.height = 0
    
    self.source_possibilities.on_change('selected', self.select_data_entry_callback)
    self._log('  Created the table.', show_time = True)
    return

  def create_mco_table(self):
    self._log('  Creating the table which will show the shopping pattern ...')
    
    self.source_mco_table = ColumnDataSource(data = dict(ID = [], name = [], co_occ_percentage = []))
    columns = [
        TableColumn(field = "ID", title = "ID Produs", width = 75),
        TableColumn(field = "name", title = "Nume Produs"),
        TableColumn(field = "co_occ_percentage", title = "Procent co-aparitie (%)", width = 75)
    ]

    self.mco_table = DataTable(source = self.source_mco_table, columns = columns, width = 440)
    self.mco_table.height = 0
    self._log('  Created the table.', show_time = True)
    return
    
  def select_data_entry_callback(self, attr, old, new):
    ID = new['1d']['indices'][0]
    self.product_id.value = str(self.df_possibilities.iloc[ID].ITEM_ID)


  def create_layout(self):
    self._log('Creating the layout of the server ...')
    layout_tab1 = row(column(self.title,
                             self.product_name,
                             self.possibilities_descript,
                             widgetbox(self.table_possibilities),
                             self.descript),
                      self.maps[1].p)
    
    layout_tab2 = column(self.maps[0].p)
     
    layout_tab3 = column(self.choice_div,
                         row(widgetbox(self.show_topk_button),
                             widgetbox(self.show_topk_button_mco)),
                         row(self.topk_plots[0].button_group_labels,
                             self.topk_plots[0].button_group_arrows),
                         row(self.topk_plots[0].p,
                             column(self.topk_plots[0].table_topk,
                                    self.mco_table)))

    
    self.panels.append(Panel(title = 'Cautare produs', child = layout_tab1))
    self.panels.append(Panel(title = 'Vizualizare harta', child = layout_tab2))
    self.panels.append(Panel(title = 'Top prod. complementare', child = layout_tab3))
    self._log('Created the layout of the server.', show_time = True)
    
    return Tabs(tabs = self.panels)
  
  
  def draw_products(self):
    self._log('Drawing all the maps which we created ...')
    for m in self.maps:
      df_prods = self.data_collector.df_prods
      m.p.xaxis.axis_label = "t-SNE-x"
      m.p.yaxis.axis_label = "t-SNE-y"
      m.p.title.text = "[Prod2Vec] Harta produselor SSB creata pe baza complementaritatii lor"
      m.source.data = dict(
          x = df_prods["x"],
          y = df_prods["y"],
          color = df_prods["COLOR"],
          name = df_prods["ITEM_NAME"],
          old_id = df_prods["ITEM_ID"]
      )
    self._log('Finished drawing all the maps.', show_time = True)


  def show_possibilities(self):
    prod_name_val = self.product_name.value.strip()
    self._log('Filling the table which shows all the possibilities of products, given [{}] ...'.format(prod_name_val))
    
    df_prods = self.data_collector.df_prods
    self.df_possibilities = df_prods[df_prods.ITEM_NAME.str.lower().str.contains(prod_name_val.lower()) == True]
    nr_possibilities = len(self.df_possibilities)
    if nr_possibilities == 0:
      self.possibilities_descript.text = "Numele introdus nu corespunde niciunui produs"
      self.choice_div.text = "Nu am gasit niciun produs care sa corespunda cu ceea ce ati introdus"
      self.table_possibilities.height = 0
      self.show_topk_button.disabled = True
      self.show_topk_button_mco.disabled = True
    else:
      self.possibilities_descript.text = "Rezultatul cautarii:"
      self.source_possibilities.data = dict(
          ID = self.df_possibilities['ITEM_ID'],
          name = self.df_possibilities['ITEM_NAME']
      )
      self.table_possibilities.height = 250
      if nr_possibilities <= 6:
        self.table_possibilities.height = 150

      if nr_possibilities <= 3:
        self.table_possibilities.height = 100

      self.show_topk_button.disabled = False
      self.show_topk_button_mco.disabled = False
    self._log('Finished filling the table.', show_time = True)
  

  def update_maps(self):
    prod_id = self.product_id.value.strip()
    self._log('Updating all the maps, gived product ID = {} ...'.format(prod_id))

    self._log('  Making annotations invisible ...')
    for m in self.maps:
      for annotation in m.annotations:
        if annotation is not None:
          annotation.visible = False
    self._log('  Finished making annotations invisible.', show_time = True)
    
    if prod_id != "":
      self._log('  Creating new annotations ...')
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
          
          # Update choice_div on the 3rd tab whenever we select a new product in the selection table
          self.choice_div.text = self.choice_text.format(prod_id, selected.iloc[0]['ITEM_NAME']) 
      self._log('  Finished creating new annotations', show_time = True)
    self._log('Updated all the maps.')


  def tsne(self, embeddings):
    self._log("    Starting TSNE algorithm for {} embeddings ...".format(embeddings.shape[0]))
    tsne = TSNE(perplexity = 30, n_components = 2, init = 'pca', n_iter = 1000, random_state = 42)    
    low_dim_embs = tsne.fit_transform(embeddings)    
    self._log("    Finished TSNE algorithm.", show_time = True)
    return low_dim_embs
  

  def select_products(self, prod_id):
    data = self.data_collector

    top_k_indexes = data.top_k_indexes
    top_k_distances = data.top_k_distances
    emb = np.array(data.norm_embeddings)

    selected = pd.DataFrame(columns = list(data.df_prods.columns) + ['DIST'])
    found_id = False
    if prod_id != "":
      selected = data.df_prods[data.df_prods.ITEM_ID == int(prod_id)]
      if len(selected) == 0:
        selected = selected.assign(DIST = np.nan)
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
      
      if self.load_mco:
        co_occ_scores = data.mco[new_id + 1]
        div_sum = co_occ_scores.sum()
        co_occ_scores = co_occ_scores[k_indexes + 1]
        co_occ_scores = np.insert(co_occ_scores, 0, 0)
        co_occ_scores = (co_occ_scores / div_sum) * 100
        selected['CO_OCC'] = pd.Series(co_occ_scores)

      #selected['x_new'] = selected['x']
      #selected['y_new'] = selected['y']
      found_id = True

    return selected, int(prod_id), found_id
  
  def populate_mco_table(self):
    if not self.load_mco:
      return
    
    prod_id = self.product_id.value.strip()
    if prod_id == "":
      self._log("No product ID given as input. The mco table won't be populated")
      return
    
    self._log('Updating the MCO table, given product ID = {} ...'.format(prod_id))
    data = self.data_collector
    selected = pd.DataFrame(columns = list(data.df_prods.columns) + ['CO_OCC'])
    selected = data.df_prods[data.df_prods.ITEM_ID == int(prod_id)]
    if len(selected) == 0:
        self._log("  No entry found with given product ID.")
        return
    
    new_id = selected.iloc[0]['NEW_ID'] - 1
    co_occ_scores = data.mco[new_id + 1]
    div_sum = co_occ_scores.sum()
    k_indexes = np.argsort(co_occ_scores)[::-1][:100]
    k_distances = co_occ_scores[k_indexes]
    k_distances = np.insert(k_distances, 0, 0)
    k_distances = (k_distances / div_sum) * 100
    
    k_products = data.df_prods.iloc[k_indexes - 1]
    selected = selected.append(k_products, ignore_index=True)
    selected['CO_OCC'] = pd.Series(k_distances)
    #print(selected)
    
    selected = selected.loc[1:, :]
    table_data_dict = dict(ID = selected['ITEM_ID'],
                           name = selected['ITEM_NAME'],
                           co_occ_percentage = selected['CO_OCC'].apply(lambda x: round(x, 2)))
    
    self.source_mco_table.data = table_data_dict
    self.mco_table.height = 250
    
    
    self._log('Updated the MCO table.', show_time = True)

  def update_topk_plots(self):
    prod_id = self.product_id.value.strip()
    self._log('Updating all the top_k plots, given product ID = {} ...'.format(prod_id))
    self._log('  Selecting top_k complementar neighbors ...')
    start = time()
    df, prod_id, found_id = self.select_products(prod_id)
    self._log('  Selected top_k complementar neighbors. [{:.2f}s]'.format(time() - start))
    
    for i, topk_plot in enumerate(self.topk_plots):
      self._log('  Making annotations invisible for topk_plot #{}'.format(i+1))
      for annotation in topk_plot.annotations:
        if annotation is not None:
          annotation.visible = False
      self._log('  Finished making annotations invisible.', show_time = True)
      
      self._log('  Updating topk_plot #{} and its associated table ...'.format(i+1))
      topk_plot.update(df, prod_id, found_id)
      self._log('  Finished updating topk_plot and its associated table.', show_time = True)
    self._log('Updated all the top_k plots.')

  def _log(self, str_msg, results = False, show_time = False):
    self.logger.VerboseLog(str_msg, results, show_time)
    return