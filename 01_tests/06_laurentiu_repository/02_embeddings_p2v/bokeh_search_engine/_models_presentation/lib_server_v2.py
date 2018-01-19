import pandas as pd
import numpy as np
import os
from time import time
from scipy.sparse import load_npz
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE  
import multiprocessing
from functools import partial

from bokeh.plotting import figure
from bokeh.layouts import widgetbox, column, row
from bokeh.models import ColumnDataSource, HoverTool, Div,\
  BoxZoomTool, ResetTool, WheelZoomTool, PanTool, BoxAnnotation,\
  Arrow, VeeHead, Title, LabelSet
from bokeh.models.widgets import TextInput, PreText, TableColumn,DataTable, Panel,\
  Tabs, RadioButtonGroup, Button


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


class DataLoader:
  def __init__(self, model_file, mco_file, emb_slice_start_field, emb_slice_end_field,
               cluster_field, color_field, keep_products):
    try:
      self.df_prods = pd.read_csv(model_file, encoding = 'ISO-8859-1')
      self.mco = load_npz(mco_file) # it is saved as a csr_matrix
    except OSError as e:
      print('[...{}] or [...{}] not found'.format(model_file[-20:], mco_file[-20:]))
    
    self.clusters = np.array(self.df_prods[cluster_field].tolist())
    colors = list()
    for key in self.clusters:
      colors.append(dict_colors[key])
    self.df_prods[color_field] = pd.Series(colors)
    self.df_prods = self.df_prods[:keep_products]

    self.norm_embeddings = np.array(self.df_prods.loc[:, emb_slice_start_field:emb_slice_end_field])
    return

class Map:
  def __init__(self, height, width):
    hover = HoverTool(tooltips=[
      ("Nume", "@name"),
      ("ID", "@old_id")
    ])

    self.source = ColumnDataSource(data = dict(x = [], y = [], color = [], name = [], old_id = []))
    self.p = figure(plot_height = height, plot_width = width, title = "",
                    tools = [hover, BoxZoomTool(), WheelZoomTool(), ResetTool(), PanTool()])
    self.p.scatter(x = "x", y = "y", source = self.source, radius = 0.25,
                   color = "color", line_color = None)
    self.p.outline_line_width = 7
    self.p.outline_line_alpha = 0.3
    self.annotations = [None, None]
    return


class TopKPlot:
  def __init__(self, height, width, item_id_field, item_name_field, color_field):
    hover = HoverTool(tooltips=[
      ("Nume", "@name"),
      ("ID", "@old_id"),
      ("Distance", "@dist{0,0.000}")
    ])
    
    self.source = ColumnDataSource(data=dict(x = [], y = [], color = [], name = [], old_id = [],
                                             dist = [], short_name = []))
    self.p = figure(#plot_height = height, plot_width = width,
                    title = "Fereastra in care se vor afisa rezultatele cautarii",
                    tools = [hover, BoxZoomTool(), WheelZoomTool(), ResetTool(), PanTool()])
    self.p.scatter(x = "x", y = "y", source = self.source, size = 8, color = "color", line_color = None)
    self.p.outline_line_width = 7
    self.p.outline_line_alpha = 0.3
    
    self.bottom_title_search = Title(text = "-", align = "center")
    self.labels = LabelSet(x = 'x', y = 'y', text = 'short_name', level = 'glyph',
                           x_offset = -30, y_offset = 5, source = self.source, render_mode = 'canvas',
                           text_font_size = "5pt")
    self.labels.visible = False
    self.p.add_layout(self.bottom_title_search, "below")
    self.p.add_layout(self.labels)
    self.annotations = [None, None]
    
    self.button_group_labels = RadioButtonGroup(labels=["Ascundere nume", "Afisare nume"], active=0)
    self.button_group_labels.on_change('active', lambda attr, old, new: self._change_labels_visibility_callback())
    
    self.button_group_arrows = RadioButtonGroup(labels=["Ascundere indicatori", "Afisare indicatori"], active=1)
    self.button_group_arrows.on_change('active', lambda attr, old, new: self._change_arrows_visibility_callback())
    
    # Create ColumnDataSource
    table_data_dict = dict(ID = [], name = [], distance = [], co_occ = [])
    self.source_table_topk = ColumnDataSource(data = table_data_dict)

    # Create Table Columns
    columns = [
        TableColumn(field = "ID", title = "ID Produs", width = 75),
        TableColumn(field = "name", title = "Nume Produs"),
        TableColumn(field = "co_occ", title = "Procent co-aparitie (%)", width = 75),
        TableColumn(field = "distance", title = "Distanta", width = 75)
    ]

    self.table_topk = DataTable(source = self.source_table_topk, columns = columns, width = 600)
    self.table_topk.height = 0
    self.source_table_topk.on_change('selected', self._select_neighbor_callback)
    
    self.item_id_field = item_id_field
    self.item_name_field = item_name_field
    self.color_field = color_field
    return


  def _change_labels_visibility_callback(self):
    self.labels.visible = (self.button_group_labels.active == 1)
    return


  def _change_arrows_visibility_callback(self):
    for annotation in self.annotations:
      if annotation is not None:
          annotation.visible = (self.button_group_arrows.active == 1)
          
    return

    
  def _select_neighbor_callback(self, attr, old, new):
    if self.annotations[0] is not None:
      self.annotations[0].visible = False
    
    ID = new['1d']['indices'][0]
    prod_x = self.topk.iloc[ID]['x_new'] #### TODOO
    prod_y = self.topk.iloc[ID]['y_new']
    
    arrow = Arrow(end = VeeHead(size = 10),
                  line_color = 'black', line_width = 3,
                  x_start = prod_x - 10, y_start = prod_y + 10,
                  x_end = prod_x, y_end = prod_y)

    self.annotations[0] = arrow
    self.p.renderers.extend([arrow])


  def _populate_table(self, data):
    self.topk = data
    table_data_dict = dict(ID = data[self.item_id_field],
                           name = data[self.item_name_field],
                           distance = data['DIST'].apply(lambda x: round(x, 4)),
                           co_occ = data['CO_OCC'].apply(lambda x: round(x, 2)))

    self.source_table_topk.data = table_data_dict
    self.table_topk.height = 550


  def Update(self, df, prod_id, found_id):
    if found_id:
      prod_x = df.iloc[0]['x_new']
      prod_y = df.iloc[0]['y_new']
      
      self.p.title.text = "Cei mai apropiati %d vecini de produsul selectat - %s" % (len(df)-1, df.iloc[0][self.item_name_field])
      self.bottom_title_search.text = df.iloc[0][self.item_name_field]
      self.p.outline_line_color = df.iloc[0][self.color_field]
      
      keep_columns = [self.item_id_field, self.item_name_field, 'DIST', 'x_new', 'y_new', 'CO_OCC']
      self._populate_table(df[1:][keep_columns])
      
      arrow = Arrow(end = VeeHead(size = 10),
                    line_color = 'orange', line_width = 3,
                    x_start = prod_x - 10, y_start = prod_y,
                    x_end = prod_x, y_end = prod_y)
  
      self.annotations[1] = arrow
      self.p.renderers.extend([arrow])
      
      self.source.data = dict(
        x = df["x_new"],
        y = df["y_new"],
        color = df[self.color_field],
        name = df[self.item_name_field],
        old_id = df[self.item_id_field],
        dist = df["DIST"],
        short_name = df[self.item_name_field].str.slice(0,20)
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
    
    loaded_model = os.path.join(self._base_folder, self.CONFIG['MODEL'])
    
    assert self.CONFIG['MCO'] != ""
    mco_file = os.path.join(self._base_folder, self.CONFIG['MCO'])

    self._log('  Loading data from [...{}]'.format(loaded_model[-40:]))
    self.data_collector = DataLoader(model_file = loaded_model,
                                     mco_file = mco_file,
                                     emb_slice_start_field = self.CONFIG["EMB_SLICE_START_FIELD"],
                                     emb_slice_end_field = self.CONFIG["EMB_SLICE_END_FIELD"],
                                     cluster_field = self.CONFIG["CLUSTER_FIELD"],
                                     color_field = self.CONFIG["COLOR_FIELD"],
                                     keep_products = self.CONFIG["KEEP_PRODUCTS"])
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
    self.table_possibilities = None

    self.product_id.on_change('value', lambda attr, old, new: self._update_maps())
    self.product_name.on_change('value', lambda attr, old, new: self._show_item_possibilities())
    self.panels = []

    self._create_list_item_possibilities()
    
    #Buttons that are responsible with changing each list from Tab #3
    self.gen_top_dist_button = Button(label = "Gasire complementaritate", button_type = "success")
    self.gen_top_dist_button.on_click(partial(self._update_topk_plot, idx = 0))
    self.gen_top_dist_button.disabled = True
    
    self.gen_top_mco_button = Button(label = "Afisare comportament de cumparare", button_type = "success")
    self.gen_top_mco_button.on_click(partial(self._update_topk_plot, idx = 1))
    self.gen_top_mco_button.disabled = True
    
    self.gen_top_clust_button = Button(label = "Excludere clustere", button_type = "success")
    self.gen_top_clust_button.on_click(partial(self._update_topk_plot, idx = 2))
    self.gen_top_clust_button.disabled = True
    
    
    self._log("Initialized the server. [{:.2f}s]".format(time() - start))
    return

  def CreatePlot(self, plot_type, plot_height, plot_width):
    if plot_type not in ['map', 'topk']:
      self._log("Error! plot_type not in ['map', 'topk']")
      raise Exception("Error! plot_type not in ['map', 'topk']")

    self._log('Creating {} plot; H,W=({}, {}) ...'.format(plot_type, plot_height, plot_width))

    if plot_type == 'map':
      self.maps.append(Map(height = plot_height, width = plot_width))
    elif plot_type == 'topk':
      self.topk_plots.append(TopKPlot(height = plot_height,
                                      width = plot_width,
                                      item_id_field = self.CONFIG["ITEM_ID_FIELD"],
                                      item_name_field = self.CONFIG["ITEM_NAME_FIELD"],
                                      color_field = self.CONFIG["COLOR_FIELD"]))

    self._log('Created {} plot.'.format(plot_type), show_time = True)
  
  
  def _update_maps(self):
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
        selected = data.df_prods[data.df_prods[self.CONFIG["ITEM_ID_FIELD"]] == int(prod_id)]
        if len(selected) != 0:
          prod_x = selected.iloc[0][self.CONFIG["TSNE2D_X_FIELD"]]
          prod_y = selected.iloc[0][self.CONFIG["TSNE2D_Y_FIELD"]]
          arrow = Arrow(end = VeeHead(size = 10),
                        line_color = 'black',
                        line_width = 3,
                        x_start = prod_x - 8, y_start = prod_y + 20,
                        x_end = prod_x, y_end = prod_y)
          box = BoxAnnotation(plot = m.p,
                              left = prod_x - 12, right = prod_x + 12,
                              top = prod_y - 12, bottom = prod_y + 12,
                              fill_alpha = 0.4,
                              fill_color = 'green')
          m.annotations[0] = box
          m.annotations[1] = arrow
          m.p.renderers.extend([box, arrow])
          
          # Update choice_div on the 3rd tab whenever we select a new product in the selection table
          self.choice_div.text = self.choice_text.format(prod_id,
                                                         selected.iloc[0][self.CONFIG['ITEM_NAME_FIELD']])
      self._log('  Finished creating new annotations', show_time = True)
    self._log('Updated all the maps.')
    return


  def CreateLayout(self):
    self._log('Creating the layout of the server ...')
    assert len(self.topk_plots) == 3
    
    layout_tab1 = row(column(self.title,
                             self.product_name,
                             self.possibilities_descript,
                             widgetbox(self.table_possibilities),
                             self.descript),
                      self.maps[1].p)
    
    layout_tab2 = column(self.maps[0].p)
    
    layout_tab3 = column(self.choice_div,
                         widgetbox(self.gen_top_dist_button),
                         row(self.topk_plots[0].button_group_labels,
                             self.topk_plots[0].button_group_arrows),
                         row(self.topk_plots[0].p,
                             self.topk_plots[0].table_topk))

    layout_tab4 = column(widgetbox(self.gen_top_mco_button),
                         row(self.topk_plots[1].button_group_labels,
                             self.topk_plots[1].button_group_arrows),
                         row(self.topk_plots[1].p,
                             self.topk_plots[1].table_topk))
                         
    layout_tab5 = column(widgetbox(self.gen_top_clust_button),
                         row(self.topk_plots[2].button_group_labels,
                             self.topk_plots[2].button_group_arrows),
                         row(self.topk_plots[2].p,
                             self.topk_plots[2].table_topk))
    
    self.panels.append(Panel(title = 'Cautare produs', child = layout_tab1))
    self.panels.append(Panel(title = 'Vizualizare harta', child = layout_tab2))
    self.panels.append(Panel(title = 'Complementaritate', child = layout_tab3))
    self.panels.append(Panel(title = 'Comportament de cumparare', child = layout_tab4))
    self.panels.append(Panel(title = 'Clustere de produse', child = layout_tab5))
    self._log('Created the layout of the server.', show_time = True)
    
    return Tabs(tabs = self.panels)
  
  
  def DrawProducts(self):
    self._log('Drawing all the maps which we created ...')
    for m in self.maps:
      df_prods = self.data_collector.df_prods
      m.p.xaxis.axis_label = "t-SNE-x"
      m.p.yaxis.axis_label = "t-SNE-y"
      m.p.title.text = "[Prod2Vec] Harta produselor SSB creata pe baza complementaritatii lor"
      m.source.data = dict(
          x = df_prods[self.CONFIG["TSNE2D_X_FIELD"]],
          y = df_prods[self.CONFIG["TSNE2D_Y_FIELD"]],
          color = df_prods[self.CONFIG["COLOR_FIELD"]],
          name = df_prods[self.CONFIG["ITEM_NAME_FIELD"]],
          old_id = df_prods[self.CONFIG["ITEM_ID_FIELD"]]
      )
    self._log('Finished drawing all the maps.', show_time = True)
    return
  
  
  def _create_list_item_possibilities(self):
    self._log('  Creating the table which will show all the possibilities of products ...')
    self.df_possibilities = None
    self.source_possibilities = ColumnDataSource(data = dict(ID = [], name = []))
    columns = [
        TableColumn(field = "ID", title = "ID Produs", width = 75),
        TableColumn(field = "name", title = "Nume Produs"),
    ]
    self.table_possibilities = DataTable(source = self.source_possibilities,
                                         columns = columns,
                                         width = 440,
                                         height = 350)
    self.table_possibilities.height = 0
    
    self.source_possibilities.on_change('selected', self.__select_item_possibility_callback)
    self._log('  Created the table.', show_time = True)
    return
  
  
  def __select_item_possibility_callback(self, attr, old, new):
    ID = new['1d']['indices'][0]
    self.product_id.value = str(self.df_possibilities.iloc[ID][self.CONFIG["ITEM_ID_FIELD"]])
    return

  
  
  def _show_item_possibilities(self):
    prod_name_val = self.product_name.value.strip()
    self._log('Filling the table which shows all the possibilities of products, given [{}] ...'.format(prod_name_val))
    
    df_prods = self.data_collector.df_prods
    self.df_possibilities = df_prods[df_prods[self.CONFIG["ITEM_NAME_FIELD"]].str.lower().str.contains(prod_name_val.lower()) == True]
    nr_possibilities = len(self.df_possibilities)
    if nr_possibilities == 0:
      self.possibilities_descript.text = "Numele introdus nu corespunde niciunui produs"
      self.choice_div.text = "Nu am gasit niciun produs care sa corespunda cu ceea ce ati introdus"
      self.table_possibilities.height = 0
      self.__toggle_lists_generator_buttons(disabled = True)
    else:
      self.possibilities_descript.text = "Rezultatul cautarii:"
      self.source_possibilities.data = dict(
          ID = self.df_possibilities[self.CONFIG['ITEM_ID_FIELD']],
          name = self.df_possibilities[self.CONFIG['ITEM_NAME_FIELD']]
      )
      self.__toggle_lists_generator_buttons(disabled = False)
      self.table_possibilities.height = 250
      if nr_possibilities <= 6:
        self.table_possibilities.height = 150

      if nr_possibilities <= 3:
        self.table_possibilities.height = 100
  
    self._log('Finished filling the table.', show_time = True)
    return


  def __toggle_lists_generator_buttons(self, disabled):
    self.gen_top_dist_button.disabled = disabled
    self.gen_top_mco_button.disabled = disabled
    self.gen_top_clust_button.disabled = disabled
    return
  
  
  
  """
    Option #1: top k products sorted by cosine similarity
    Option #2: top k products sorted by co occurrence
    Option #3: top k products, excluding certain clusters
  """
  
  def __tsne(self, embeddings):
    self._log("    Starting TSNE algorithm for {} embeddings ...".format(embeddings.shape[0]))
    tsne = TSNE(perplexity = 30, n_components = 2, init = 'pca', n_iter = 1000, random_state = 42)    
    low_dim_embs = tsne.fit_transform(embeddings)    
    self._log("    Finished TSNE algorithm.", show_time = True)
    return low_dim_embs
  
  def __select_k_products_by_cosine(self, prod_id):
    data = self.data_collector
    selected = pd.DataFrame(columns = list(data.df_prods.columns) + ['DIST', 'CO_OCC'])
    
    found_id = False
    if prod_id != "":
      selected = data.df_prods[data.df_prods[self.CONFIG["ITEM_ID_FIELD"]] == int(prod_id)]
      if len(selected) == 0:
        selected = selected.assign(DIST = np.nan)
        return selected, int(prod_id), False
      
      found_id = True
      new_id = selected.iloc[0][self.CONFIG["IDE_FIELD"]] - 1
      
      self._log("    Computing cosine distances between the selected product [{}] and the other {} products ..."
                .format(selected.iloc[0][self.CONFIG["ITEM_ID_FIELD"]], self.CONFIG["KEEP_PRODUCTS"]))
      distances = pairwise_distances(data.norm_embeddings[new_id].reshape(1, -1),
                                     data.norm_embeddings,
                                     metric = 'cosine',
                                     n_jobs = multiprocessing.cpu_count())
      distances = distances.flatten()

      K = 100  ## this should be inserted manually from the interface
      top_k_indexes = np.argsort(distances)[1 : (K + 1)]
      top_k_distances = distances[top_k_indexes]
      top_k_distances = np.insert(top_k_distances, 0, 0)
      self._log("    Computed cosine distances.", show_time = True)
      
      k_embeddings = data.norm_embeddings[top_k_indexes]
      k_embeddings = np.vstack([k_embeddings, data.norm_embeddings[new_id]])
      k_low_dim_embs = self.__tsne(k_embeddings)
      
      k_products = data.df_prods.iloc[top_k_indexes]
      selected = selected.append(k_products, ignore_index=True)
      selected['DIST'] = pd.Series(top_k_distances)
      selected['x_new'] = pd.Series(k_low_dim_embs[:, 0])
      selected['y_new'] = pd.Series(k_low_dim_embs[:, 1])
      
      co_occ_scores = data.mco[new_id + 1]
      div_sum = co_occ_scores.sum()
      
      list_scores = []
      for idx in (top_k_indexes + 1):
        list_scores.append(co_occ_scores[0, idx])
      co_occ_scores = np.array(list_scores)
      co_occ_scores = np.insert(co_occ_scores, 0, 0)
      co_occ_scores = (co_occ_scores / div_sum) * 100
      selected['CO_OCC'] = pd.Series(co_occ_scores)

      return selected, int(prod_id), found_id
  
  def __select_k_products_by_mco(self, prod_id):
    data = self.data_collector
    selected = pd.DataFrame(columns = list(data.df_prods.columns) + ['DIST', 'CO_OCC'])
    
    found_id = False
    if prod_id != "":
      selected = data.df_prods[data.df_prods[self.CONFIG["ITEM_ID_FIELD"]] == int(prod_id)]
      if len(selected) == 0:
        selected = selected.assign(DIST = np.nan)
        return selected, int(prod_id), False
      
      found_id = True
      new_id = selected.iloc[0][self.CONFIG["IDE_FIELD"]] - 1
      
      self._log("    Extracting top k mco for the selected product [{}] ..."
                .format(selected.iloc[0][self.CONFIG["ITEM_ID_FIELD"]]))
      co_occ_scores = data.mco[new_id + 1]
      div_sum = co_occ_scores.sum()
      
      list_scores = []
      for idx in range(1, co_occ_scores.shape[1]):
        list_scores.append(co_occ_scores[0, idx])
      co_occ_scores = np.array(list_scores)
      
      K = 100
      top_k_indexes = np.argsort(co_occ_scores)[::-1][:K]
      co_occ_scores = co_occ_scores[top_k_indexes]
      co_occ_scores = np.insert(co_occ_scores, 0, 0)
      co_occ_scores = (co_occ_scores / div_sum) * 100
      self._log("    Extracted top k mco.", show_time = True)
  
      k_products = data.df_prods.iloc[top_k_indexes]
      selected = selected.append(k_products, ignore_index = True)
      selected['CO_OCC'] = pd.Series(co_occ_scores)
      
      k_embeddings = data.norm_embeddings[top_k_indexes]
      k_embeddings = np.vstack([k_embeddings, data.norm_embeddings[new_id]])
      k_low_dim_embs = self.__tsne(k_embeddings)
      selected['x_new'] = pd.Series(k_low_dim_embs[:, 0])
      selected['y_new'] = pd.Series(k_low_dim_embs[:, 1])
      
      self._log("    Computing cosine distances between the selected product [{}] and the other {} products ..."
                .format(selected.iloc[0][self.CONFIG["ITEM_ID_FIELD"]], self.CONFIG["KEEP_PRODUCTS"]))
      distances = pairwise_distances(data.norm_embeddings[new_id].reshape(1, -1),
                                     data.norm_embeddings,
                                     metric = 'cosine',
                                     n_jobs = multiprocessing.cpu_count())
      distances = distances.flatten()
      self._log("    Computed cosine distances.", show_time = True)
      
      distances = distances[top_k_indexes]
      distances = np.insert(distances, 0, 0)
      selected['DIST'] = pd.Series(distances)

      return selected, int(prod_id), found_id
  
  def __select_k_products_by_clusters(self, prod_id):
    data = self.data_collector
    selected = pd.DataFrame(columns = list(data.df_prods.columns) + ['DIST', 'CO_OCC'])
    
    found_id = False
    if prod_id != "":
      selected = data.df_prods[data.df_prods[self.CONFIG["ITEM_ID_FIELD"]] == int(prod_id)]
      if len(selected) == 0:
        selected = selected.assign(DIST = np.nan)
        return selected, int(prod_id), False
      
      found_id = True
      new_id = selected.iloc[0][self.CONFIG["IDE_FIELD"]] - 1
      cluster_to_remove = selected.iloc[0][self.CONFIG["CLUSTER_FIELD"]]
      
      self._log("    Computing cosine distances between the selected product [{}] and the other {} products ..."
                .format(selected.iloc[0][self.CONFIG["ITEM_ID_FIELD"]], self.CONFIG["KEEP_PRODUCTS"]))
      distances = pairwise_distances(data.norm_embeddings[new_id].reshape(1, -1),
                                     data.norm_embeddings,
                                     metric = 'cosine',
                                     n_jobs = multiprocessing.cpu_count())
      distances = distances.flatten()

      K = 500  ## this should be inserted manually from the interface
      top_k_indexes = np.argsort(distances)[1 : (K + 1)]
      top_k_distances = distances[top_k_indexes]
      top_k_distances = np.insert(top_k_distances, 0, 0)
      self._log("    Computed cosine distances.", show_time = True)
      
      k_embeddings = data.norm_embeddings[top_k_indexes]
      k_embeddings = np.vstack([k_embeddings, data.norm_embeddings[new_id]])
      k_low_dim_embs = self.__tsne(k_embeddings)
      
      k_products = data.df_prods.iloc[top_k_indexes]
      selected = selected.append(k_products, ignore_index=True)
      selected['DIST'] = pd.Series(top_k_distances)
      selected['x_new'] = pd.Series(k_low_dim_embs[:, 0])
      selected['y_new'] = pd.Series(k_low_dim_embs[:, 1])
      
      co_occ_scores = data.mco[new_id + 1]
      div_sum = co_occ_scores.sum()
      
      list_scores = []
      for idx in (top_k_indexes + 1):
        list_scores.append(co_occ_scores[0, idx])
      co_occ_scores = np.array(list_scores)
      co_occ_scores = np.insert(co_occ_scores, 0, 0)
      co_occ_scores = (co_occ_scores / div_sum) * 100
      selected['CO_OCC'] = pd.Series(co_occ_scores)
      
      selected = selected[(selected[self.CONFIG["CLUSTER_FIELD"]] != cluster_to_remove) |\
                          (selected[self.CONFIG["IDE_FIELD"]] == (new_id + 1))]
      selected.reset_index(drop = True, inplace = True)
      selected = selected[:101]

      return selected, int(prod_id), found_id
  
  
  def _update_topk_plot(self, idx):
    prod_id = self.product_id.value.strip()
    self._log('Updating top_k plot with idx = {}, given product ID = {} ...'
              .format(idx, prod_id))
    
    self._log('  Creating market basket for option #{} ...'.format(idx + 1))
    start = time()
    if idx == 0:
      df, prod_id, found_id = self.__select_k_products_by_cosine(prod_id)
      
    if idx == 1:
      df, prod_id, found_id = self.__select_k_products_by_mco(prod_id)
      
    if idx == 2:
      df, prod_id, found_id = self.__select_k_products_by_clusters(prod_id)
    self._log('  Created market basket. [{:.2f}s]'.format(time() - start))
    
    self._log('  Making annotations invisible for topk_plot #{}'.format(idx))
    for annotation in self.topk_plots[idx].annotations:
      if annotation is not None:
        annotation.visible = False
    self._log('  Finished making annotations invisible.', show_time = True)
    
    self._log('  Updating topk_plot #{} and its associated table ...'.format(idx))
    self.topk_plots[idx].Update(df, prod_id, found_id)
    self._log('  Finished updating topk_plot and its associated table.', show_time = True)
    return
  
  def _log(self, str_msg, results = False, show_time = False):
    self.logger.VerboseLog(str_msg, results, show_time)
    return