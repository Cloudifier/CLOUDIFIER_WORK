from keras.models import Model
from keras.layers import Input, Dense, Dropout
from logger_helper import LoadLogger
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import keras.backend as K
#import seaborn as sns
#import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import recall_score, precision_score, confusion_matrix, accuracy_score

if __name__ == '__main__':
  logger = LoadLogger(lib_name='TFTEST', config_file='config.txt')

  file_df_header = os.path.join(logger._base_folder, '_data/T2016C2017.xlsx')
  file_df = os.path.join(logger._base_folder, '_data/_DS_CHURN_TRAN_2016.csv')
  
  
  df_header = pd.read_excel(file_df_header)
  header = list(df_header.columns)
  
  logger.P("Reading churn dataset..")
  df = pd.read_csv(file_df, names=header)
  logger.P("Dataset in memory.", show_time=True)
  
  logger.P("Filling missing values..")
  df.fillna(0, inplace=True)
  logger.P("Missing values filled.", show_time=True)
  
  X = df.loc[:, "R1":"ZDROVITAL4"].values
  y = df.loc[:, "CHURN"].values
  nr_feats = X.shape[1]

  logger.P("There are {:,} clients. Each client is described by {} attributes.".format(X.shape[0], X.shape[1]))
  
  nr_hidden_1 = 168
  nr_hidden_2 = 84
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=33)
  
  K.clear_session()
  tf_input = Input(shape=(nr_feats,), name='input')
  
  DL1 = Dense(units=nr_hidden_1, activation='selu')
  Drop = Dropout(rate=0.5)
  DL2 = Dense(units=nr_hidden_2, activation='selu')
  DL3 = Dense(units=1, activation='sigmoid')
  
  output_hidden_1 = DL1(tf_input)
  drop_hidden_1 = Drop(output_hidden_1)
  output_hidden_2 = DL2(drop_hidden_1)
  yhat = DL3(output_hidden_2)
  
  model = Model(inputs=tf_input, outputs=yhat)
  model.compile(optimizer='adam', loss='binary_crossentropy')
  
  logger.P(logger.GetKerasModelSummary(model))
  
  history = model.fit(x=X_train, y=y_train, batch_size=1024, epochs=20)

  #logger.PlotKerasHistory(history)
  """
  sns.set()
  plt.figure()
  x = np.arange(1, len(history.history['loss'])+1)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Convergence of loss')
  plt.plot(x, history.history['loss'], 'r--')
  """
  y_hat_train = model.predict(X_train)
  y_hat_train = (y_hat_train >= 0.5).astype(np.int32)
  y_hat_test = model.predict(X_test)
  y_hat_test = (y_hat_test >= 0.5).astype(np.int32)
  
  conf_train = confusion_matrix(y_train, y_hat_train)
  conf_test  = confusion_matrix(y_test, y_hat_test)
  
  acc_train = accuracy_score(y_train, y_hat_train)
  acc_test = accuracy_score(y_test, y_hat_test)
  
  reca_train = recall_score(y_train, y_hat_train)
  reca_test = recall_score(y_test, y_hat_test)
  
  prec_train = precision_score(y_train, y_hat_train)
  prec_test = precision_score(y_test, y_hat_test)
