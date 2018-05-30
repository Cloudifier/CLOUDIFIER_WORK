"""
@author: Laurentiu Piciu
"""

from keras.models import Model
from keras.layers import Input, CuDNNLSTM, Embedding, Dense, concatenate, Bidirectional, Lambda
import bot_utils
import numpy as np
from collections import OrderedDict
import keras.backend as K
import os
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

__lib__       = "TranslatorBot"
__version__   = "0.1"
__author__    = "4E Software"
__copyright__ = "(C) 4E Software SRL"
__project__   = "TempRent"
__reference__ = ""



class BeamNode:
  def __init__(self):
    return






valid_lstms = ['UNIDIRECTIONAL', 'BIDIRECTIONAL']
valid_prediction_methods = ['sampling', 'argmax', 'beamsearch']

class Seq2SeqWrapper:
  def __init__(self, config_file='config.txt'):
    self.logger = bot_utils.LoadLogger(__lib__, config_file=config_file)
    self._log("Initializing {} seq2seq wrapper ..".format(__lib__))
    self.config_data = self.logger.config_data
    self.data_loader = bot_utils.DataLoaderV1(self.config_data["SEQUENCES_FILE"], self.logger,
                                              max_samples=self.config_data["MAX_SAMPLES"])
    self._parse_config_data()

    self.tf_enc_input = None
    self.enc_layers_full_state = OrderedDict()
    self.enc_full_state = []
    
    self.tf_dec_input = None
    self.tf_embeddings = None
    self.dec_lstm_cells = []
    self.SMLayer = None
    
    self.trainable_model = None
    self.enc_pred_model = None
    self.gen_seq_model = None

    self.model_trained = False
    self.epoch_loaded_model = 0
    self.test_size = 0.1


    self._log("Initialized the wrapper.")
    
    """
      "quick" methods:
        (1) CreateModelUsingCrtConfig(): using the config_file given as parameter in __init__ function,
            will compile the encoder-decoder architecture specified (config_data["ENCODER_ARCHITECTURE"]
            and config_data["DECODER_ARCHITECTURE"])
        
        (2) LoadModelWeightsAndConfig(model_label): searches in the models directory, a folder named
            '[model_label]'. Here will find 'weights.pkl' and 'config.txt'. The method will compile the
            encoder-decoder architecture found in '[model_label]/config.txt' and loads the weights
            for each trainable layer. The resulting model can be trained more epochs, can be used
            for predictions etc.
           
        (3) Fit(epochs, test_size, save_period): Trains the resulted model from (1) or (2). save_period
            tells the interval for saving the weights and config file.  (3)
           
        (4) CheckRandomTrainingExamples(self, nr_samples, method): Chooses 'nr_samples' training
            examples and decodes them using _step_by_step_prediction method. Method can be
            ['sampling', 'argmax', 'beamsearch']
        
        (5) CheckHumanInput(method): Allows the user to input sentences that will
            be decoded using _step_by_step_prediction method. Method can be ['sampling', 'argmax', 'beamsearch']
        
        (6) FitPredict(epochs, test_size, save_period, nr_samples, method): combination between (3) and (4)
        
        (7) FitPredictHumanInputs(epochs, test_size, save_period, method): combination between (3) and (5)
    """
    
    
    
    return



  def _log(self, str_msg, results=False, show_time=False):
    self.logger.VerboseLog(str_msg, results, show_time)
    return
  
  
  def _get_last_key_val_odict(self, odict):
    for k in odict.keys():
      continue
    return k, odict[k]


  def _get_key_index_odict(self, odict, key):
    for i,k in enumerate(odict.keys()):
      if k == key:
        return i
    return


  def __get_lstm_cells_names(self):
    names =[]
    for d in self.encoder_architecture:
      names.append(d['NAME'])
    for d in self.decoder_architecture:
      names.append(d['NAME'])
    return names


  def _parse_config_data(self):
    self.model_trained_layers = []
    self.peek_at_encoder = False
    
    # Last LSTM encoder cell should have a number of units equal to first LSTM decoder cell 
    self.encoder_architecture = self.config_data["ENCODER_ARCHITECTURE"]
    self.decoder_architecture = self.config_data["DECODER_ARCHITECTURE"]
    self.peek_at_encoder = (self.decoder_architecture[0]['PEEK'] == 1)
    
    # if our encoder-decoder is multi-language (e.g. en-ro), then we will have 2 embeddings layers;
    # otherwise one embeddings layer is sufficient
    self.multi_language = self.config_data["MULTI_LANGUAGE"]

    self.model_label = self.config_data["MODEL_LABEL"]


    self.tensors_and_layers = self.config_data['TENSORS_AND_LAYERS']
    self.__name_enc_emb_layer = self.tensors_and_layers['ENCODER_EMBEDDING_LAYER'][str(self.multi_language)]
    self.__name_dec_emb_layer = self.tensors_and_layers['DECODER_EMBEDDING_LAYER'][str(self.multi_language)]
    self.__name_dec_rdout_layer = self.tensors_and_layers['DECODER_READOUT_LAYER']
    
    
    self.model_trained_layers = self.__get_lstm_cells_names() +\
                                [self.__name_enc_emb_layer, self.__name_dec_emb_layer,
                                 self.__name_dec_rdout_layer]
    return





  ########################### TRAINABLE ENCODER-DECODER ARCHITECTURE ########################### <S>
  def _create_embeddings_layers(self):
    embeddings_layers = {}
    if bool(self.multi_language) == False:
      # Only an Embedding Layer if the bot is not multi_language
      assert self.data_loader.inp_word_vocab_size == self.data_loader.out_word_vocab_size
      EmbeddingLayer = Embedding(input_dim=self.data_loader.inp_word_vocab_size,
                                 output_dim=self.config_data["ENC_NR_EMBED"],
                                 name=self.__name_enc_emb_layer)
      embeddings_layers['encode'] = EmbeddingLayer
      embeddings_layers['decode'] = EmbeddingLayer
    else:
      embeddings_layers['encode'] = Embedding(input_dim=self.data_loader.inp_word_vocab_size,
                                              output_dim=self.config_data["ENC_NR_EMBED"],
                                              name=self.__name_enc_emb_layer)
      embeddings_layers['decode'] = Embedding(input_dim=self.data_loader.out_word_vocab_size,
                                              output_dim=self.config_data["DEC_NR_EMBED"],
                                              name=self.__name_dec_emb_layer)
    return embeddings_layers
             
  

  def _create_configurated_encoder(self, tf_input):
    lstm_sequences = {}
    crt_tf_input = tf_input
    for i in range(len(self.encoder_architecture)):
      layer_dict = self.encoder_architecture[i]
      name = layer_dict['NAME']
      units = layer_dict['NR_UNITS']
      lstm_type = layer_dict['TYPE'].upper()
      
      ### Check lstm_type
      if lstm_type not in valid_lstms:
        str_err = "ERROR! [EncLayer '{}'] The specified type ('{}') is not valid.".format(name, lstm_type)
        self._log(str_err)
        raise Exception(str_err)
      #endif
      
      
      ### Check skip connections
      all_skip_connections = [crt_tf_input]
      for skip in layer_dict['SKIP_CONNECTIONS']:
        if skip.upper() == 'INPUT':
          all_skip_connections.append(tf_input)
        else:
          if skip not in lstm_sequences.keys():
            str_err = "ERROR! [EncLayer '{}'] The specified skip connection ('{}') does not exist.".format(name, skip)
            self._log(str_err)
            raise Exception(str_err)
          #endif
          all_skip_connections.append(lstm_sequences[skip])
      #endfor
      if len(all_skip_connections) >= 2: crt_tf_input = concatenate(all_skip_connections)


      if lstm_type == 'BIDIRECTIONAL':
        EncLSTMCell = Bidirectional(CuDNNLSTM(units=units, return_sequences=True, return_state=True), name=name)
        crt_tf_input, tf_enc_h1, tf_enc_c1, tf_enc_h2, tf_enc_c2 = EncLSTMCell(crt_tf_input)
        tf_enc_h = concatenate([tf_enc_h1, tf_enc_h2])
        tf_enc_c = concatenate([tf_enc_c1, tf_enc_c2])
      
      if lstm_type == 'UNIDIRECTIONAL':
        EncLSTMCell = CuDNNLSTM(units=units, return_sequences=True, return_state=True, name=name)
        crt_tf_input, tf_enc_h, tf_enc_c = EncLSTMCell(crt_tf_input)
      
      self.enc_layers_full_state[name] = [tf_enc_h, tf_enc_c]
      self.enc_full_state += [tf_enc_h, tf_enc_c]
      lstm_sequences[name] = crt_tf_input
    #endfor
    return



  def _create_configurated_decoder(self, tf_input):
    crt_tf_input = tf_input
    lstm_sequences = {}

    # now we have allow decoder to peek at encoder at each timestep so we will 
    # replicate enc_peek_state for each timestep in tf_input
    if bool(self.decoder_architecture[0]['PEEK']) is True:
      _, enc_peek_state = self._get_last_key_val_odict(self.enc_layers_full_state)
      concat_enc_peek_state = concatenate(enc_peek_state)

      enc_feat_size = K.int_shape(concat_enc_peek_state)[-1]
      tf_timesteps = K.shape(tf_input)[1]

      reshape_layer = Lambda(lambda x: K.reshape(x, (-1, 1, enc_feat_size)), name='reshape_layer')
      repeat_layer = Lambda(lambda x: K.tile(x, (1,tf_timesteps,1)), name='repeat_layer')
      
      enc_peek_state_reshaped = reshape_layer(concat_enc_peek_state)
      enc_peek_state_tiled = repeat_layer(enc_peek_state_reshaped)
      crt_tf_input = concatenate([crt_tf_input, enc_peek_state_tiled], name='peek_layer')
      tf_input = crt_tf_input
    #endif
    
    
    for i in range(len(self.decoder_architecture)):
      layer_dict = self.decoder_architecture[i]
      name = layer_dict['NAME']
      units = layer_dict['NR_UNITS']
      lstm_type = layer_dict['TYPE'].upper()
      enc_layer_initial_state = layer_dict['INITIAL_STATE']
      
      ### Check lstm_type
      if lstm_type not in valid_lstms:
        str_err = "ERROR! [EncLayer '{}'] The specified type ('{}') is not valid.".format(name, lstm_type)
        self._log(str_err)
        raise Exception(str_err)
      #endif
      
      
      ### Check initial_state
      initial_state = None
      if enc_layer_initial_state != "":
        if enc_layer_initial_state not in self.enc_layers_full_state.keys():
          self._log("[DecLayer '{}'] The specified initial_state ('{}') does not exist."
                    .format(enc_layer_initial_state))
          initial_state = None
        else:
          initial_state = self.enc_layers_full_state[enc_layer_initial_state]
      #endif

      ### Check skip connections
      all_skip_connections = [crt_tf_input]
      for skip in layer_dict['SKIP_CONNECTIONS']:
        if skip.upper() == 'INPUT':
          all_skip_connections.append(tf_input)
        else:
          if skip not in lstm_sequences.keys():
            str_err = "ERROR! [DecLayer '{}'] The specified skip connection ('{}') does not exist.".format(name, skip)
            self._log(str_err)
            raise Exception(str_err)
          #endif
          all_skip_connections.append(lstm_sequences[skip])
      #endfor
      if len(all_skip_connections) >= 2: crt_tf_input = concatenate(all_skip_connections)
      

      if lstm_type == 'BIDIRECTIONAL':
        raise Exception("Decoder lstm_type='BIDIRECTIONAL' not implemented")
      if lstm_type == 'UNIDIRECTIONAL':
        DecLSTMCell = CuDNNLSTM(units=units, return_sequences=True, return_state=True, name=name)
        crt_tf_input, _, _ = DecLSTMCell(crt_tf_input, initial_state=initial_state)
        self.dec_lstm_cells.append(DecLSTMCell)
      lstm_sequences[name] = crt_tf_input
    #endfor

    self.SMLayer = Dense(units=self.data_loader.out_word_vocab_size, activation='softmax',
                         name=self.__name_dec_rdout_layer)
    tf_dec_preds = self.SMLayer(crt_tf_input)
    return tf_dec_preds
  



  def _compile_configurated_encoder_decoder(self):
    self._log("Creating encoder-decoder architecture based on embedding-space repr of words..")
    embeddings_layers = self._create_embeddings_layers()

    self.tf_enc_input = Input((None,), name=self.tensors_and_layers['ENCODER_INPUT'])
    self.tf_embeddings = embeddings_layers['encode'](self.tf_enc_input)
    self._create_configurated_encoder(self.tf_embeddings)

    self.tf_dec_input = Input((None,), name=self.tensors_and_layers['DECODER_INPUT'])
    self.tf_embeddings = embeddings_layers['decode'](self.tf_dec_input)
    tf_dec_preds = self._create_configurated_decoder(self.tf_embeddings)


    ### Encoder-Decoder Model
    self.trainable_model = Model(inputs=[self.tf_enc_input, self.tf_dec_input], outputs=tf_dec_preds)
    summary = self.logger.GetKerasModelSummary(self.trainable_model, full_info=False)
    self._log(" Architecture of the encoder: {} LSTM cells\nArchitecture of the decoder: {} LSTM cells. {}"
              .format(len(self.encoder_architecture), len(self.decoder_architecture), summary))

    self.trainable_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return
  
  # <E> ######################### TRAINABLE ENCODER-DECODER ARCHITECTURE ###########################




  def _create_prediction_models(self):
    self._log("Creating prediction models ...")
    self.enc_pred_model = Model(self.tf_enc_input, self.enc_full_state)

    # The model that predicts softmax probabilities is effectively a sequence generator:
    #   - given the encoded information (cell state and carry state) it will be able to
    #     generate a token until EOS is returned.
    crt_tf_input = self.tf_embeddings
    tf_input = self.tf_embeddings
    if self.peek_at_encoder is True:
      _, enc_peek_state = self._get_last_key_val_odict(self.enc_layers_full_state)
      enc_feat_size = K.int_shape(enc_peek_state[0])[-1]

      tf_peek_input1 = Input((enc_feat_size,))
      tf_peek_input2 = Input((enc_feat_size,))

      tf_peek_full_input = concatenate([tf_peek_input1, tf_peek_input2])

      tf_timesteps = K.shape(self.tf_dec_input)[1]    
      reshape_layer = Lambda(lambda x: K.reshape(x,(-1, 1, enc_feat_size*2)), name='reshape_layer_dm')
      repeat_layer = Lambda(lambda x: K.tile(x,(1,tf_timesteps,1)), name='repeat_layer_dm')
  
      dec_peek_inp_tiled = repeat_layer(reshape_layer(tf_peek_full_input))
      crt_tf_input = concatenate([crt_tf_input, dec_peek_inp_tiled], name='gen_inp_peek')
      tf_input = crt_tf_input
  
    dec_model_inputs = []
    dec_model_outputs = []
    lstm_sequences = {}
    for i in range(len(self.dec_lstm_cells)):
      layer_dict = self.decoder_architecture[i]
      name = layer_dict['NAME']
      units = layer_dict['NR_UNITS']

      tf_inp_h = Input((units,), name='gen_inp_h_' + str(i+1))
      tf_inp_c = Input((units,), name='gen_inp_c_' + str(i+1))
      dec_model_inputs.append(tf_inp_h)
      dec_model_inputs.append(tf_inp_c)

      ### Check skip connections
      all_skip_connections = [crt_tf_input]
      for skip in layer_dict['SKIP_CONNECTIONS']:
        if skip.upper() == 'INPUT':
          all_skip_connections.append(tf_input)
        else:
          if skip not in lstm_sequences.keys():
            str_err = "ERROR! [DecLayer '{}'] The specified skip connection ('{}') does not exist.".format(name, skip)
            self._log(str_err)
            raise Exception(str_err)
          #endif
          all_skip_connections.append(lstm_sequences[skip])
      #endfor
      if len(all_skip_connections) >= 2: crt_tf_input = concatenate(all_skip_connections)

      crt_tf_input, tf_h, tf_c = self.dec_lstm_cells[i](crt_tf_input,
                                                        initial_state=[tf_inp_h, tf_inp_c])
      dec_model_outputs.append(tf_h)
      dec_model_outputs.append(tf_c)
      
      lstm_sequences[name] = crt_tf_input
    #endfor

    tf_gen_preds = self.SMLayer(crt_tf_input)

    if self.peek_at_encoder is False:
      self.gen_seq_model = Model(inputs=dec_model_inputs + [self.tf_dec_input],
                                 outputs=dec_model_outputs + [tf_gen_preds])
    else:
      self.gen_seq_model = Model(inputs=dec_model_inputs + [self.tf_dec_input, tf_peek_input1, tf_peek_input2],
                                 outputs=dec_model_outputs + [tf_gen_preds])

    summary = self.logger.GetKerasModelSummary(self.enc_pred_model, full_info=False)
    self._log(summary)
    summary = self.logger.GetKerasModelSummary(self.gen_seq_model, full_info=False)
    self._log(summary)
    return



  def __get_train_test_data(self, debug):
    return self.data_loader.get_train_test_data(as_words=True, test_size=self.test_size,
                                                random_state=33, debug=debug)

  def _generate_fit_batches(self, debug):
    self.X_train_enc, X_train_dec, y_train_dec, _, _, _ = self.__get_train_test_data(debug)

    while True:
      for i in range(len(self.X_train_enc)):
        X_e = np.expand_dims(np.array(self.X_train_enc[i]), axis=0)
        X_d = np.expand_dims(np.array(X_train_dec[i]), axis=0)
        y_d = np.array(y_train_dec[i])
        y_d = y_d.reshape((1, -1, 1))
        
        yield [X_e, X_d], y_d


  def _generate_train_predict_batches(self):
    X_train_enc, _, y_train_dec, _, _, _ = self.__get_train_test_data(debug=False)
    for i in range(len(X_train_enc)):
      yield X_train_enc[i], y_train_dec[i][:-1]

  
  
  def _generate_test_predict_batches(self):
    _, _, _, X_test_enc, _, y_test_dec = self.__get_train_test_data(debug=False)    
    for i in range(len(X_test_enc)):
      yield X_test_enc[i], y_test_dec[i][:-1]


  def Fit(self, epochs=1, test_size=0.1, save_period=None, verbose=1):
    DEBUG = False  ### TODO change DEBUG to False when it's obviously that data_loader does the right job


    self._log("Training {} seq2seq model ..".format(__lib__))
    steps_per_epoch = int(self.data_loader.n_pairs * (1-test_size))
    self.save_period = save_period
    self.test_size = test_size

    epoch_callback = self.logger.GetKerasEpochCallback(predef_callback=self._on_epoch_end_callback)
    self.loss_hist = []
    self.trainable_model.fit_generator(self._generate_fit_batches(DEBUG),
                                       steps_per_epoch=steps_per_epoch,
                                       epochs=epochs,
                                       verbose=verbose,
                                       callbacks=[epoch_callback])
    
    self.model_trained = True

    return


  def FitPredict(self, epochs=1, test_size=0.1, save_period=None, verbose=1,
                 nr_samples=10, method='sampling'):
    self.Fit(epochs=epochs, test_size=test_size, save_period=save_period,
             verbose=verbose)
    self.CheckRandomTrainingExamples(nr_samples=nr_samples, method=method)
    return


  def FitPredictHumanInputs(self, epochs=1, test_size=0.1, save_period=None,
                            verbose=1, method='sampling'):
    self.Fit(epochs=epochs, test_size=test_size, save_period=save_period,
             verbose=verbose)
    self.CheckHumanInput(method=method)
    return


  

  def _beam_search_prediction(self, enc_initial_states, beam_size=10):
    beam_tree = [BeamNode(parent=None, state=enc_initial_states, cum_cost=1.0, dist_to_root=0,
                          token=self.data_loader.out_word_to_id[self.data_loader.START_CHAR])]
    
    current_beam = []
    length = len(beam_tree)
    for _ in range(length):
      node = beam_tree.pop()
    
      crt_token = node.token
      crt_token = np.array(crt_token).reshape((1,1))
      
      if self.peek_at_encoder is False:
        dec_model_outputs = self.gen_seq_model.predict(node.state + [crt_token])
      else:
        dec_model_outputs = self.gen_seq_model.predict(node.state + [crt_token, enc_initial_states[-2],
                                                                     enc_initial_states[-1]])
     
      P = dec_model_outputs[-1]
      current_beam.append((node, P, dec_model_outputs[:-1]))
    #endfor
    for T in current_beam:
      pass
    


  def _step_by_step_prediction(self, _input, method='sampling', verbose=1):
    if self.__check_can_start_prediction(method) is False:
      return
    
    if type(_input) == str:
      str_input = _input
      input_tokens = self.data_loader.input_word_text_to_tokens(str_input)
      input_tokens = np.expand_dims(np.array(input_tokens), axis=0)
    elif type(_input) == list:
      str_input = self.data_loader.input_word_tokens_to_text(_input)
      input_tokens = np.expand_dims(np.array(_input), axis=0)
    
    if verbose: self._log("Given '{}' the decoder predicted:".format(str_input))
    enc_states = self.enc_pred_model.predict(input_tokens)
    dec_model_inputs = []
    
    for i in range(len(self.decoder_architecture)):
      layer_dict = self.decoder_architecture[i]
      units = layer_dict['NR_UNITS']
      enc_layer_initial_state = layer_dict['INITIAL_STATE']
      inp_h = np.zeros((1, units))
      inp_c = np.zeros((1, units))
      
      if enc_layer_initial_state != "":
        idx = self._get_key_index_odict(self.enc_layers_full_state, enc_layer_initial_state)
        if idx is not None:
          inp_h, inp_c = enc_states[2*idx:2*(idx+1)]
      #endif
      
      dec_model_inputs += [inp_h, inp_c]
    #endfor



    current_gen_token = self.data_loader.out_word_to_id[self.data_loader.START_CHAR]
    predicted_tokens = []
    while current_gen_token != self.data_loader.out_word_to_id[self.data_loader.END_CHAR]:
      current_gen_token = np.array(current_gen_token).reshape((1,1))
      
      if self.peek_at_encoder is False:
        dec_model_outputs = self.gen_seq_model.predict(dec_model_inputs + [current_gen_token])
      else:
        dec_model_outputs = self.gen_seq_model.predict(dec_model_inputs +\
                                                       [current_gen_token, enc_states[-2], enc_states[-1]])

      P = dec_model_outputs[-1]
      if method == 'sampling':
        current_gen_token = np.random.choice(range(P.shape[-1]), p=P[-1].squeeze())
      if method == 'armgax':
        current_gen_token = np.argmax(P)
      if method == 'beamsearch':
        raise Exception("{} not implemented yet.".format(method))  ### TODO implement beamsearch

      predicted_tokens.append(current_gen_token)
      dec_model_inputs = dec_model_outputs[:-1]
    #end_while
    predicted_tokens = predicted_tokens[:-1]
    if verbose:
      self._log("  --> '{}'".format(self.data_loader.output_word_tokens_to_text(predicted_tokens)))
    return predicted_tokens
  
  
  def __check_can_start_prediction(self, method):
    if self.model_trained is False:
      self._log("Cannot predict because the model is not trained.")
      return False

    method = method.lower()
    if method not in valid_prediction_methods:
      self._log("The specified method is not valid. Try one from '{}'".format(valid_prediction_methods))
      return False
    
    if (self.enc_pred_model == None) or (self.gen_seq_model == None):
      self._create_prediction_models()
    return True



  def CheckRandomTrainingExamples(self, nr_samples=10, method='sampling'):
    if self.__check_can_start_prediction(method) is False:
      return

    test_batch = np.random.choice(self.X_train_enc, nr_samples)
    for seq in test_batch:
      self._step_by_step_prediction(seq, method)
    return
  

  def CheckHumanInput(self, method='sampling'):
    if self.__check_can_start_prediction(method) is False:
      return
    
    self._log("Now you can input sentences. Whenever you want to stop, type '!quit' ..")
    while True:
      str_input = input()
      if "!quit" in str_input.lower():
        break
      self._step_by_step_prediction(str_input, method)

    return


  def __Tqdm_Enumerate(self, _iter):
    i = 0
    for y in tqdm(_iter):
      yield i, y
      i += 1


  def Predict(self, dataset='train', method='sampling'):
    if self.__check_can_start_prediction(method) is False:
      return

    valid_datasets = ['train', 'test']
    if dataset.lower() not in valid_datasets:
      self._log("Dataset not valid. Try 'train' or 'test'.")
      return

    self._log("[{}] Predicting '{}' examples, using '{}' method..."
              .format(self.model_label, dataset, method))
    if dataset is 'train':
      gen = self._generate_train_predict_batches()
    if dataset is 'test':
      gen = self._generate_test_predict_batches()

    scores = []
    
    for _, (x, y) in self.__Tqdm_Enumerate(gen):
      list_tokens_MT  = list(map(lambda tok: self.data_loader.out_id_to_word[tok],
                                 self._step_by_step_prediction(x, method, verbose=0)))
      list_tokens_ref = list(map(lambda tok: self.data_loader.out_id_to_word[tok],
                                 y))
      smoothie = SmoothingFunction().method1
      scores.append(sentence_bleu([list_tokens_ref], list_tokens_MT, smoothing_function=smoothie))

    avg_score = sum(scores) / float(len(scores))
    return scores, avg_score




  def _on_epoch_end_callback(self, epoch, logs):
    epoch = epoch + 1 + self.epoch_loaded_model
    str_logs = ""
    for key,val in logs.items():
      str_logs += "{}:{:.6f}  ".format(key,val)
    self._log(" Train/Fit: Epoch: {} Results: {}".format(epoch,str_logs))
    
    loss = logs['loss']
    self.loss_hist.append((epoch, loss))    
    self._save_model(epoch, loss)
    return
    

  def _mk_plot(self, fig_path):
    sns.set()
    plt.figure(figsize=(1280/96, 720/96), dpi=96)
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    epochs = [i[0] for i in self.loss_hist]
    losses = [i[1] for i in self.loss_hist]
    
    plt.plot(epochs, losses, linestyle='--', marker='o', color='r')
    plt.savefig(fig_path, dpi=96)
    plt.close()
    return

  def _save_model(self, epoch, loss):
    if self.save_period is None:
      return
    if epoch != 1:  ## epoch 1 is always saved in order to see progress
      if (epoch % self.save_period) != 0:
        return
      

    assert len(self.model_trained_layers) > 0, 'Unknown list of trained layers!'

    str_epoch = str(epoch).zfill(2)
    str_loss = "{:.2f}".format(loss)
    model_folder = self.model_label + "_epoch" + str_epoch + "_loss" + str_loss

    model_full_path = os.path.join(self.logger.GetModelsFolder(), model_folder)
    if not os.path.exists(model_full_path):
      os.makedirs(model_full_path)

    fn_weights = model_folder + '/weights'
    fn_config = os.path.join(self.logger.GetModelsFolder(), model_folder + '/config.txt')
    fn_losshist = os.path.join(self.logger.GetModelsFolder(), model_folder + '/loss_hist.jpg')
    self.config_data['EPOCH'] = epoch
    self.config_data['TEST_SIZE'] = self.test_size
    with open(fn_config, 'w') as f:
      f.write(json.dumps(self.config_data, indent=2))
    self._log("Saved model config in '{}'.".format(fn_config))
    self.logger.SaveKerasModelWeights(fn_weights, self.trainable_model, self.model_trained_layers)
    self._mk_plot(fn_losshist)
    return






  def CreateModelUsingCrtConfig(self):
    self._compile_configurated_encoder_decoder()
    return


  def LoadModelWeightsAndConfig(self, model_label):
    fn_config = os.path.join(self.logger.GetModelsFolder(), model_label + '/config.txt')
    if not os.path.exists(fn_config):
      self._log("Did not found the specified model_label: '{}'.".format(model_label))
      return
    
    self.logger._configure_data_and_dirs(fn_config)
    self.config_data = self.logger.config_data
    self._parse_config_data()
    self.epoch_loaded_model = self.config_data['EPOCH']
    self.test_size = self.config_data['TEST_SIZE']

    assert len(self.model_trained_layers) > 0, 'Unknown list of trained layers!'
    self._compile_configurated_encoder_decoder()

    fn_weights = model_label + '/weights'
    if self.logger.LoadKerasModelWeights(fn_weights, self.trainable_model, self.model_trained_layers):
      self.model_trained = True

    return

  
  
  

####################################################

if __name__ == '__main__':
  K.clear_session()
  bot = Seq2SeqWrapper()
  bot.LoadModelWeightsAndConfig('en_ro_translator_v1_epoch60_loss0.25')


