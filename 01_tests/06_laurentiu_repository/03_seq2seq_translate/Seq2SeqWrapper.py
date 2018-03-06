from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense
import bot_utils
import numpy as np
from tqdm import tqdm

__lib__ = "ENFRobot"

class Seq2SeqWrapper:
  def __init__(self, config_file='config.txt'):
    self.logger = bot_utils.LoadLogger(__lib__, config_file=config_file)
    self._log("Initializing {} seq2seq model ..".format(__lib__))
    self.config_data = self.logger.config_data
    self.data_loader = bot_utils.DataLoaderV1(self.config_data["SEQUENCES_FILE"], self.logger,
                                              max_samples=self.config_data["MAX_SAMPLES"])
    self._parse_config_data()

    self.tf_enc_input = None
    self.enc_full_state = None
    
    self.tf_dec_input = None
    self.tf_embeddings = None
    self.dec_lstm_cells = []
    self.SMLayer = None
    
    self.trainable_model = None
    self.enc_pred_model = None
    self.gen_seq_model = None
    
    self._create_encoder_decoder()
    self._log("Initialized the model.")
    return


  def _log(self, str_msg, results=False, show_time=False):
    self.logger.VerboseLog(str_msg, results, show_time)
    return


  def _parse_config_data(self):
    # Last LSTM encoder cell should have a number of units equal to first LSTM decoder cell 
    self.encoder_architecture = self.config_data["ENCODER_ARCHITECTURE"]
    self.decoder_architecture = self.config_data["DECODER_ARCHITECTURE"]
    assert self.encoder_architecture[-1] == self.decoder_architecture[0]

    # each sentence token will be represented using embeddings, rather than one-hot..
    self.nr_embeddings = self.config_data["NR_EMBEDDINGS"]
    
    # if our encoder-decoder is multi-language (e.g. en-ro), then we will have 2 embeddings layers;
    # otherwise one embeddings layer is sufficient
    self.multi_language = bool(self.config_data["MULTI_LANGUAGE"])
    
    # bidirectional lstm
    self.bidirectional = bool(self.config_data["BIDIRECTIONAL"])
    
    # flag that specifies if the model is saved on disk at the end of training
    self.save_model_eot = bool(self.config_data["SAVE_MODEL"])
    self.model_label = self.config_data["MODEL_LABEL"]
    return


  def _create_encoder_decoder(self):
    self._log("Creating encoder-decoder architecture based on embedding-space repr of words..")

    embeddings_layers = {}
    if self.multi_language == False:
      # Only an Embedding Layer if the bot is not multi_language
      assert self.data_loader.inp_word_vocab_size == self.data_loader.out_word_vocab_size
      EmbeddingLayer = Embedding(input_dim=self.data_loader.inp_word_vocab_size,
                                 output_dim=self.nr_embeddings,
                                 name='emb')
      embeddings_layers['encode'] = EmbeddingLayer
      embeddings_layers['decode'] = EmbeddingLayer
    else:
      embeddings_layers['encode'] = Embedding(input_dim=self.data_loader.inp_word_vocab_size,
                                              output_dim=self.nr_embeddings,
                                              name='enc_emb')
      embeddings_layers['decode'] = Embedding(input_dim=self.data_loader.out_word_vocab_size,
                                              output_dim=self.nr_embeddings,
                                              name='dec_emb')

    ### Stacked encoding LSTM Cells
    self.tf_enc_input = Input((None,), name='enc_input')
    self.tf_embeddings = embeddings_layers['encode'](self.tf_enc_input)
    
    # First n-1 LSTM cells return sequences and get as input previous returned sequences
    crt_tf_input = self.tf_embeddings
    for i in range(len(self.encoder_architecture) - 1):
      EncLSTMCell = LSTM(units=self.encoder_architecture[i], return_sequences=True,
                         name='enc_lstm_' + str(i+1))
      crt_tf_input = EncLSTMCell(crt_tf_input)
    # The last LSTM cell returns the state that is processed by the decoder
    EncLSTMCell = LSTM(units=self.encoder_architecture[-1], return_sequences=True, return_state=True,
                       name='enc_lstm_top')
    tf_enc_output, tf_enc_h, tf_enc_c = EncLSTMCell(crt_tf_input)
    self.enc_full_state = [tf_enc_h, tf_enc_c]
    ###

    ### Stacked decoding LSTM Cells
    self.tf_dec_input = Input((None,), name='dec_input')
    self.tf_embeddings = embeddings_layers['decode'](self.tf_dec_input)
    
    # First LSTM cell receives the initial_state from the encoder
    crt_tf_input = self.tf_embeddings
    DecLSTMCell = LSTM(units=self.decoder_architecture[0], return_sequences=True, return_state=True,
                       name='dec_lstm_bottom')
    crt_tf_input, _, _ = DecLSTMCell(crt_tf_input, initial_state=self.enc_full_state)
    self.dec_lstm_cells.append(DecLSTMCell)
    # The top ones just connects with the previous via the returned sequences
    for i in range(1, len(self.decoder_architecture)):
      DecLSTMCell = LSTM(units=self.decoder_architecture[i], return_sequences=True, return_state=True,
                         name='dec_lstm_' + str(i+1))
      crt_tf_input, _, _ = DecLSTMCell(crt_tf_input)
      self.dec_lstm_cells.append(DecLSTMCell)
    self.SMLayer = Dense(units=self.data_loader.out_word_vocab_size, activation='softmax')
    tf_dec_preds = self.SMLayer(crt_tf_input)
    ###

    ### Encoder-Decoder Model
    self.trainable_model = Model(inputs=[self.tf_enc_input, self.tf_dec_input], outputs=tf_dec_preds)
    summary = self.logger.GetKerasModelSummary(self.trainable_model, full_info=False)
    self._log(" Architecture of the encoder: {} LSTM cells -- {}\nArchitecture of the decoder: {} LSTM cells -- {}. {}"
              .format(len(self.encoder_architecture), self.encoder_architecture,
                      len(self.decoder_architecture), self.decoder_architecture, summary))

    self.trainable_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    return


  def _create_prediction_models(self):
    self._log("Creating prediction models ...")
    self.enc_pred_model = Model(self.tf_enc_input, self.enc_full_state)
    
    # The model that predicts softmax probabilities is effectively a sequence generator:
    #   - given the encoded information (cell state and carry state) it will be able to
    #     generate a token until EOS is returned.
    dec_model_inputs = []
    dec_model_outputs = []
    crt_tf_input = self.tf_embeddings
    for i in range(len(self.dec_lstm_cells)):
      tf_inp_h = Input((self.decoder_architecture[i],), name='gen_inp_h_' + str(i+1))
      tf_inp_c = Input((self.decoder_architecture[i],), name='gen_inp_c_' + str(i+1))
      dec_model_inputs.append(tf_inp_h)
      dec_model_inputs.append(tf_inp_c)

      crt_tf_input, tf_h, tf_c = self.dec_lstm_cells[i](crt_tf_input,
                                                        initial_state=[tf_inp_h, tf_inp_c])
      dec_model_outputs.append(tf_h)
      dec_model_outputs.append(tf_c)
    #endfor

    tf_gen_preds = self.SMLayer(crt_tf_input)
    self.gen_seq_model = Model(inputs=dec_model_inputs + [self.tf_dec_input],
                               outputs=dec_model_outputs + [tf_gen_preds])
    summary = self.logger.GetKerasModelSummary(self.enc_pred_model, full_info=False)
    self._log(summary)
    summary = self.logger.GetKerasModelSummary(self.gen_seq_model, full_info=False)
    self._log(summary)
    return


  def Fit(self, epochs=1):
    self._log("Training {} seq2seq model ..".format(__lib__))
    self.X_train_enc, X_train_dec, y_train_dec, _, _, _ = self.data_loader.get_train_test_data(as_words=True,
                                                                                               test_size=0.1,
                                                                                               random_state=33,
                                                                                               debug=True)

    obj_tqdm = tqdm(range(len(self.X_train_enc)))
    for i in obj_tqdm:
      X_e = np.expand_dims(np.array(self.X_train_enc[i]), axis=0)
      X_d = np.expand_dims(np.array(X_train_dec[i]), axis=0)
      y_d = np.array(y_train_dec[i])
      y_d = y_d.reshape((1, -1, 1))
      history = self.trainable_model.fit(x=[X_e, X_d], y=y_d, batch_size=1, epochs=epochs, verbose=0)
      obj_tqdm.set_description("Loss: {:.2f}".format(history.history['loss'][0]))

    if self.save_model_eot:
      self.SaveTrainableModel(self.model_label)
    return


  def FitPredict(self, epochs=1, nr_pred_samples=10):
    self.Fit(epochs=epochs)
    self.CheckRandomTrainingExamples(nr_samples=nr_pred_samples)
    return


  def FitPredictHumanInputs(self, epochs=1):
    self.Fit(epochs=epochs)
    self.CheckHumanInput()
    return


  def _generative_prediction(self, _input, random_sampling=True):
    if type(_input) == str:
      str_input = _input
      input_tokens = self.data_loader.input_word_text_to_tokens(str_input)
      input_tokens = np.expand_dims(np.array(input_tokens), axis=0)
    elif type(_input) == list:
      str_input = self.data_loader.input_word_tokens_to_text(_input)
      input_tokens = np.expand_dims(np.array(_input), axis=0)
    
    self._log("Given [{}] the decoder predicted:".format(str_input))    
    h_state, c_state = self.enc_pred_model.predict(input_tokens)
    dec_model_inputs = [h_state, c_state]
    for i in range(1, len(self.decoder_architecture)):
      initial_state_upper_cells = [np.zeros((1, self.decoder_architecture[i])),
                                   np.zeros((1, self.decoder_architecture[i]))]
      dec_model_inputs += initial_state_upper_cells

    current_gen_token = self.data_loader.out_word_to_id[self.data_loader.START_CHAR]
    predicted_seq = []
    while current_gen_token != self.data_loader.out_word_to_id[self.data_loader.END_CHAR]:
      current_gen_token = np.array(current_gen_token).reshape((1,1))
      dec_model_outputs = self.gen_seq_model.predict(dec_model_inputs + [current_gen_token])

      P = dec_model_outputs[-1]
      if random_sampling:
        current_gen_token = np.random.choice(range(P.shape[-1]), p=P[-1].squeeze())
      else:
        current_gen_token = np.argmax(P)
      predicted_seq.append(self.data_loader.out_id_to_word[current_gen_token])

      dec_model_inputs = dec_model_outputs[:-1]
    #end_while
    str_pred = ' '.join(predicted_seq)
    if str_pred[-1] == "\n":
      str_pred = str_pred[:-1]
      str_pred += "[EOS]"
    self._log("  --> [{}]".format(str_pred))
    return
  
  
  def CheckRandomTrainingExamples(self, nr_samples=10, random_sampling=True):
    if (self.enc_pred_model == None) or (self.gen_seq_model == None):
      self._create_prediction_models()
      
    test_batch = np.random.choice(self.X_train_enc, nr_samples)
    for seq in test_batch:
      self._generative_prediction(seq, random_sampling)
    return
  

  def CheckHumanInput(self, random_sampling=True):
    self._log("Now you can input sentences. Whenever you want to stop, type '!quit' ..")
    if (self.enc_pred_model == None) or (self.gen_seq_model == None):
      self._create_prediction_models()

    while True:
      str_input = input()
      if "!quit" in str_input.lower():
        break
      self._generative_prediction(str_input, random_sampling)

    return

  def SaveTrainableModel(self, label="", use_prefix=True):
    self.logger.SaveKerasModel(self.trainable_model, label, use_prefix)
    return
  
  def SaveEncoderDecoder(self, label="", use_prefix=True):
    self.logger.SaveKerasModel(self.enc_pred_model, "encoder_" + label, use_prefix)
    self.logger.SaveKerasModel(self.gen_seq_model, "decoder_" + label, use_prefix)
    return

  def LoadTrainableModel(self, model_name):
    self.trainable_model = self.logger.LoadKerasModel(model_name)
    return
  
  def LoadEncoderDecoder(self, *args):
    model_names = list(args)
    self.enc_pred_model = self.logger.LoadKerasModel(model_names[0])
    self.gen_seq_model = self.logger.LoadKerasModel(model_names[1])
    return

####################################################

if __name__ == '__main__':
  bot = Seq2SeqWrapper()
  #bot.FitPredict(epochs=40)