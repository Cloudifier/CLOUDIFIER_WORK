from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense
import bot_utils
import keras.backend as K
import numpy as np
from tqdm import tqdm

DEBUG = True

if __name__ == '__main__':
  logger = bot_utils.LoadLogger("ROEN_BOT", config_file='config.txt')
  data_loader = bot_utils.DataLoaderV1('ron.txt', logger)
  
  as_words = True
  debug = True
  
  X_enc, X_dec, y_dec, X_v_enc, X_v_dec, y_v_dec = data_loader.get_train_test_data(as_words=as_words,
                                                                                   debug=debug)
  
  output_dim_emb = 32
  units_lstm_enc = 128
  units_lstm_dec = 128
  
  encoder_input = Input((None,), name='enc_input')
  embeddings_enc = Embedding(input_dim=data_loader.inp_word_vocab_size,
                             output_dim=output_dim_emb,
                             name='enc_emb')(encoder_input)
  encoder_lstm = LSTM(units=units_lstm_enc, return_state=True, name='encoder')
  enc_output, enc_h, enc_c = encoder_lstm(embeddings_enc)
  encoder_full_state = [enc_h, enc_c]

  decoder_input = Input((None,), name='dec_input')
  embeddings_dec = Embedding(input_dim=data_loader.out_word_vocab_size,
                             output_dim=output_dim_emb,
                             name='dec_emb')(decoder_input)
  decoder_lstm = LSTM(units=units_lstm_dec, return_sequences=True, return_state=True, name='decoder')
  decoder_output, _, _ = decoder_lstm(embeddings_dec)
  sm_layer = Dense(units=data_loader.out_word_vocab_size, activation='softmax')
  preds = sm_layer(decoder_output)
  
  model = Model(inputs=[encoder_input, decoder_input], outputs=preds)
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
  
  
  if not DEBUG:
    tqdm_obj = tqdm(range(len(X_enc)))
    
    for i in tqdm_obj:
      X_e = np.expand_dims(np.array(X_enc[i]), axis=0)
      X_d = np.expand_dims(np.array(X_dec[i]), axis=0)
      y_d = np.array(y_dec[i])
      y_d = y_d.reshape((1, -1, 1))
      history = model.fit(x=[X_e, X_d], y=y_d, batch_size=1, epochs=20, verbose=0)
      
      tqdm_obj.set_description("Loss: {:.2f}".format(history.history['loss'][0]))


    encoder = Model(encoder_input, encoder_full_state)
  
    decoder_h_input = Input((units_lstm_enc,))
    decoder_c_input = Input((units_lstm_enc,))
    decoder_2_output, decoder_2_h, decoder_2_c = decoder_lstm(embeddings_dec, initial_state=[decoder_h_input, decoder_c_input])
    preds2 = sm_layer(decoder_2_output)
    decoder = Model([decoder_h_input, decoder_c_input, decoder_input], [decoder_2_h, decoder_2_c, preds2])

    while True:
      str_input = input()
      input_tokens = data_loader.input_word_text_to_tokens(str_input)
      input_tokens = np.expand_dims(np.array(input_tokens), axis=0)
      encoder_h, encoder_c = encoder.predict(input_tokens)
      d_h = encoder_h
      d_c = encoder_c
      current_token = data_loader.out_word_to_id[data_loader.START_CHAR]
      predicted_seq = []
      while current_token != data_loader.out_word_to_id[data_loader.END_CHAR]:
        current_token = np.array(current_token).reshape((1,1))
        d_h, d_c, P = decoder.predict([d_h, d_c, current_token])
        current_token = np.argmax(P, axis=-1)
        predicted_seq.append(data_loader.out_id_to_word[current_token[0,0]])
      print("Translated to: {}".format(predicted_seq))
  