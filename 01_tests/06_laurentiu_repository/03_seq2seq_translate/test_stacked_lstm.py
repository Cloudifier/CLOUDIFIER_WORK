from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense
import numpy as np

lstm_architecture = [128,256,128]
vocab_size = 100000

tf_enc_input = Input((None, 32))
crt_tf_enc_input = tf_enc_input
for i in range(len(lstm_architecture) - 1):
    EncLSTMCell = LSTM(units=lstm_architecture[i], return_sequences=True)
    crt_tf_enc_input = EncLSTMCell(crt_tf_enc_input)
EncLSTMCell = LSTM(units=lstm_architecture[-1], return_state=True, name='encoder')
tf_enc_output, tf_enc_h, tf_enc_c = EncLSTMCell(crt_tf_enc_input)
enc_full_state = [tf_enc_h, tf_enc_c]


tf_dec_input = Input((None, 32))
DecLSTMCell = LSTM(units=lstm_architecture[0], return_sequences=True, return_state=True)
dec_out, _, _ = DecLSTMCell(tf_dec_input, initial_state=enc_full_state)
SMLayer = Dense(units=vocab_size, activation='softmax')
tf_dec_preds = SMLayer(dec_out)



input1 = Input((128,))
input2 = Input((128,))

dec_out2, dec_h, dec_c = DecLSTMCell(tf_dec_input, initial_state=[input1, input2])
preds = SMLayer(dec_out2)
model2 = Model(inputs=[input1, input2, tf_dec_input], outputs=[dec_h, dec_c, preds])
##Attention! for 2 different architectures for encodder and decoder, last enc_lstm_cell shout
## have same nr units as first dec_lstm_cell
"""

dec_lstm_cells = []
tf_dec_input = Input((None, 32))
crt_tf_dec_input = tf_dec_input
DecLSTMCell = LSTM(units=lstm_architecture[0], return_sequences=True, return_state=True)
crt_tf_dec_input, dec_h, dec_c = DecLSTMCell(crt_tf_dec_input, initial_state=enc_full_state)
dec_lstm_cells.append((DecLSTMCell, dec_h, dec_c))
for i in range(1, len(lstm_architecture)):
  DecLSTMCell = LSTM(units=lstm_architecture[i], return_sequences=True, return_state=True)
  crt_tf_dec_input, dec_h, dec_c = DecLSTMCell(crt_tf_dec_input)
  dec_lstm_cells.append((DecLSTMCell, dec_h, dec_c)) ## dec_h, dec_c nenecesar
SMLayer = Dense(units=vocab_size, activation='softmax')
tf_dec_preds = SMLayer(crt_tf_dec_input)

model = Model(inputs=[tf_enc_input, tf_dec_input], outputs=[tf_dec_preds])
encoder = Model(inputs=[tf_enc_input], outputs=enc_full_state)


inputs = []
outputs = []
crt_dec_in = tf_dec_input
for i in range(len(dec_lstm_cells)):
  in1 = Input((lstm_architecture[i],), name='inh_' + str(i+1))
  in2 = Input((lstm_architecture[i],), name='inc_' + str(i+1))
  inputs.append(in1)
  inputs.append(in2)

  crt_dec_in, dec_h_i, dec_c_i = dec_lstm_cells[i][0](crt_dec_in, initial_state=[in1,in2])
  outputs.append(dec_h_i)
  outputs.append(dec_c_i)
preds = SMLayer(crt_dec_in)

model2 = Model(inputs=inputs+[tf_dec_input], outputs=outputs+[preds])
"""