from keras.layers.core import Permute
from keras.layers import Dense, Activation, RepeatVector, merge,Flatten, TimeDistributed, Input
from keras.layers import Embedding, LSTM
from keras.models import Model
from keras.optimizers import Adam

hidden = 225

features = get_features()
outputs = get_outputs()

features_length = len(features)
output_length = len(outputs)

inputs = Input(shape=(features_length, 100))
inp = Embedding(100, features_length, mask_zero=0)(inputs)
lstm_out = LSTM(hidden, return_sequences=True)(inputs)

attention = Dense(1, activation='elu')(lstm_out)
attention = Flatten()(attention)
attention = Activation('softmax')(attention)
attention = RepeatVector(hidden)(attention)
attention = Permute([2,1])(attention)

combined = merge([lstm_out, attention], mode='mul')
combined_mul = Flatten()(combined)
decode = RepeatVector(output_length)(combined_mul)
decode = LSTM(hidden, return_sequences=True)(decode)
decode = TimeDistributed(Dense(100))(decode)
decode = Activation('linear')(decode)

model = Model(inputs=[inputs], outputs=decode)
optimizer = Adam(lr=0.001, decay=.0001)
model.compile(loss='mse', optimizer=optimizer)

model.fit(features, outputs, epochs=100, batch_size=50)