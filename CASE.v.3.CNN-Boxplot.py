import tensorflow as tf
import numpy as np
import datetime
from load_seq_data import load_train_data, generate_arrays_from_dataset
from keras import Model
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Input, ReLU, Conv1D, Lambda, \
     LeakyReLU, Concatenate, Permute, UpSampling1D, Flatten, MaxPooling1D, GlobalAveragePooling1D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from losses import LossHistory, figPlot, trainPlot
import metrics as mtr
from keras.utils import plot_model
from Omittention import StackAttention, KTAttention
from sklearn.model_selection import cross_val_score, RepeatedKFold, StratifiedKFold


def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=5, error_score='raise')
    return scores


log_dir = "logs/"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def myModel(input_shape):
    #sequence_input = Input((train_X.shape[1], train_X.shape[2])):vector_input = Input((12,))
    # design network
    model = Sequential()
    cnn_mod = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='Convolution_01')(input_shape)
    cnn_mod = BatchNormalization(name='Normalization_01')(cnn_mod)
    cnn_mod = MaxPooling1D(pool_size=4, strides=1, padding='valid', name='MaxPooling')(cnn_mod)
    cnn_mod = Dense(64, activation='softmax', name='Dense_01')(cnn_mod)
    #cnn_mod = UpSampling1D(size=2, name='UpSampling_1')(cnn_mod)
    cnn_mod = Dropout(0.25, name='DropOut_01')(cnn_mod)
    cnn_mod = LeakyReLU(alpha=0.01)(cnn_mod)

    cnn_mod = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='Convolution_02')(cnn_mod)
    cnn_mod = BatchNormalization(name='Normalization_02')(cnn_mod)
    cnn_mod = Dense(64, activation='softmax', name='Dense_02')(cnn_mod)
    cnn_mod = MaxPooling1D(pool_size=4, strides=1, padding='same')(cnn_mod) #padding = 'valid' gave ValueError: Negative dimension size.
    cnn_mod = UpSampling1D(size=4, name='UpSampling_2')(cnn_mod)
    cnn_mod = Dropout(0.25, name='DropOut_02')(cnn_mod)
    cnn_mod = LeakyReLU(alpha=0.01)(cnn_mod)

    cnn_mod1 = GlobalAveragePooling1D(name='GlobalPooling')(cnn_mod)
    cnn_mod1 = UpSampling1D(size=4, name='UpSampling_3')(cnn_mod1)
    cnn_mod1 = Dense(64, activation='softmax', name='Dense_03')(cnn_mod)
    cnn_mod1 = Dropout(0.25, name='DropOut_03')(cnn_mod1)
    cnn_mod1 = LeakyReLU(alpha=0.01)(cnn_mod1)

    #cnn_mod1 = Flatten(name='Flattening')(cnn_mod1) #Last line is used to avoid ValueError: Input 0 is incompatible with layer lstm_1: expected ndim=3, found ndim=4

    lstm_mod = LSTM(1600, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name='LSTM_400')(input_shape)
    #lstm_mod = StackAttention(return_sequences=True)(lstm_mod)
    lstm_mod = Dense(64, activation='sigmoid', name='Dense_5')(lstm_mod)
    lstm_mod = LeakyReLU(alpha=0.01)(lstm_mod)

    lstm_mod = LSTM(800, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, name='LSTM_200')(lstm_mod)
    lstm_mod = Lambda(lambda node: K.expand_dims(node, axis=1))(lstm_mod) #Wrapped arbitrary expressions as a Layer object.
    lstm_mod = Dropout(0.45, name='DropOut_04')(lstm_mod)
    lstm_mod = LeakyReLU(alpha=0.01)(lstm_mod)
    lstm_mod = UpSampling1D(size=4, name='UpSampling_3')(lstm_mod)

    cnn_lstm_mod = Concatenate(name='Concatenate')([cnn_mod, cnn_mod1, lstm_mod])
    cnn_lstm_mod = Flatten(name='Flatten')(cnn_lstm_mod)
    cnn_lstm_mod = ReLU(negative_slope=0.01)(cnn_lstm_mod)
    output = Dense(6, activation='softmax', name='Dense_softmax')(cnn_lstm_mod) #activation='sigmoid'

    #model = Model(input_shape, cnn_lstm_mod, name='Output_Layer')#, outputs=buttons)
    model = Model(inputs=[input_shape], outputs=[output], name='Output_Layer')
    plot_model(model, to_file='CASE-CNN-AttentiveLSTM-05-3.png', show_shapes=True, show_layer_names=True, dpi=100)

    # The way the learning rate drops, val_loss does not drop three times,
    # and the learning rate drops to continue training
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.001, patience=3, verbose=1)

    # To stop early, model can be stopped when val_loss has not dropped. Basically, training is completed
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    adam = Adam(lr=1e-4, decay=1e-6)
    #model.compile(optimizer=sgd, loss='categorical_crossentropy')
    #loss='categorical_crossentropy'
    model.compile(loss='mse', optimizer=adam, metrics=['acc', 'mae', 'mse'])#,  'cosine_proximity'])
    return model

    # load dataset
TrainData, ValidData, TestData, TrainClass, ValidClass, TestClass = load_train_data()
TrainCat = to_categorical(TrainClass-1)
ValidCat = to_categorical(ValidClass-1)
TestCat = to_categorical(TestClass-1)
train_X, train_y = TrainData, TrainCat
Valid_X, Valid_y = ValidData, ValidCat
test_X, test_y = TestData, TestCat
print(train_X.shape, train_y.shape, Valid_X.shape, Valid_y.shape, test_X.shape, test_y.shape)

input_shape = Input(shape = (train_X.shape[1], train_X.shape[2]), name='Input_Layer')
vector_input = Input(shape = (train_X.shape[1], train_X.shape[2]), name='Input_Layer')
batch_size = 128
model = KerasClassifier(build_fn=myModel(input_shape), epochs=2, batch_size=batch_size, verbose=1)

# fit network
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#epochs = [10, 50, 100]
#param_grid = dict(epochs=epochs, optimizer=optimizer)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', n_jobs=-1, refit='boolean')
#grid_result = grid.fit(X_train, Y_train, validation_data=(X_test, Y_test))
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

strt_time = datetime.datetime.now()
#scores = evaluate_model(model, train_X, train_y)
#print('>%s %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, TrainData, TrainClass,
        scoring='neg_mean_absolute_error', cv=2, n_jobs=5, error_score='raise')
curr_time = datetime.datetime.now()

timedelta = curr_time - strt_time
model_train_time = timedelta.total_seconds()
print("Training completed in ", timedelta.total_seconds(), "s")
