import tensorflow as tf
import numpy as np
from keras import Model
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Input, Lambda,  Conv1D, Flatten, \
     LeakyReLU, Concatenate, UpSampling1D, MaxPooling1D, GlobalAveragePooling1D, PReLU
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from matplotlib import pyplot
from losses import LossHistory
import metrics as mtr
from keras.utils import plot_model
from Omittention import StackAttention, KTAttention
import sklearn.metrics as SKMmetrics
from load_seq_data import load_train_data, generate_arrays_from_dataset
from cf_matrix import make_confusion_matrix
from draw_confusion_matrix import plot_confusion_matrix_from_data
from keras.layers import Reshape, Permute, ReLU, Softmax, ELU, ThresholdedReLU, Bidirectional, TimeDistributed
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pandas import read_csv, DataFrame, concat
from attention import Attention
from keras.models import Sequential

log_dir = "logs/"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


'''Load the Dataset'''
TrainData, ValidData, TestData, TrainClass, ValidClass, TestClass = load_train_data()

TrainCat = to_categorical(TrainClass-1)
ValidCat = to_categorical(ValidClass-1)
TestCat = to_categorical(TestClass-1)

train_X, train_y = TrainData, TrainCat
Valid_X, Valid_y = ValidData, ValidCat
test_X, test_y = TestData, TestCat

'''Design the Network Graph'''
input_shape = Input(shape = (train_X.shape[1], train_X.shape[2]), name='Input_Layer')

'''Feature Extraction: CNN Module'''
cnn_mod = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='Convolution_01')(input_shape)
cnn_mod = BatchNormalization(name='Normalization_01')(cnn_mod)
#LeakyReLU,ReLU, Softmax, LeakyReLU, ELU, ThresholdedReLU
'''
cnn_mod = LeakyReLU(name='LeakyReLU_01')(cnn_mod)
cnn_mod = MaxPooling1D(pool_size=4, strides=1, padding='valid', name='MaxPooling')(cnn_mod)
cnn_mod = LeakyReLU(name='LeakyReLU_02')(cnn_mod)
'''
cnn_mod = Dropout(0.2, name='DropOut_01')(cnn_mod)

cnn_mod = UpSampling1D(size=4, name='UpSampling_1D')(cnn_mod)
cnn_mod = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='Convolution_02')(cnn_mod)
cnn_mod = BatchNormalization(name='Normalization_02')(cnn_mod)

cnn_mod = LeakyReLU(name='LeakyReLU_03')(cnn_mod)
cnn_mod = MaxPooling1D(pool_size=4, strides=1, padding='same')(cnn_mod) #padding = 'valid' gave ValueError: Negative dimension size.
cnn_mod = LeakyReLU(name='LeakyReLU_04')(cnn_mod)
#cnn_mod = GlobalAveragePooling1D(name='GlobalPooling')(cnn_mod)
#cnn_mod = Lambda(lambda node: K.expand_dims(node, axis=1))(cnn_mod) #Wrapped arbitrary expressions as a Layer object.
'Done to remove effect of GlobalAveragePooling1D which was needed for the Concatenation operation done later'

print(cnn_mod, cnn_mod.shape)
cnn_mod = UpSampling1D(size=4, name='UpSampling_2')(cnn_mod) #Only needed if GlobalAveragePooling was done above
cnn_mod = Dense(64, activation='softmax', name='Dense_01')(cnn_mod)
cnn_mod = Dropout(0.2, name='DropOut_CNN_1')(cnn_mod)

#Next line is used to avoid ValueError: Input 0 is incompatible with layer lstm_1: expected ndim=3, found ndim=4
#cnn_mod = Flatten(name='Flattening')(cnn_mod)
lstm_mod = LSTM(800, return_sequences=True, return_state=False, dropout=0.1, recurrent_activation='sigmoid', recurrent_dropout=0.1, name='LSTM_Feature_1')(input_shape)
#lstm_mod = Attention(name='attention_weight')(lstm_mod)# lstm_mod = AttentionLSTM(6)(input_shape)
lstm_mod = StackAttention(return_sequences=True)(lstm_mod)
lstm_mod = LeakyReLU(name='PReLU_1')(lstm_mod)
lstm_mod = Dense(64, activation='sigmoid', name='Dense_LSTM_1')(lstm_mod)
lstm_mod = Dropout(0.2, name='DropOut_LSTM_1')(lstm_mod)

lstm_mod = LSTM(100, return_sequences=True, return_state=False, dropout=0.1, recurrent_dropout=0.1, recurrent_activation='sigmoid', name='LSTM_Feature_2')(lstm_mod)
lstm_mod = StackAttention(return_sequences=True)(lstm_mod)
lstm_mod = LeakyReLU(name='PReLU_2')(lstm_mod)
lstm_mod = Dense(64, activation='sigmoid', name='Dense_LSTM_2')(lstm_mod)
lstm_mod = Dropout(0.2, name='DropOut_LSTM_2')(lstm_mod)

lstm_mod = LSTM(50, return_sequences=True, return_state=False, dropout=0.1, recurrent_dropout=0.1, recurrent_activation='sigmoid', name='LSTM_Feature_3')(lstm_mod)
lstm_mod = StackAttention(return_sequences=True)(lstm_mod)
lstm_mod = LeakyReLU()(lstm_mod)
lstm_mod = UpSampling1D(size=16, name='UpSampling_Feature_1')(lstm_mod)
lstm_mod = Dense(64, activation='sigmoid', name='Dense_LSTM_3')(lstm_mod)
lstm_mod = Dropout(0.2, name='DropOut_LSTM_3')(lstm_mod)

print(cnn_mod, cnn_mod.shape)
print(lstm_mod, lstm_mod.shape)
'''Module Hybridization for fusing extracted features.'''
cnn_lstm_mod = Concatenate(name='Concatenate')([cnn_mod, lstm_mod])

'''Prediction Module'''
cnn_lstm_mod = LSTM(200, return_sequences=False, return_state=False, dropout=0.1, recurrent_dropout=0.1, name='LSTM_Predicts')(cnn_lstm_mod)
cnn_lstm_mod = Lambda(lambda node: K.expand_dims(node, axis=1))(cnn_lstm_mod) #Wrapped arbitrary expressions as a Layer object.
cnn_lstm_mod = UpSampling1D(size=32, name='UpSample_Predicts')(cnn_lstm_mod) #Only needed if GlobalAveragePooling was done above

cnn_lstm_mod = Flatten(name='Flatten')(cnn_lstm_mod)
output = Dense(6, activation='softmax', name='Dense_softmax')(cnn_lstm_mod) #activation='sigmoid'
model = Model(inputs=[input_shape], outputs=[output], name='Output_Layer')
plot_model(model, to_file='CNN-AttentiveLSTM-02-2b.png', show_shapes=True, show_layer_names=True, dpi=300)


'''Save method, save once in 1 generation'''
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)

'''The way learning rate drops, val_loss does not drop three times, and learning rate drops to continue training'''
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-10, patience=5, verbose=1)

'''To stop early, model can be stopped when val_loss has not dropped. Basically, training is completed'''
early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=10,
                               verbose=1, mode = 'min', restore_best_weights=True)

# = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=1e-4, beta_1=1e-2, beta_2=1e-16, epsilon=1e-6, amsgrad=True)
#model.compile(optimizer=sgd, loss='categorical_crossentropy')
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc', mtr.rmse, 'mae', 'mse'])#,  'cosine_proximity'])

lossDetails = LossHistory()
print(train_X.shape, train_y.shape, Valid_X.shape, Valid_y.shape, test_X.shape, test_y.shape)
model.build(input_shape)
model.summary()

'''80% is used for training and 20% is used for estimation.'''
num_val  = int(train_X.shape[1]) * 0.2
num_train = train_X.shape[1] - num_val
bs, validationSplit = 128, 0.2

print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, bs))

# Start training
#history = model.fit_generator(generate_arrays_from_dataset(train_X, train_y, bs), steps_per_epoch=max(1, num_train // bs),
#        validation_data=generate_arrays_from_dataset(train_X, train_y, bs),
#        validation_steps=max(1, num_val // bs),
#        epochs=50, initial_epoch=0, callbacks=[checkpoint_period, reduce_lr, history])


history = model.fit(train_X, train_y , epochs=1000, batch_size=bs, shuffle=False,
        validation_data=(Valid_X, Valid_y))#, callbacks=[checkpoint, reduce_lr, early_stop, lossDetails])

#model.save_weights(log_dir + 'last1.h5')
'''plot history'''
fig, axs = pyplot.subplots(2, 3)
axs[0, 0].set_title('Training and Validation Acuraccy')
axs[0, 0].plot(history.history['acc'], label='Training')
axs[0, 0].plot(history.history['val_acc'], label='Validation')

axs[0, 1].set_title('Training and Validation Loss')
axs[0, 1].plot(history.history['loss'], label='Training')
axs[0, 1].plot(history.history['val_loss'], label='Validation')

axs[1, 0].set_title('Training and Validation RMSE')
axs[1, 0].plot(history.history['rmse'], label='Training')
axs[1, 0].plot(history.history['val_rmse'], label='Validation')

axs[1, 1].set_title('Training and Validation MAE')
axs[1, 1].plot(history.history['mae'], label='Training')
axs[1, 1].plot(history.history['val_mae'], label='Validation')

axs[1, 2].set_title('Training and Validation MSE')
axs[1, 2].plot(history.history['mse'], label='Training')
axs[1, 2].plot(history.history['val_mse'], label='Validation')

#axs[2, 2].set_title('Training and Validation Co-Proximity')
#axs[2, 2].plot(history.history['cosine_proximity'], label='Training')
#axs[2, 2].plot(history.history['val_cosine_proximity'], label='Validation')

pyplot.legend()
pyplot.show()

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

'''Hide x labels and tick labels for top plots and y ticks for right plots.'''
for ax in axs.flat:
    ax.label_outer()


'''Final evaluation of the model'''
Y_pred = model.predict(test_X, verbose=0, batch_size=bs)

#print("Accuracy: %.2f%%" % (Y_pred[1]*100))
y_pred = np.argmax(Y_pred, axis=1)
y_actl = np.argmax(test_y, axis=1)

print('Confusion Matrix')
cf_matrix = SKMmetrics.confusion_matrix(y_actl, y_pred)
print(cf_matrix)


labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
categories=['PUSH', 'PULL', 'CLRT', 'CCRT', 'CRPH', 'CCRP']
dispTitle = ''
make_confusion_matrix(cf_matrix, group_names=labels, categories=categories, cmap='Blues')


print('Classification Report')
print(SKMmetrics.classification_report(y_actl, y_pred))
#cm_analysis = ConfusionMatrix(actual_vector=y_actl, predict_vector=y_pred)
if(len(y_actl) > 10):
    fz=9; figsize=[14,14];
plot_confusion_matrix_from_data(y_actl, y_pred, columns = categories, annot = True, cmap = 'Oranges', fmt = '.2f', fz = 12,
                                lw=0.5, cbar = False, figsize = [9,9], show_null_values = 2, pred_val_axis = 'y')
#plt.pause(10010)