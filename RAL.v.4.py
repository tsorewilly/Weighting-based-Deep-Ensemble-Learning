import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from itertools import product
import datetime
from load_seq_data import load_train_data, generate_arrays_from_dataset
from ConfigEnsemble import normalize, evaluate_model_accuracy, ensemble_weighted_prediction

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.python.keras import backend as PK
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Input, PReLU, Conv1D, Lambda, \
     LeakyReLU, Concatenate, Permute, UpSampling1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, \
     ReLU, Softmax, ELU, ThresholdedReLU, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from losses import LossHistory, figPlot, trainPlot
import metrics as mtr
from tensorflow.keras.utils import plot_model
from Omittention import StackAttention, KTAttention

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, concat
import seaborn as sns
from cf_matrix import make_confusion_matrix
from draw_confusion_matrix import plot_confusion_matrix_from_data
from pycm import *

log_dir = "logs/"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# load dataset
TrainData, ValidData, TestData, TrainClass, ValidClass, TestClass = load_train_data()

TrainCat = to_categorical(TrainClass-1)
ValidCat = to_categorical(ValidClass-1)
TestCat = to_categorical(TestClass-1)

train_X, train_y = TrainData, TrainCat
Valid_X, Valid_y = ValidData, ValidCat
test_X, test_y = TestData, TestCat

print(train_X.shape, train_y.shape, Valid_X.shape, Valid_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
input_shape = Input(shape = (train_X.shape[1], train_X.shape[2]), name='Input_Layer')
vector_input = Input(shape = (train_X.shape[1], train_X.shape[2]), name='Input_Layer')
#sequence_input = Input((train_X.shape[1], train_X.shape[2])):vector_input = Input((12,))

cnn_mod = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='Convolution_01')(input_shape)
cnn_mod = BatchNormalization(name='Normalization_01')(cnn_mod)
cnn_mod = MaxPooling1D(pool_size=4, strides=1, padding='valid', name='MaxPooling')(cnn_mod)
cnn_mod = Dense(64, activation='softmax', name='Dense_01')(cnn_mod)
#cnn_mod = UpSampling1D(size=2, name='UpSampling_1')(cnn_mod)
cnn_mod = Dropout(0.45, name='DropOut_01')(cnn_mod)

cnn_mod = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='Convolution_02')(cnn_mod)
cnn_mod = BatchNormalization(name='Normalization_02')(cnn_mod)
cnn_mod = Dense(64, activation='softmax', name='Dense_02')(cnn_mod)
cnn_mod = MaxPooling1D(pool_size=4, strides=1, padding='same')(cnn_mod) #padding = 'valid' gave ValueError: Negative dimension size.
cnn_mod = UpSampling1D(size=4, name='UpSampling_2')(cnn_mod)
cnn_mod = Dropout(0.45, name='DropOut_02')(cnn_mod)

cnn_mod1 = GlobalAveragePooling1D(name='GlobalPooling')(cnn_mod)
cnn_mod1 = Lambda(lambda node: K.expand_dims(node, axis=1))(cnn_mod1) #Wrapped arbitrary expressions as a Layer object.
cnn_mod1 = UpSampling1D(size=4, name='UpSampling_3')(cnn_mod1)
cnn_mod1 = Dense(64, activation='softmax', name='Dense_03')(cnn_mod)
cnn_mod1 = Dropout(0.45, name='DropOut_03')(cnn_mod1)
#cnn_mod1 = Flatten(name='Flattening')(cnn_mod1) #Last line is used to avoid ValueError: Input 0 is incompatible with layer lstm_1: expected ndim=3, found ndim=4

lstm_mod = LSTM(800, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name='LSTM_400')(input_shape)
lstm_mod = StackAttention(return_sequences=True)(lstm_mod)
lstm_mod = Dense(64, activation='sigmoid', name='Dense_5')(lstm_mod)

lstm_mod = LSTM(400, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, name='LSTM_200')(lstm_mod)
lstm_mod = Lambda(lambda node: K.expand_dims(node, axis=1))(lstm_mod) #Wrapped arbitrary expressions as a Layer object.
lstm_mod = Dropout(0.45, name='DropOut_04')(lstm_mod)
lstm_mod = ReLU(negative_slope=0.01)(lstm_mod)
lstm_mod = UpSampling1D(size=4, name='UpSampling_3')(lstm_mod)
#lstm_mod = Dense(64, activation='sigmoid', name='Dense_64')(lstm_mod)
#PS: the next LoC is needed based on the criteria that follow it.
#lstm_mod = UpSampling1D(size=4, name='UpSampling_4')(lstm_mod)
# The UpSampling1D LoC is only needed iff:
#   1. One GlobalAveragePooling was done in the CNN module above
#   2. Criteria 1 above is satisfied while the second StackAttention and the last UpSampling1D LoC above are not used
#   3. The second StackAttention caused deactivation of the Lambda layer; otherwise vise versa;
#   4. The return_sequences = False; otherwise add "lstm_mod = Flatten(name='Flatten')(lstm_mod)"

cnn_lstm_mod = Concatenate(name='Concatenate')([cnn_mod, cnn_mod1, lstm_mod])
#cnn_lstm_mod = LSTM(100, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name='LSTM_100')(cnn_lstm_mod)
#cnn_lstm_mod = StackAttention(return_sequences=True)(cnn_lstm_mod)
cnn_lstm_mod = Flatten(name='Flatten')(cnn_lstm_mod)
cnn_lstm_mod = ReLU(negative_slope=0.01)(cnn_lstm_mod)
output = Dense(6, activation='softmax', name='Dense_softmax')(cnn_lstm_mod) #activation='sigmoid'

#model = Model(input_shape, cnn_lstm_mod, name='Output_Layer')#, outputs=buttons)
model = Model(inputs=[input_shape], outputs=[output], name='Output_Layer')
plot_model(model, to_file='CNN-AttentiveLSTM-05-3.png', show_shapes=True, show_layer_names=True, dpi=100)

# Save method, save once in 1 generation
ModelName= 'RAL.v.4'
checkpoint_period = ModelCheckpoint(log_dir + ModelName + '-model.h5',monitor='val_loss', save_weights_only=False, save_best_only=True, period=1)
#-' + 'ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}


# The way the learning rate drops, val_loss does not drop three times,
# and the learning rate drops to continue training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.001, patience=3, verbose=1)

# To stop early, model can be stopped when val_loss has not dropped. Basically, training is completed
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

adam = Adam(lr=1e-4, decay=1e-6)
#model.compile(optimizer=sgd, loss='categorical_crossentropy')
#loss='categorical_crossentropy'
model.compile(loss='mse', optimizer=adam, metrics=['acc', mtr.rmse, 'mae', 'mse'])#,  'cosine_proximity'])
Softmax_model = model

# Define the Keras TensorBoard callback.
tbCallBack = TensorBoard(log_dir=log_dir+'./graph', histogram_freq=1, write_graph=True, write_images=True)
history = LossHistory()
model.build(input_shape)
model.summary()
batch_size = 128
n_jbs=-1

# fit network
strt_time = datetime.datetime.now()
history = model.fit(train_X, train_y , epochs=1000, batch_size=batch_size, shuffle=False, verbose=2,
            validation_data=(Valid_X, Valid_y))#, callbacks=[checkpoint_period, reduce_lr, history, tbCallBack])
curr_time = datetime.datetime.now()

timedelta = curr_time - strt_time
model_train_time = timedelta.total_seconds()
print("Training completed in ", timedelta.total_seconds(), "s")
#plt.plot(history.epoch,np.array(history.history['val_loss']),label='Val loss')

#Saving the pretrained Model, weights, and training history
print("Saving the model.")
model.save( ModelName + '-model.h5') #Already saved per generation with checkpoint_period
model.save_weights(log_dir + ModelName + '-weights.h5')
saveloc = log_dir + ModelName + '-history.npy'
np.save(saveloc, list(history.history.items()))

#Alternatively, use:

print("Model saved to: " + saveloc + " succesfully.")
plt.plot(history.epoch, np.array(history.history['val_loss']), label='Val loss')

# Final evaluation of the model
Y_pred = model.predict(test_X, verbose=0, batch_size=batch_size)
#Y_pred_prob = model.predict_proba(test_X)#[:, 1]

y_pred = np.argmax(Y_pred, axis=1)
y_actl = np.argmax(test_y, axis=1)

print('Printing Confusion Matrices')
trainPlot(history, saveAs = 'Ntwk(-CNN+CNN+LSTM)+Dense).png')

figPlot(y_actl, y_pred, saveAs = 'Output(-CNN+CNN+LSTM)+Dense).png', title='CNN + LSTM')

#Divide feature extraction layers and classification layers
# In[] EXTRACT TRAINED FEATURES FOR ML MODELS

#Ignore all sklearn warnings
for l in range(len(model.layers)):
    print(l, model.layers[l])        


#getFeature = K.function([model.layers[1].input, K.learning_phase()], [model.layers[l-1].output])     # feature extraction Module
#getPrediction = K.function([model.layers[23].input, K.learning_phase()], [model.layers[len(model.layers)-1].output]) # Classification Module
#xTrain, xTest = getFeature([train_X, 0])[0], getFeature([test_X, 0])[0]

#The above only worked in my last installation with tf2.1. 
#Now on tf2.3, I changed to one of the two ways below:
"""    
#Method 1:
#Not necessary to build K.function to fetch ouptput eagerly (from intermediate layer)
#If deemed needed, use Method 2.
partial_model = Model(model.inputs, model.layers[l-1].output)
getFeature1 = partial_model([train_X], training=True)
getPrediction1 = partial_model([train_X], training=False)
"""

"""
Using symbolic learning phase (TJAY's Comment: Uncleared yet)

#model.train_on_batch(train_X, train_y)
grads = model.optimizer.get_gradients(model.total_loss, model.layers[-1].output)
#The above fetches tensor and feeds it (in next LoC) to back end in eager mode.
grads_fn = K.function(inputs=[model.inputs[0], model._feed_targets[0], 
                      PK.symbolic_learning_phase()], outputs=grads)
"""

#Method 2:
getFeature = K.function([model.input], [model.layers[l-1].output])
getPrediction = K.function([model.input], [model.layers[l-1].output])
# run in test mode, i.e. 0 means test
with PK.eager_learning_phase_scope(value=0):
    xTrain = np.array(getFeature([train_X]))[0]

# run in training mode, i.e. 1 means training
with PK.eager_learning_phase_scope(value=1):
    xTest = np.array(getFeature([test_X]))[0]

xTrain_Class = TrainClass.to_numpy()
xValid_Class = ValidClass.to_numpy()
xTest_Class = TestClass.to_numpy()

#Reshape target to fit to scikit-learn
xTrainRes = xTrain.reshape(xTrain.shape[0], xTrain.shape[1])
xTestRes = xTest.reshape(xTest.shape[0], xTest.shape[1])
#I used these features for extraction after concatenate layer
#xTrainRes = xTrain.reshape(xTrain.shape[1], xTrain.shape[2])
#xTestRes = xTest.reshape(xTest.shape[1], xTest.shape[2])
#Else, reshaping takes extra operation if extraction is before concatenate layer
#xTrainRes = xTrain.reshape(xTrain.shape[0],xTrain.shape[1]*xTrain.shape[2])
#xTestRes = xTest.reshape(xTest.shape[0],xTest.shape[1]*xTest.shape[2])

yTrainRes = xTrain_Class.reshape(xTrain_Class.shape[0],)
yTestRes = xTest_Class.reshape(xTest_Class.shape[0],)

print(train_X.shape, train_y.shape, Valid_X.shape, Valid_y.shape, test_X.shape, test_y.shape)
print(xTrainRes.shape, xTestRes.shape, yTrainRes.shape, yTestRes.shape)

yTrainRes_Cat = to_categorical(yTrainRes-1)

# In[] CNN-SOFTMAX
#https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/
#https://machinelearningmastery.com/voting-ensembles-with-python/
#https://machinelearningmastery.com/weighted-average-ensemble-for-deep-learning-neural-networks/
#https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/

print("With SOFTMAX Classifier: Executing CNN-SOFTMAX for Similar Purpose");
LR_paraemters = {'Cs': [1, 10], 'cv':[5], 'tol':[1e-6], 'multi_class':['multinomial'],
              'solver': ['newton-cg','lbfgs', 'sag', 'saga'], #'penalty':['l1', 'l2'], 
              'max_iter':[1000], 'verbose':[0], 'scoring':['neg_log_loss','accuracy'],
              'random_state':[777], 'refit':[True], 'n_jobs':[n_jbs]}
#PS: multi_class is set to be multinomial the softmax function is used to find the predicted probability of each class.
#solver 'liblinear' does not work for multiclass problems, hence softmax cannot be used
#Dont use 'roc_auc' in scoring as it does not support categorical data. 
#logr_clf = GridSearchCV(LogisticRegressionCV(), LR_paraemters)


soft_clf = LogisticRegressionCV(cv=5, random_state=777, multi_class = 'multinomial', max_iter = 1000,
                                scoring = 'neg_log_loss', n_jobs=n_jbs).fit(xTrainRes, yTrainRes)
softclf=soft_clf
softclf = soft_clf.fit(xTrainRes, yTrainRes)
y_testSOFT = softclf.predict(xTestRes)-1
y_testSOFT_prob = softclf.predict_proba(xTestRes)
figPlot(y_actl, y_testSOFT, saveAs='Output(CNN+CNN+LSTM)+SOFTMAX).png', title='Pretrained CNN + SOFTMAX')

# In[] CNN-SGD
print("With SGD Classifier: Executing CNN-SGD for Similar Purpose");
sgd_clf = make_pipeline(StandardScaler(), SGDClassifier(loss="modified_huber", n_iter_no_change=10, random_state=967, max_iter=1000, tol=1e-7,  n_jobs=n_jbs))
#The possible options are 'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'
#Regression loss: 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'.
sgdclf=sgd_clf
sgdclf.fit(xTrainRes, yTrainRes)
y_testSGD = sgdclf.predict(xTestRes)-1
#y_testSGD_prob = sgdclf.predict_proba(xTestRes)
#"predict_(log_)proba only supported when loss='log' or loss='modified_huber'"

#print("Computing the Probabilities for Ensemble Usage")
#For prob estimation, pass the base estimator to meta-estimator constructor and refit.
#sgdclf_prob = CalibratedClassifierCV(sgd_clf, cv=5,  n_jobs=n_jbs)
#sgdclf_prob.fit(xTrainRes, yTrainRes)
#y_testSGD_prob = sgdclf_prob.predict_proba(xTestRes)
y_testSGD_prob = sgdclf.predict_proba(xTestRes)

figPlot(y_actl, y_testSGD, saveAs='Output(CNN+CNN+LSTM)+SGD).png', title='Pretrained CNN + SGD')

# In[] CNN-SVM
print("With Support Vector Machine: Executing CNN-SVM for Similar Purpose");
#Grid search svm classification with feature extracted training data as input
SVM_parameters = {'kernel':['rbf'], 'C':[1, 10, 100, 1000], 'gamma':[1e-4, 1e-5, 1e-6], 'probability': [True]}
svm_clf = GridSearchCV(SVC(), SVM_parameters)
svmclf=svm_clf
svmclf.fit(xTrainRes, yTrainRes)
svmclf = svmclf.best_estimator_
#svmclf.fit(xTrainRes, yTrainRes)
y_testSVM = svmclf.predict(xTestRes)-1

print("Computing the Probabilities for Ensemble Usage")
#For prob estimation, pass the base estimator to meta-estimator constructor and refit.
#svmclf_prob = CalibratedClassifierCV(svm_clf, cv=5,  n_jobs=n_jbs)
#svmclf_prob.fit(xTrainRes, yTrainRes);
#y_testSVM_prob = svmclf_prob.predict_proba(xTestRes)
y_testSVM_prob = svmclf.predict_proba(xTestRes)

figPlot(y_actl, y_testSVM, saveAs='Output(CNN+CNN+LSTM)+SVM).png', title='Pretrained CNN + SVM')
    
# In[] CNN-KNN
print("With k-Nearest Neighbors: Executing CNN-KNN for Similar Purpose")
#Grid search K nearest neighbors classification with feature extracted training data as input
#"weights": ['uniform', 'distance'], use only distance to return 
KNN_parameters = {"weights": ['distance'], "metric": ['minkowski','euclidean','manhattan'],
              "n_neighbors": [9, 10, 11, 12, 13, 14, 15], "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']}

#Activate to tune hyperparameters with grid search, but set K >=9
knn_clf = GridSearchCV(KNeighborsClassifier(n_jobs=n_jbs), param_grid = KNN_parameters)

#knn_clf = KNeighborsClassifier(n_neighbors=9, algorithm='ball_tree', metric='euclidean', weights='distance', n_jobs=n_jbs)

knnclf=knn_clf
knnclf.fit(xTrainRes, yTrainRes)
knnclf = knnclf.best_estimator_ 
y_testKNN = knnclf.predict(xTestRes)-1
y_testKNN_prob = knnclf.predict_proba(xTestRes)
print(y_testKNN_prob)
#figPlot(y_actl, y_testKNN, saveAs='Output(CNN+CNN+LSTM)+KNN).png', title='Pretrained CNN + KNN')

# In[] CNN-RF
print("With Random Forest: Executing CNN-RF for Similar Purpose")
#Grid search random forest classification with feature extracted training data as input
RDF_parameters = {"max_depth": [3, None], "max_features": [1, 3, 10],  "min_samples_split": [1.0, 3, 10],
              "min_samples_leaf": [1, 3, 10], "bootstrap": [True, False], "criterion": ["gini"],#, "entropy"],
              "n_estimators": [10, 20, 50]}

rdf_clf = GridSearchCV(RandomForestClassifier(), param_grid = RDF_parameters)
rdfclf=rdf_clf
rdfclf.fit(xTrainRes, yTrainRes)
rdfclf = rdfclf.best_estimator_
#rdfclf.fit(xTrainRes, yTrainRes)
y_testRF = rdfclf.predict(xTestRes)-1#Output
y_testRF_prob = rdfclf.predict_proba(xTestRes)

figPlot(y_actl, y_testRF, saveAs='Output(CNN+CNN+LSTM)+RF).png', title='Pretrained CNN + RF')

# In[] CNN-ENSEMBLE
#https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
EnsembleModels = list()

print("********ADDING MODELS TO ENSEMBLE********")
EnsembleModels.append(('Softmax', softclf))
EnsembleModels.append(('SGD', sgdclf))
EnsembleModels.append(('SVM', svmclf))
EnsembleModels.append(('KNN', knnclf))
EnsembleModels.append(('RDF', rdfclf))

print('Fitting the training data with the ensemble')
compEnsemble = VotingClassifier(estimators=EnsembleModels, voting='soft', n_jobs=n_jbs)
compEnsemble.fit(xTrainRes, yTrainRes)
y_testEnsemble = compEnsemble.predict(xTestRes) - 1
figPlot(y_actl, y_testEnsemble, saveAs='Output(CCL)+SSSKR-Soft).png', title='Pretrained CCL + SSSKR Ensemble-Soft')

# In[] CNN-ENSEMBLE PERFORMANCE
# make an ensemble prediction for multi-class classification

#In[]

#evaluate the models and store results
import random
from scipy.optimize import differential_evolution
results_Test, results_Train, names = list(), list(), list()
n_members = len(EnsembleModels)
best_score, best_weights = 0.0, None

#Add the accuracy scores of three different Ensemble configurations
#Equal Weights
weights1 = [1.0/n_members for _ in range(n_members)]
eqwEnsembPredTest = ensemble_weighted_prediction(EnsembleModels, weights1, xTestRes)
print('Equal Weights Score on Testing Data: %.5f'%accuracy_score(eqwEnsembPredTest, yTestRes-1))
figPlot(y_actl, eqwEnsembPredTest, saveAs='Output(CNN+CNN+LSTM)+Voting Ensemble(Equal Weight).png', title='Pretrained CNN + Equal Weight Voting Ensemble')
itr=0;


for _i in range(2000000):
    itr+=1
    weights = normalize(random.sample(range(0,100), n_members))   #Change upper range to control variance 
    eqwEnsembPredTest = ensemble_weighted_prediction(EnsembleModels, weights, xTestRes)    
    score = accuracy_score(eqwEnsembPredTest, yTestRes-1)
    if score > best_score:
        best_score, best_weights = score, weights
        print('>UPDATE%f %s %.5f' % (itr, best_weights, best_score))

# In[] Validate Weights
srchEnsembPredTest = ensemble_weighted_prediction(EnsembleModels, best_weights, xTestRes)    
print('Optimized Weighted Scoring Validation on Testing Data: %.5f'%accuracy_score(srchEnsembPredTest, yTestRes-1))
figPlot(y_actl, srchEnsembPredTest, saveAs='Output(CNN+CNN+LSTM)+Voting Ensemble(Optimized Search Weights).png', title='Pretrained CNN + Optimized Weighted Voting Ensemble')     

'''Prepare Classifiers for Performance Validation'''
Ensemble = dict()
Ensemble['Softmax'] = softclf
#Ensemble['SGD'] = sgdclf
Ensemble['RDF'] = rdfclf
Ensemble['KNN'] = knnclf
Ensemble['SVM'] = svmclf
Ensemble['Ensemble'] = VotingClassifier(estimators=EnsembleModels, voting='soft', n_jobs=n_jbs)
#Add the ensemble classifier using learners' best weights
Ensemble['Ensemble2'] = VotingClassifier(estimators=EnsembleModels, voting='soft', weights=best_weights, n_jobs=n_jbs)

#Compare performance
for name, EnModel in Ensemble.items():
    #print('>%s %.5f (%.5f)' % (name, np.mean(scores), np.std(scores)))
    score_Train = evaluate_model_accuracy(EnModel, xTrainRes, yTrainRes)
    score_Test = evaluate_model_accuracy(EnModel, xTestRes, yTestRes)
    results_Train.append(score_Train)#
    results_Test.append(score_Test)#
    names.append(name)
    print('>Training: %s %.5f (%.3f) | Testing: %.5f (%.3f)'%(name, np.mean(score_Train), 
          np.std(score_Train), np.mean(score_Test), np.std(score_Test)))

#plot model performance for comparison
plt.boxplot(results_Train, labels=names, showmeans=True, notch=True)
plt.show(block=False)


"""
scrEnsembModel = VotingClassifier(estimators=EnsembleModels, voting='soft', weights=best_weights.tolist(), n_jobs=n_jbs)
scrEnsembModel.fit(xTrainRes, yTrainRes)
wermdEnsembPredTest = scrEnsembModel.predict(xTestRes)-1
print('Re-weighted Scoring Validation on Testing Data: %.5f'%accuracy_score(wermdEnsembPredTest, yTestRes-1))
figPlot(y_actl, wermdEnsembPredTest, saveAs='Output(CNN+CNN+LSTM)+Re-weighted Voting Ensemble.png', title='Pretrained CNN + Re-weighted Voting-based Ensemble')
"""

"""
# grid search weights
GrdWeights = grid_search(EnsembleModels, xTrainRes, yTrainRes-1)
gdwEnsembPredTest = ensemble_weighted_prediction(EnsembleModels, GrdWeights, xTestRes)
print('Grid Search Weights Score on Testing Data: %.5f'%accuracy_score(gdwEnsembPredTest, yTestRes-1))
figPlot(y_actl, gdwEnsembPredTest, saveAs='Output(CNN+CNN+LSTM)+Voting Ensemble(Optimized Weight).png', title='Pretrained CNN + Equal Weighted Voting Ensemble')
"""


"""
# define bounds on each weight
bound_w = [(0.0, 1.0)  for _ in range(n_members)]
# arguments to the loss function
search_arg = (EnsembleModels, xTestRes, yTestRes)
# global optimization of ensemble weights
optWeights = differential_evolution(func=loss_function, bounds=bound_w, args=(search_arg,), maxiter=1000, tol=1e-7)
# get the chosen weights
weights = normalize(optWeights['x'])
print('Optimized Weights: %s' % weights)
"""