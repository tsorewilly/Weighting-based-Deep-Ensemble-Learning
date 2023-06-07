import numpy as np
from keras.utils import to_categorical
from losses import figPlot
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from load_seq_data import load_train_data, load_test_data

# load dataset
TrainData, ValidData, TestData, TrainClass, ValidClass, TestClass = load_train_data()
RCS_Data, RCS_DataClass = load_test_data(path='Data/', prefix='Robot-', MuscleChannels=4, SegLength=900)

xTrain, yTrain = TrainData, TrainClass
Valid_X, Valid_y = ValidData, ValidClass
xTest, yTest = TestData, TestClass

""""""
xTrain, xTest, yTrain, yTest = train_test_split(RCS_Data, RCS_DataClass, test_size=0.20, random_state=42)
train_X, Valid_X, train_y, Valid_y = train_test_split(xTrain, yTrain, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)


#Reshape target to fit to scikit-learn
xTrainRes = xTrain.reshape(xTrain.shape[0],xTrain.shape[1]*xTrain.shape[2])
xTestRes = xTest.reshape(xTest.shape[0],xTest.shape[1]*xTest.shape[2])

xTrain_Class = yTrain.to_numpy()
xValid_Class = Valid_y.to_numpy()
xTest_Class = yTest.to_numpy()

yTrainRes = xTrain_Class.reshape(xTrain_Class.shape[0],)
yTestRes = xTest_Class

y_actl = np.argmax(yTest, axis=1)

print(xTrain.shape, yTrain.shape, Valid_X.shape, Valid_y.shape, xTest.shape, yTest.shape)


print("With Support Vector Machine: Executing CNN-SVM for Similar Purpose");
# Grid search svm classification with feature extracted training data as input
SVM_parameters = {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [1e-4, 1e-5, 1e-6], 'probability': [True]}
svm_clf = GridSearchCV(SVC(), SVM_parameters)
svmclf = svm_clf
svmclf.fit(xTrainRes, yTrainRes)
svmclf = svmclf.best_estimator_
y_testSVM = svmclf.predict(xTestRes) - 1
figPlot(y_actl, y_testSVM, saveAs='Output(SVM).png', title='Support Vector Machine')


