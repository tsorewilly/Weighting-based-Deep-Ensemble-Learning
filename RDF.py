import numpy as np
from losses import figPlot
from sklearn.ensemble import RandomForestClassifier
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


RDF_parameters = {"max_depth": [3, None], "max_features": [1, 3, 10],  "min_samples_split": [1.0, 3, 10],
              "min_samples_leaf": [1, 3, 10], "bootstrap": [True, False], "criterion": ["gini", "entropy"],
              "n_estimators": [10, 20, 50]}

rdf_clf = GridSearchCV(RandomForestClassifier(), param_grid = RDF_parameters)
rdfclf=rdf_clf
rdfclf.fit(xTrainRes, yTrainRes)
rdfclf = rdfclf.best_estimator_
y_testRF = rdfclf.predict(xTestRes)-1#Output
figPlot(y_actl, y_testRF, saveAs='Output(Random Forest).png', title='Random Forest')

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	print(scores)

scores = evaluate_model(rdfclf, xTrainRes, yTrainRes)