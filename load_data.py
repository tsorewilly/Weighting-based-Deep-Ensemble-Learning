import os
import numpy as np
#import tensorflow as tf
import json
import pandas as pd


#LOAD ALL THE TRAINING SEQUENCE DATA
# TrainData = pd.read_excel (r'TrainSet-01.xlsx')
# TrainData = np.array(TrainData, np.float)
#path = os.getcwd()
#xls_files = os.listdir(path)
#training_files = [f for f in xls_files if f[0:8] == 'TrainSet']

#CONCATENATE THE LOADED TRAINING SEQUENCES
#df = pd.DataFrame()
#for f in training_files:
#    data = pd.read_excel(f, 'Sheet1')
#    print(len(data))
#    df = df.append(data)
#
#TrainData = np.array(df, np.float) #Convert into a numpy array
#print(len(TrainData))

#LOAD ALL THE TEST SEQUENCE DATA AND THE IDS
#TrainData = pd.read_excel ('AllTrainData.xlsx', sheet_name=['Sheet1', 'Sheet2', 'Sheet3', 'Sheet4', 'Sheet5'])
TrainSet = pd.read_csv('TrainSet.csv')  
TrainData = np.array(TrainSet, np.float)

#TestData = pd.read_excel (r'TestSet.xlsx')
TestSet = pd.read_csv('TestSet.csv')  
TestData = np.array(TestSet, np.float)

#TrainDataSize = np.array(pd.read_excel (r'TrainIds.xlsx'), np.int)+1
TrainDataSize = pd.read_csv('TrainIds.csv')  

#TestDataSize = np.array(pd.read_excel (r'TestIds.xlsx'), np.int)+1
TestDataSize = pd.read_csv('TestIds.csv')  

#Arrange the data into sequence of (L-Seg-Mus) training and (L-1) test datasets
print([len(TrainData), len(TestData), len(TrainDataSize), len(TestDataSize)])
TrainSize = int((len(TrainData)+1)/900)
TrainData = TrainData.reshape(TrainSize, 4, 900)

TestSize = int((len(TestData)+1)/900)
TestData = TestData.reshape(TestSize, 4, 900)

# #dataset = tf.keras.preprocessing.timeseries_dataset_from_array(TrainData, TestData, sequence_length=900)
#
# print (dataset)
#
