# imports
#import tensorflow as tf
import numpy as np
import pandas as pd


# Data Dimensions
input_dim = 4           # input dimension i.e. no of features
seq_max_len = 900       # sequence length i.e. steps
out_dim = 1             # output dimension


def generate_data(count=1000, max_length=900, dim=4):
    x = np.random.randint(0, 10, size=(count, max_length, dim))
    length = np.random.randint(1, max_length+1, count)
    for i in range(count):
        x[i, length[i]:, :] = 0
    y = np.sum(x, axis=1)
    return x, y, length


def divide_train_data(dataset, dataclass, MuscleChannels=4, SegLength=900):
    TrainData = ValidData = TestData = dataset
    TrainClass = ValidClass = TestClass = dataclass

    # Arrange the data into sequence of (L-Seg-Mus) training and (L-1) test datasets
    # print([len(TrainData), len(TestData), len(TrainDataSize), len(TestDataSize)])
    TrainSize = int((len(TrainData) + 1) / SegLength)
    TrainData = TrainData.reshape(TrainSize, MuscleChannels, SegLength)

    ValidSize = int((len(ValidData) + 1) / SegLength)
    ValidData = ValidData.reshape(ValidSize, MuscleChannels, SegLength)

    TestSize = int((len(TestData) + 1) / SegLength)
    TestData = TestData.reshape(TestSize, MuscleChannels, SegLength)

    return TrainData, ValidData, TestData, TrainClass, ValidClass, TestClass


def load_train_data(path='', MuscleChannels=4, SegLength=900):
    # LOAD ALL THE TEST SEQUENCE DATA AND THE IDS
    TrainSet = pd.read_csv('Data/TrainSet.csv')
    TrainData = np.array(TrainSet, np.float)
    TrainClass = pd.read_csv('Data/TrainClass.csv')
    #TrainClass = np.array(TrainSet, np.int)

    ValidationSet = pd.read_csv('Data/ValidationSet.csv')
    ValidData = np.array(ValidationSet, np.float)
    ValidClass = pd.read_csv('Data/ValidationClass.csv')
    #ValidationClass = np.array(ValidationClass, np.int)

    TestSet = pd.read_csv('Data/TestSet.csv')
    TestData = np.array(TestSet, np.float)
    TestClass = pd.read_csv('Data/TestClass.csv')
    #TestClass = np.array(TestClass, np.int)

    TrainDataSize = pd.read_csv('Data/TrainId.csv')
    ValidDataSize = pd.read_csv('Data/ValidationId.csv')
    TestDataSize = pd.read_csv('Data/TestId.csv')

    # Arrange the data into sequence of (L-Seg-Mus) training and (L-1) test datasets
    #print([len(TrainData), len(TestData), len(TrainDataSize), len(TestDataSize)])
    TrainSize = int((len(TrainData) + 1) / SegLength)
    TrainData = TrainData.reshape(TrainSize, MuscleChannels, SegLength)

    ValidSize = int((len(ValidData) + 1) / SegLength)
    ValidData = ValidData.reshape(ValidSize, MuscleChannels, SegLength)

    TestSize = int((len(TestData) + 1) / SegLength)
    TestData = TestData.reshape(TestSize, MuscleChannels, SegLength)
    
    return TrainData, ValidData, TestData, TrainClass, ValidClass, TestClass#, TrainDataSize, ValidDataSize, TestDataSize

def load_test_data(path='', prefix = '', MuscleChannels=4, SegLength=900):
    # LOAD ALL THE TEST SEQUENCE DATA AND THE IDS
    TestSet = pd.read_csv(path+''+prefix+'TestSet.csv')
    TestData = np.array(TestSet, np.float)
    TestClass = pd.read_csv(path+''+prefix+'TestClass.csv')

    TestSize = int((len(TestData) + 1) / SegLength)
    TestData = TestData.reshape(TestSize, MuscleChannels, SegLength)
    return TestData,TestClass


def generate_arrays_from_dataset(TrainData, TrainClass, batch_size):
    n = len(TrainData)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(TrainData)
            X_train.append(TrainData())
            Y_train.append(TrainClass())
            i = (i + 1) % n
        yield (np.array(X_train), np.array(Y_train))