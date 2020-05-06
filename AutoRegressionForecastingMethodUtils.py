# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:50:57 2020

@author: Santosh Sah
"""
import pandas as pd
import pickle
"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importAutoRegressionForecastingMethodDataset(autoRegressionForecastingMethodDatasetFileName):
    
    autoRegressionForecastingMethodDataset = pd.read_csv(autoRegressionForecastingMethodDatasetFileName,index_col='DATE',parse_dates=True)
    
    #the dataset is minthly dataset. Hence setting its frequency as monthly.
    autoRegressionForecastingMethodDataset.index.freq = "MS"
    
    return autoRegressionForecastingMethodDataset

#splitting dataset into training and testing set
def splitAutoRegressionForecastingMethodDataset(autoRegressionForecastingMethodDataset):
    
    #splitting the dataset into training and testing set.
    autoRegressionForecastingMethodTrainingSet = autoRegressionForecastingMethodDataset.iloc[:84]
    autoRegressionForecastingMethodTestingSet = autoRegressionForecastingMethodDataset.iloc[84:]
    
    return autoRegressionForecastingMethodTrainingSet, autoRegressionForecastingMethodTestingSet

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)

"""
read X_train from pickle file
"""
def readAutoRegressionForecastingMethodXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readAutoRegressionForecastingMethodXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
Save AutoRegressionForecastingMethod as a pickle file.
"""
def saveAutoRegressionForecastingMethodModel(autoRegressionForecastingMethodModel):
    
    #Write AutoRegressionForecastingMethodModel as a picke file
    with open("AutoRegressionForecastingMethodModel.pkl",'wb') as AutoRegressionForecastingMethodModel_Pickle:
        pickle.dump(autoRegressionForecastingMethodModel, AutoRegressionForecastingMethodModel_Pickle, protocol = 2)

"""
read AutoRegressionForecastingMethodMethod from pickle file
"""
def readAutoRegressionForecastingMethodModel():
    
    #load AutoRegressionForecastingMethodModel model
    with open("AutoRegressionForecastingMethodModel.pkl","rb") as AutoRegressionForecastingMethodModel:
        autoRegressionForecastingMethodModel = pickle.load(AutoRegressionForecastingMethodModel)
    
    return autoRegressionForecastingMethodModel

"""
Save AutoRegressionForecastingMethod as a pickle file.
"""
def saveAutoRegressionForecastingMethodModelForFullDataset(autoRegressionForecastingMethodModelForFullDataset):
    
    #Write AutoRegressionForecastingMethodModelForFullDataset as a picke file
    with open("AutoRegressionForecastingMethodModelForFullDataset.pkl",'wb') as AutoRegressionForecastingMethodModelForFullDataset_Pickle:
        pickle.dump(autoRegressionForecastingMethodModelForFullDataset, AutoRegressionForecastingMethodModelForFullDataset_Pickle, protocol = 2)

"""
read AutoRegressionForecastingMethod from pickle file
"""
def readAutoRegressionForecastingMethodModelForFullDataset():
    
    #load AutoRegressionForecastingMethodModelForFullDataset model
    with open("AutoRegressionForecastingMethodModelForFullDataset.pkl","rb") as AutoRegressionForecastingMethodModelForFullDataset:
        autoRegressionForecastingMethodModelForFullDataset = pickle.load(AutoRegressionForecastingMethodModelForFullDataset)
    
    return autoRegressionForecastingMethodModelForFullDataset

"""
save AutoRegressionForecastingMethodPredictedValues as a pickle file
"""

def saveAutoRegressionForecastingMethodPredictedValues(autoRegressionForecastingMethodPredictedValues):
    
    #Write AutoRegressionForecastingMethodPredictedValues in a picke file
    with open("AutoRegressionForecastingMethodPredictedValues.pkl",'wb') as autoRegressionForecastingMethodPredictedValues_Pickle:
        pickle.dump(autoRegressionForecastingMethodPredictedValues, autoRegressionForecastingMethodPredictedValues_Pickle, protocol = 2)

"""
read AutoRegressionForecastingMethodPredictedValues from pickle file
"""
def readAutoRegressionForecastingMethodPredictedValues():
    
    #load AutoRegressionForecastingMethodPredictedValues
    with open("AutoRegressionForecastingMethodPredictedValues.pkl","rb") as autoRegressionForecastingMethodPredictedValues_pickle:
        autoRegressionForecastingMethodPredictedValues = pickle.load(autoRegressionForecastingMethodPredictedValues_pickle)
    
    return autoRegressionForecastingMethodPredictedValues

"""
save AutoRegressionForecastingMethodForecastedValues as a pickle file
"""

def saveAutoRegressionForecastingMethodForecastedValues(autoRegressionForecastingMethodForecastedValues):
    
    #Write AutoRegressionForecastingMethodForecastedValues in a picke file
    with open("AutoRegressionForecastingMethodForecastedValues.pkl",'wb') as autoRegressionForecastingMethodForecastedValues_Pickle:
        pickle.dump(autoRegressionForecastingMethodForecastedValues, autoRegressionForecastingMethodForecastedValues_Pickle, protocol = 2)

"""
read AutoRegressionForecastingMethodForecastedValues from pickle file
"""
def readAutoRegressionForecastingMethodForecastedValues():
    
    #load AutoRegressionForecastingMethodForecastedValues
    with open("AutoRegressionForecastingMethodForecastedValues.pkl","rb") as autoRegressionForecastingMethodForecastedValues_pickle:
        autoRegressionForecastingMethodForecastedValues = pickle.load(autoRegressionForecastingMethodForecastedValues_pickle)
    
    return autoRegressionForecastingMethodForecastedValues

"""
save AutoRegressionForecastingMethodForecastedValuesForLag1 as a pickle file
"""

def saveAutoRegressionForecastingMethodForecastedValuesForLag1(autoRegressionForecastingMethodForecastedValuesForLag1):
    
    #Write AutoRegressionForecastingMethodForecastedValuesForLag1 in a picke file
    with open("AutoRegressionForecastingMethodForecastedValuesForLag1.pkl",'wb') as autoRegressionForecastingMethodForecastedValuesForLag1_Pickle:
        pickle.dump(autoRegressionForecastingMethodForecastedValuesForLag1, autoRegressionForecastingMethodForecastedValuesForLag1_Pickle, protocol = 2)

"""
read AutoRegressionForecastingMethodForecastedValuesForLag1 from pickle file
"""
def readAutoRegressionForecastingMethodForecastedValuesForLag1():
    
    #load AutoRegressionForecastingMethodForecastedValuesForLag1
    with open("AutoRegressionForecastingMethodForecastedValuesForLag1.pkl","rb") as autoRegressionForecastingMethodForecastedValuesForLag1_pickle:
        autoRegressionForecastingMethodForecastedValuesForLag1 = pickle.load(autoRegressionForecastingMethodForecastedValuesForLag1_pickle)
    
    return autoRegressionForecastingMethodForecastedValuesForLag1

"""
save AutoRegressionForecastingMethodForecastedValuesForLag2 as a pickle file
"""

def saveAutoRegressionForecastingMethodForecastedValuesForLag2(autoRegressionForecastingMethodForecastedValuesForLag2):
    
    #Write AutoRegressionForecastingMethodForecastedValuesForLag2 in a picke file
    with open("AutoRegressionForecastingMethodForecastedValuesForLag2.pkl",'wb') as autoRegressionForecastingMethodForecastedValuesForLag2_Pickle:
        pickle.dump(autoRegressionForecastingMethodForecastedValuesForLag2, autoRegressionForecastingMethodForecastedValuesForLag2_Pickle, protocol = 2)

"""
read AutoRegressionForecastingMethodForecastedValuesForLag2 from pickle file
"""
def readAutoRegressionForecastingMethodForecastedValuesForLag2():
    
    #load AutoRegressionForecastingMethodForecastedValuesForLag2
    with open("AutoRegressionForecastingMethodForecastedValuesForLag2.pkl","rb") as autoRegressionForecastingMethodForecastedValuesForLag2_pickle:
        autoRegressionForecastingMethodForecastedValuesForLag2 = pickle.load(autoRegressionForecastingMethodForecastedValuesForLag2_pickle)
    
    return autoRegressionForecastingMethodForecastedValuesForLag2

"""
save AutoRegressionForecastingMethodForecastedValuesWithoutLag as a pickle file
"""

def saveAutoRegressionForecastingMethodForecastedValuesWithoutLag(autoRegressionForecastingMethodForecastedValuesWithoutLag):
    
    #Write AutoRegressionForecastingMethodForecastedValuesWithoutLag in a picke file
    with open("AutoRegressionForecastingMethodForecastedValuesWithoutLag.pkl",'wb') as autoRegressionForecastingMethodForecastedValuesWithoutLag_Pickle:
        pickle.dump(autoRegressionForecastingMethodForecastedValuesWithoutLag, autoRegressionForecastingMethodForecastedValuesWithoutLag_Pickle, protocol = 2)

"""
read AutoRegressionForecastingMethodForecastedValuesWithoutLag from pickle file
"""
def readAutoRegressionForecastingMethodForecastedValuesWithoutLag():
    
    #load AutoRegressionForecastingMethodForecastedValuesWithoutLag
    with open("AutoRegressionForecastingMethodForecastedValuesWithoutLag.pkl","rb") as autoRegressionForecastingMethodForecastedValuesWithoutLag_pickle:
        autoRegressionForecastingMethodForecastedValuesWithoutLag = pickle.load(autoRegressionForecastingMethodForecastedValuesWithoutLag_pickle)
    
    return autoRegressionForecastingMethodForecastedValuesWithoutLag



