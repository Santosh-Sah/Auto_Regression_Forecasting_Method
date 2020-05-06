# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:51:54 2020

@author: Santosh Sah
"""
from statsmodels.tsa.ar_model import AR, ARResults

from AutoRegressionForecastingMethodUtils import (saveAutoRegressionForecastingMethodModel, readAutoRegressionForecastingMethodXTrain, 
                                               importAutoRegressionForecastingMethodDataset, saveAutoRegressionForecastingMethodModelForFullDataset)

"""
Train AutoRegressionForecastingMethod model on training set
"""
def trainAutoRegressionForecastingMethodModel():
    
    X_train = readAutoRegressionForecastingMethodXTrain()
    
    #training model on the training set
    autoRegressionForecastingMethodModel = AR(X_train["PopEst"])   
    
    #saving the model in pickle file
    saveAutoRegressionForecastingMethodModel(autoRegressionForecastingMethodModel)

"""
Train AutoRegressionForecastingMethod model on full dataset
"""
def trainAutoRegressionForecastingMethodModelOnFullDataset():
    
    autoRegressionForecastingMethodDataset = importAutoRegressionForecastingMethodDataset("uspopulation.csv")
    
    #training model on the whole dataset
    autoRegressionForecastingMethodModel = AR(autoRegressionForecastingMethodDataset["PopEst"])
    
    #saving the model in pickle files
    saveAutoRegressionForecastingMethodModelForFullDataset(autoRegressionForecastingMethodModel)

if __name__ == "__main__":
    #trainAutoRegressionForecastingMethodModel()
    trainAutoRegressionForecastingMethodModelOnFullDataset()    
