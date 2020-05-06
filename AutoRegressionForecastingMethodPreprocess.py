# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:51:38 2020

@author: Santosh Sah
"""

from AutoRegressionForecastingMethodUtils import (importAutoRegressionForecastingMethodDataset, saveTrainingAndTestingDataset, 
                                                  splitAutoRegressionForecastingMethodDataset)

def preprocess():
    
    autoRegressionForecastingMethodDataset = importAutoRegressionForecastingMethodDataset("uspopulation.csv")
    
    X_train, X_test = splitAutoRegressionForecastingMethodDataset(autoRegressionForecastingMethodDataset)
    
    saveTrainingAndTestingDataset(X_train, X_test)
    

if __name__ == "__main__":
    preprocess()