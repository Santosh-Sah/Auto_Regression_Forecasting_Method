# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:52:57 2020

@author: Santosh Sah
"""

from AutoRegressionForecastingMethodUtils import (importAutoRegressionForecastingMethodDataset, saveAutoRegressionForecastingMethodForecastedValues,
                                               readAutoRegressionForecastingMethodForecastedValues, readAutoRegressionForecastingMethodModelForFullDataset)
from AutoRegressionForecastingMethodVisualization import visualizeAutoRegressionForecastingMethodForecastedValues

def forecastAutoRegressionForecastingMethodModel():
    
    autoRegressionForecastingMethodDataset = importAutoRegressionForecastingMethodDataset("uspopulation.csv")
    
    start = len(autoRegressionForecastingMethodDataset)
    
    end = len(autoRegressionForecastingMethodDataset) + 12
    
    #reading the model whichis trained on the whole dataset
    autoRegressionForecastingMethodModel = readAutoRegressionForecastingMethodModelForFullDataset()
    
    #fit the model with lag 11. 
    #when we have created the model without lag the AR figure out the number of optimum lag.
    autoRegressionForecastingMethodFinalModel = autoRegressionForecastingMethodModel.fit(maxlag = 11, method = "mle")
    
    #predicting vlaues for 12 month
    autoRegressionForecastingMethodForecastedValues = autoRegressionForecastingMethodFinalModel.predict(start = start, end = end, dynamic = False).rename("Forecast")
    
    #saving the forecasted values
    saveAutoRegressionForecastingMethodForecastedValues(autoRegressionForecastingMethodForecastedValues)

def plotAutoRegressionForecastingMethodForecastedValues():
    
    #reading the dataset
    autoRegressionForecastingMethodDataset = importAutoRegressionForecastingMethodDataset("uspopulation.csv")
    
    #reading the forecated values
    autoRegressionForecastingMethodForecastedValues = readAutoRegressionForecastingMethodForecastedValues()
    
    #visualizing the forecated values
    visualizeAutoRegressionForecastingMethodForecastedValues(autoRegressionForecastingMethodDataset, autoRegressionForecastingMethodForecastedValues)

if __name__ == "__main__":
    #forecastAutoRegressionForecastingMethodModel()
    plotAutoRegressionForecastingMethodForecastedValues()
    