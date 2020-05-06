# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:53:28 2020

@author: Santosh Sah
"""
import pylab

def visualizeAutoRegressionForecastingMethodPredictedValues(X_test, ForecastedValuesForLag1, ForecastedValuesForLag2, ForecastedValuesWithoutLag):
    
    #plotting the predicted values, training set and testing set
    X_test["PopEst"].plot(legend = True)
    ForecastedValuesForLag1.plot(legend = True)
    ForecastedValuesForLag2.plot(legend = True)
    ForecastedValuesWithoutLag.plot(legend = True)
    
    pylab.savefig('PredeictedValuesForDifferentLags.png')

def visualizeAutoRegressionForecastingMethodForecastedValues(autoRegressionForecastingMethodDataset, autoRegressionForecastingMethodForecastedValues):
    
    #plotting the forecated values with full dataset
    autoRegressionForecastingMethodDataset["PopEst"].plot()
    
    autoRegressionForecastingMethodForecastedValues.plot()
    
    pylab.savefig('ForecastedValues.png')