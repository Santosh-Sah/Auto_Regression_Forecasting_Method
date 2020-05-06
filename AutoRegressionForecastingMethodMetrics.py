# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:41:20 2020

@author: Santosh Sah
"""
from sklearn.metrics import mean_squared_error

from AutoRegressionForecastingMethodUtils import (readAutoRegressionForecastingMethodXTest,readAutoRegressionForecastingMethodForecastedValuesForLag1, 
                                                  readAutoRegressionForecastingMethodForecastedValuesForLag2, readAutoRegressionForecastingMethodForecastedValuesWithoutLag)

"""

calculating AutoRegressionForecastingMethod metrics

"""
def testAutoRegressionForecastingMethodMetrics():
    
    #reading testing set
    X_test = readAutoRegressionForecastingMethodXTest()
    
    #reading predicted value for lag1
    forecastedValuesForLag1 = readAutoRegressionForecastingMethodForecastedValuesForLag1()
    
    #reading predicted value for lag2
    forecastedValuesForLag2 = readAutoRegressionForecastingMethodForecastedValuesForLag2()
    
    #reading predicted value
    forecastedValuesWithoutLag = readAutoRegressionForecastingMethodForecastedValuesWithoutLag()
    
    #mean squared error for lag 1
    meanSquaredErrorForLag1 = mean_squared_error(X_test["PopEst"], forecastedValuesForLag1)
    
    #mean squared error for lag 2
    meanSquaredErrorForLag2 = mean_squared_error(X_test["PopEst"], forecastedValuesForLag2)
    
    #mean squared error for without lag
    meanSquaredErrorForWithoutLag = mean_squared_error(X_test["PopEst"], forecastedValuesWithoutLag)
    
    print(meanSquaredErrorForLag1) #1143.4649378653387
    
    print(meanSquaredErrorForLag2) #30.24228895401259
    
    print(meanSquaredErrorForWithoutLag) #33.815158403670665
    
    
    
if __name__ == "__main__":
    testAutoRegressionForecastingMethodMetrics()