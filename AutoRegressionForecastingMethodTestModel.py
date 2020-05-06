# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:52:22 2020

@author: Santosh Sah
"""
from statsmodels.tsa.ar_model import AR, ARResults
from AutoRegressionForecastingMethodUtils import (readAutoRegressionForecastingMethodXTest, readAutoRegressionForecastingMethodModel, 
                                               saveAutoRegressionForecastingMethodPredictedValues, readAutoRegressionForecastingMethodXTrain,
                                               readAutoRegressionForecastingMethodPredictedValues, saveAutoRegressionForecastingMethodForecastedValuesForLag1,
                                               saveAutoRegressionForecastingMethodForecastedValuesForLag2, saveAutoRegressionForecastingMethodForecastedValuesWithoutLag,
                                               readAutoRegressionForecastingMethodForecastedValuesForLag1, readAutoRegressionForecastingMethodForecastedValuesForLag2,
                                               readAutoRegressionForecastingMethodForecastedValuesWithoutLag)
from AutoRegressionForecastingMethodVisualization import visualizeAutoRegressionForecastingMethodPredictedValues

"""
test the model on testing dataset
"""
def testAutoRegressionForecastingMethodModel():
    
    #reading testing data
    X_train = readAutoRegressionForecastingMethodXTrain()
    
    #reading testing set
    X_test = readAutoRegressionForecastingMethodXTest()
    
    start = len(X_train)
    
    end = len(X_train) + len(X_test) - 1
    
    #reading model from pickle file
    autoRegressionForecastingMethodModel = readAutoRegressionForecastingMethodModel()
    
    #fitting for lag 1
    autoRegressionForecastingMethodModelFitWithLag1 = autoRegressionForecastingMethodModel.fit(maxlag = 1, method = "mle")
    
    #predicting for lag 1
    predictedValuesForLag1 = autoRegressionForecastingMethodModelFitWithLag1.predict(start = start, end = end, dynamic = False).rename("AR(1) Prediction")
    
    #fitting for lag 2
    autoRegressionForecastingMethodModelFitWithLag2 = autoRegressionForecastingMethodModel.fit(maxlag = 2, method = "mle")
    
    #predicting for lag 2
    predictedValuesForLag2 = autoRegressionForecastingMethodModelFitWithLag2.predict(start = start, end = end, dynamic = False).rename("AR(2) Prediction")
    
    #fitting without any lag. AR will find suitable P for the model.
    autoRegressionForecastingMethodModelFitWithoutLag = autoRegressionForecastingMethodModel.fit(method = "mle")
    
    #predicting for lag 1
    predictedValuesForWithoutLag = autoRegressionForecastingMethodModelFitWithoutLag.predict(start = start, end = end, dynamic = False).rename("AR(11) Prediction")
    
    
    #saving the foreasted values for lag1
    saveAutoRegressionForecastingMethodForecastedValuesForLag1(predictedValuesForLag1)
    
    #saving the foreasted values for lag2
    saveAutoRegressionForecastingMethodForecastedValuesForLag2(predictedValuesForLag2)
    
    #saving the foreasted values without lag
    saveAutoRegressionForecastingMethodForecastedValuesWithoutLag(predictedValuesForWithoutLag)
    
    #finding the lag of the model which we have created without lag. AR will find the optimum P value.
    print(autoRegressionForecastingMethodModelFitWithoutLag.k_ar) #value of lag is 11

def plotAutoRegressionForecastingMethodPredictedValues():
    
    #reading testing set
    X_test = readAutoRegressionForecastingMethodXTest()
    
    #reading predicted value for lag1
    ForecastedValuesForLag1 = readAutoRegressionForecastingMethodForecastedValuesForLag1()
    
    #reading predicted value for lag2
    ForecastedValuesForLag2 = readAutoRegressionForecastingMethodForecastedValuesForLag2()
    
    #reading predicted value
    ForecastedValuesWithoutLag = readAutoRegressionForecastingMethodForecastedValuesWithoutLag()
    
    #visualizing the predicted values with training set and the testing set
    visualizeAutoRegressionForecastingMethodPredictedValues(X_test, ForecastedValuesForLag1, ForecastedValuesForLag2, ForecastedValuesWithoutLag)
    
if __name__ == "__main__":
    #testAutoRegressionForecastingMethodModel()
    plotAutoRegressionForecastingMethodPredictedValues()