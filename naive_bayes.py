#-------------------------------------------------------------------------
# AUTHOR: Roberto Reyes
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #4
# TIME SPENT: 3 hrs
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pandas
import numpy as numpy

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#reading the training data
trainingData = pandas.read_csv('weather_training.csv')
xTraining = trainingData.iloc[:, 1:-1].values  
yTraining = trainingData.iloc[:, -1].values

#update the training class values according to the discretization (11 values only)
def discretize(temp):
    return min(classes, key=lambda x: abs(x - temp))

y_training_discrete = numpy.array([discretize(temp) for temp in yTraining])

#reading the test data
testData = pandas.read_csv('weather_test.csv')
xTest = testData.iloc[:, 1:-1].values
yTest = testData.iloc[:, -1].values

#update the test class values according to the discretization (11 values only)
y_test_discrete = numpy.array([discretize(temp) for temp in yTest])

highest_accuracy = 0
best_s = 0

#loop over the hyperparameter value (s)
#--> add your Python code here

for x in s_values :

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=x)
    clf = clf.fit(xTraining, y_training_discrete)

    correctPredictions = 0

    #make the naive_bayes prediction for each test sample and start computing its accuracy
    #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
    #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    for y in range(len(xTest)):
        prediction = clf.predict([xTest[y]])[0]
        realValue = yTest[y]

        if realValue != 0: 
            percentDiff = 100 * (abs(prediction - realValue) / abs(realValue))

        else:
            percentDiff = 0

        if percentDiff <= 15:
            correctPredictions += 1

    accuracy = correctPredictions / len(xTest)
    

    # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
    # with the KNN hyperparameters. Example: "Highest Naive Bayes accuracy so far: 0.32, Parameters: s=0.1

    if accuracy > highest_accuracy: 
        highest_accuracy = accuracy
        best_s = s_values
print("Highest Naive Bayes accuracy so far: " , highest_accuracy )
print("Parameter: s = " , best_s)




