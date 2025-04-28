
#-------------------------------------------------------------------------
# AUTHOR: Roberto Reyes
# FILENAME: knn.py
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #4
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas
import numpy

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
trainDF = pandas.read_csv('weather_training.csv')
xTrain = trainDF.iloc[:, 1:-1].values
yTrain = trainDF.iloc[:, -1].values
#reading the test data
testDF = pandas.read_csv('weather_test.csv')
xTest = testDF.iloc[:, 1:-1].values
yTest = testDF.iloc[:, -1].values
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')

def discretize(temp):
    for i in range(len(classes)-1):
        if classes[i] <= temp < classes[i+1]:
            return classes[i]
        return classes[-1] if temp >= classes[-1] else classes[0]
    if temp >= classes[-1]:
        return classes[i]
    else:
        return classes[0]
    
yTrainDiscrete = numpy.array([discretize(temp) for temp in yTrain])
yTestDiscrete = numpy.array([discretize(temp) for temp in yTest])

highestAcc = 0
bestParams ={}



#loop over the hyperparameter values (k, p, and w) ok KNN
#--> add your Python code here
for x in k_values:
    for y in p_values:
        for z in w_values:

            #fitting the knn to the data
            clf = KNeighborsClassifier(n_neighbors=x, p=y, weights = z)
            clf = clf.fit(xTrain, yTrainDiscrete)

            correctPredict = 0

            #make the KNN prediction for each test sample and start computing its accuracy
            for(xTestSamp, yTestSampleReal) in zip (xTest, yTest):
                predictedClass = clf.predict([xTestSamp])[0]
                predictedValue = predictedClass + 3

                if yTestSampleReal != 0: 
                    percent_diff = 100 * (abs(predictedValue - yTestSampleReal)/abs(yTestSampleReal))
                else:
                    percent_diff = 0

                if percent_diff <= 15:
                    correctPredict +=1

            accuracy = correctPredict / len(yTest)

            if accuracy > highestAcc:
                highestAcc = accuracy
                bestParams = {'k': x, 'p': y, 'w': z}

print("Highest KNN accuracy so far: ", highestAcc)

print(f"Parameters: k = {bestParams[0]}, p = {bestParams[1]}, weight = {bestParams[2]}")


