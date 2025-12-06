#-------------------------------------------------------------------------
# AUTHOR: Joshua Estrada
# FILENAME: naive_bayes.py
# SPECIFICATION: Naive Bayes
# FOR: CS 4440- Assignment #4
# TIME SPENT: around an hour and 15 minutes
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#reading the training data
#--> add your Python code here
import csv
import numpy as np

# load training data
X_training = []
Y_training_cont = []
with open('weather_training.csv', 'r', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        try:
            features = [float(row[i]) for i in range(1, 6)]
            temp = float(row[6])
        except Exception:
            continue
        X_training.append(features)
        Y_training_cont.append(temp)

X_training = np.array(X_training)
Y_training_cont = np.array(Y_training_cont)

#update the training class values according to the discretization (11 values only)
# map each continuous target value to the nearest value in 'classes'
def discretize_values(real_values, classes_list):
    discrete = []
    for v in real_values:
        closest = min(classes_list, key=lambda c: abs(v - c))
        discrete.append(closest)
    return np.array(discrete)

y_training = discretize_values(Y_training_cont, classes)

#reading the test data
#--> add your Python code here
X_test = []
Y_test_cont = []
with open('weather_test.csv', 'r', newline='') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        try:
            features = [float(row[i]) for i in range(1, 6)]
            temp = float(row[6])
        except Exception:
            continue
        X_test.append(features)
        Y_test_cont.append(temp)

X_test = np.array(X_test)
Y_test_cont = np.array(Y_test_cont)

#update the test class values according to the discretization (11 values only)
y_test = discretize_values(Y_test_cont, classes)

#loop over the hyperparameter value (s)
#--> add your Python code here

highest_accuracy = 0.0
best_s = None

for s in s_values:

    # fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(X_training, y_training)

    # make the naive_bayes prediction for each test sample and start computing its accuracy
    predictions = clf.predict(X_test)
    correct = 0
    for pred, real_cont in zip(predictions, Y_test_cont):
        # percent difference using absolute real value in denominator to avoid sign issues
        if abs(real_cont) > 1e-8:
            percent_diff = 100.0 * abs(pred - real_cont) / abs(real_cont)
        else:
            percent_diff = 100.0 * abs(pred - real_cont)

        if percent_diff <= 15.0:
            correct += 1

    accuracy = correct / len(Y_test_cont) if len(Y_test_cont) > 0 else 0.0

    # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        best_s = s
        # print in the format shown in the assignment image
        print(f"Highest Naïve Bayes accuracy so far: {highest_accuracy}")
        print(f"Parameter: s = {best_s}")



