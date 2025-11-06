# -------------------------------------------------------------------------
# AUTHOR: Joshua Estrada
# FILENAME: decision_tree.py
# SPECIFICATION: trains and tests a decision tree
# FOR: CS 4440 (Data Mining) - Assignment #3
# TIME SPENT: around an hour
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)
    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    print("Original Data:")
    print(data_training[:5])
    
    refund_encoded = np.array(df.iloc[:, 1].map({"Yes": 1, "No": 0}))
    print("\nEncoded Refund Column:")
    print(refund_encoded[:5])
    
    # remove K from every number and convert to float
    converted_income = np.array(df.iloc[:, -2].str[:-1].astype(float))
    print("\nConverted Income to Float:")
    print(converted_income[:5])

    # one hot encode marital columns
    single = (df.iloc[:, 2] == 'Single').astype(int).values
    divorced = (df.iloc[:, 2] == 'Divorced').astype(int).values
    married = (df.iloc[:, 2] == 'Married').astype(int).values

    # recombine features into X
    X = np.column_stack((refund_encoded, single, divorced, married, converted_income)).tolist()
    print(X[:5])

    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    class_encoded = df.iloc[:, -1].map({"Yes": 1, "No": 2}).astype(int)
    Y = class_encoded.values.tolist()
    print("\nEncoded Class Labels:")
    print(Y[:5])
    correct_predictions = 0
    #loop your training and test tasks 10 times here
    for i in range (10):

       #fitting the decision tree to the data by using Gini index and no max_depth
       clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
       clf = clf.fit(X, Y)

       #plotting the decision tree
       tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
    #    plt.show()

       #read the test data and add this data to data_test NumPy
       #--> add your Python code here
       df_test = pd.read_csv('cheat_test.csv', sep=',', header=0)
       
       refund_encoded_test = np.array(df_test.iloc[:, 1].map({"Yes": 1, "No": 0}))
       converted_income_test = np.array(df_test.iloc[:, -2].str[:-1].astype(float))

       single_test = (df_test.iloc[:, 2] == 'Single').astype(int).values
       divorced_test = (df_test.iloc[:, 2] == 'Divorced').astype(int).values
       married_test = (df_test.iloc[:, 2] == 'Married').astype(int).values
       
       data_test = np.column_stack((
                refund_encoded_test, 
                single_test, 
                divorced_test,
                married_test,
                converted_income_test,
            )).tolist()
       Y_test = df_test.iloc[:, -1].map({"Yes": 1, "No": 2}).astype(int)

       for i, data in enumerate(data_test):
           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
           class_predicted = clf.predict([data])[0]

           #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
           #--> add your Python code here
           actual_class = Y_test[i]
           if actual_class == class_predicted:
               correct_predictions += 1

       #find the average accuracy of this model during the 10 runs (training and test set)
       #--> add your Python code here
    print(f"Final Accuracy when training on {ds}: {correct_predictions/(Y_test.size*10)}")

    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    #--> add your Python code here



