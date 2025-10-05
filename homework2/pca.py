# -------------------------------------------------------------------------
# AUTHOR: Joshua Estrada
# FILENAME: pca.py
# SPECIFICATION: description of the program
# FOR: CS 4440 (Data Mining) - Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data
#--> add your Python code here
df = pd.read_csv('./heart_disease_dataset.csv')

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
df_features = pd.DataFrame(scaled_data, columns=df.columns)

#Get the number of features
#--> add your Python code here
num_features = df.shape[1]

results = []

# Run PCA using 9 features, removing one feature at each iteration
for i in range(num_features):
    removed_feature = df_features.columns[i]
    # Create a new dataset by dropping the i-th feature
    reduced_data = df_features.drop(columns=[removed_feature])

    # Run PCA on the reduced dataset (only PC1 needed)
    pca = PCA(n_components=1)
    pca.fit(reduced_data)

    #Store PC1 variance and the feature removed
    pc1_var = pca.explained_variance_ratio_[0]
    results.append((removed_feature, pc1_var))
    print(f"Removing {removed_feature}: PC1 variance = {pc1_var:.6f}")

# Find the maximum PC1 variance
best_feature, best_var = max(results, key=lambda x: x[1])

#Print results
print(f"Highest PC1 variance found: {best_var:.6f} when removing {best_feature}")





