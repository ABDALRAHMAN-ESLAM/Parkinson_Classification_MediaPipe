import pandas as pd
from cuml.neighbors import KNeighborsClassifier as cuKNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from cuml.preprocessing import StandardScaler
import time  # Import the time module
import numpy as np
import os

# Directory containing the Excel files
directory = '/mnt/c/Users/Hong/Downloads/MediaPipe_Testing/MediaPipePoseEstimation/Analysis_of_Excel_Files_1'

# Initialize an empty list to store the data from each file
data_list = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.xlsx'):
        # Load the Excel file into a Pandas DataFrame
        df = pd.read_excel(os.path.join(directory, filename))
        # Append the relevant columns to the data list
        data_list.append(df.iloc[:, 3:159].values)

# Combine all arrays into a single feature set
X = np.concatenate(data_list, axis=1)

# Assuming 'group' column is present in the first file for target variable
df0 = pd.read_excel(os.path.join(directory, os.listdir(directory)[0]))
y = df0['group'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit on the training data and transform both the training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

start_time = time.time()  # Start the timer

# Create and fit the KNN model
k = 1  # Choose the value of k
knn = cuKNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Define the hyperparameter search space
k_range = list(range(1, 32))  # Values between 1 and 31 for n_neighbors
weight_options = ['uniform', 'distance']  # Specify weight options
algorithm_options = ['ball_tree', 'kd_tree', 'brute', 'auto']  # Specify algorithm option
p_values = [1, 2, 3]  # Specify p values for Minkowski distance

# Create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range, weights=weight_options, algorithm=algorithm_options, p=p_values)

# Create a StratifiedKFold object
stratified_kfold = StratifiedKFold(n_splits=10)

# Use scikit-learn's GridSearchCV with cuML's KNeighborsClassifier
grid = GridSearchCV(cuKNeighborsClassifier(), param_grid, cv=stratified_kfold, scoring='accuracy')

# Fit the grid with data
grid.fit(X_train_scaled, y_train)

# Examine the best model
print("\nBest score: ", grid.best_score_)
print("Best params: ", grid.best_params_)

# Predict on the testing set using the best model
knn_best = cuKNeighborsClassifier(**grid.best_params_)
knn_best.fit(X_train_scaled, y_train)
y_pred = knn_best.predict(X_test_scaled)

end_time = time.time()  # End the timer

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

print(f"The processing time of the KNN is {end_time - start_time} seconds.")  # Print the processing time

