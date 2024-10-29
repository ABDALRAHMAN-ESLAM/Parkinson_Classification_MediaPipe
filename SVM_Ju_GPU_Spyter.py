import pandas as pd
from cuml.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from scipy.stats import uniform
import time
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

# Convert data to cuDF DataFrames
import cudf
X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X))
y_cudf = cudf.Series(y)

# Split the data into training and testing sets
X_train_cudf, X_test_cudf, y_train_cudf, y_test_cudf = train_test_split(X_cudf, y_cudf, test_size=0.2, random_state=42)

# Convert cuDF DataFrames to NumPy arrays for sklearn
X_train = X_train_cudf.to_numpy()
X_test = X_test_cudf.to_numpy()
y_train = y_train_cudf.to_numpy()
y_test = y_test_cudf.to_numpy()

# Define the parameter values that should be searched
param_dist = {'C': uniform(loc=0, scale=3000), 
              'gamma': uniform(loc=0, scale=1),
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'random_state': [None, 42]} 

# Instantiate the randomized search
random_search = RandomizedSearchCV(SVC(), param_distributions=param_dist, refit=True, verbose=3, n_iter=300)

start_time = time.time()  # Start the timer

# Fit the grid with data
random_search.fit(X_train, y_train)

# Print the best parameters
print(random_search.best_params_)

# Predict using the best parameters
random_search_predictions = random_search.predict(X_test)

end_time = time.time()  # End the timer

# Evaluate the model
print(classification_report(y_test, random_search_predictions))

print(f"The processing time of the SVM is {end_time - start_time} seconds.")  # Print the processing time