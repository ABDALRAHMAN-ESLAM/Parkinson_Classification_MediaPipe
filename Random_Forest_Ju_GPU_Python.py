import cudf
import pandas as pd
from cuml.ensemble import RandomForestClassifier as cuMLRandomForestClassifier
from cuml.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
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

X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X))
y_cudf = cudf.Series(y)

# Split the data into training and testing sets
X_train_cudf, X_test_cudf, y_train_cudf, y_test_cudf = train_test_split(X_cudf, y_cudf, test_size=0.2, random_state=42)

# Convert cuDF DataFrames to NumPy arrays for sklearn
X_train = X_train_cudf.to_numpy()
X_test = X_test_cudf.to_numpy()
y_train = y_train_cudf.to_numpy()
y_test = y_test_cudf.to_numpy()

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf = cuMLRandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42)

start_time = time.time()  # Start the timer

# Fit the random search model
rf_random.fit(X_train, y_train)

# You can get the best parameters like this
print(rf_random.best_params_)

# Make predictions on the test set using the best model
y_pred = rf_random.predict(X_test)

end_time = time.time()  # End the timer

# Calculate the processing time
processing_time = end_time - start_time

from sklearn.metrics import classification_report

# Evaluate the model
print(classification_report(y_test, y_pred))

# Print the processing time
print(f"The processing time of the Random Forest is {processing_time} seconds.")
