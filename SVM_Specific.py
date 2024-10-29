import pandas as pd
from cuml.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

# Load the Excel file into a Pandas DataFrame
df = pd.read_excel('/mnt/c/Users/Hong/Downloads/MediaPipe_Testing/MediaPipePoseEstimation/Analysis_of_Excel_Files/LEFT_ANKLE_data.xlsx_compiled.xlsx')

# Prepare the features (X) and the target (y)
X = df.iloc[:, 3:159].values
y = df['group'].values
column_labels = df.columns[3:159]
print(column_labels)

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

# Instantiate the SVC with the specified parameters
svc = SVC(C=376.45996941054483, gamma=0.15292674255804906, kernel='poly', random_state=42)

start_time = time.time()  # Start the timer

# Fit the model with the training data
svc.fit(X_train, y_train)

# Predict using the fitted model
svc_predictions = svc.predict(X_test)

end_time = time.time()  # End the timer

# Evaluate the model
print(classification_report(y_test, svc_predictions))

print(f"The processing time of the SVM is {end_time - start_time} seconds.")  # Print the processing time
