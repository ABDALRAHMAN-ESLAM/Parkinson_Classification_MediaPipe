import pickle
import os

# Define the directory where the pickle files are saved
pickle_dir = "C:\\Users\\Hong\\Downloads\\MediaPipe_Testing\\MediaPipePoseEstimation\\Deep_Learning"

# Load the Y_array pickle file
with open(os.path.join(pickle_dir, 'X_array.pkl'), 'rb') as f:
    Y_array = pickle.load(f)

# Load the Group pickle file
with open(os.path.join(pickle_dir, 'Group.pkl'), 'rb') as f:
    Group = pickle.load(f)

# Print the shape of Y_array and the length of Group
if Y_array:
    num_dirs = len(Y_array)
    num_files = len(Y_array[0]) if Y_array[0] is not None else 0
    y_length = len(Y_array[0][0]) if Y_array[0] is not None and len(Y_array[0]) > 0 and Y_array[0][0] is not None else 0
    print("Shape of X_array:", (num_dirs, num_files, y_length))
else:
    print("X_array is empty")

print("Length of Group:", len(Group))

# Print the content of Y_array and Group
print(Y_array)
print(Group)
