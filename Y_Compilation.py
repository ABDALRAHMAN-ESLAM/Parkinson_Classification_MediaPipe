import os
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Define the directories
directories = {
    "C:\\Users\\Hong\\Downloads\\MediaPipe_Testing\\MediaPipePoseEstimation\\PD_MD_1_Excel_Files": 2,
    "C:\\Users\\Hong\\Downloads\\MediaPipe_Testing\\MediaPipePoseEstimation\\PD_ML_1_Excel_Files": 1,
    "C:\\Users\\Hong\\Downloads\\MediaPipe_Testing\\MediaPipePoseEstimation\\Normal_1_Excel_Files": 0
}

# Initialize the arrays
Y_array = []
Group = []

# Iterate through each main directory
for dir_path, group_value in directories.items():
    for subdir in os.listdir(dir_path):
        subdir_path = os.path.join(dir_path, subdir)
        if os.path.isdir(subdir_path):
            dir_Y_values = []
            for root, _, files in os.walk(subdir_path):
                for file in files:
                    if file.endswith('.xlsx') or file.endswith('.xls'):
                        file_path = os.path.join(root, file)
                        df = pd.read_excel(file_path)
                        Y_values = df['Z'].tolist()
                        dir_Y_values.append(Y_values)
            if dir_Y_values:  # Only append if there are Excel files in the subdirectory
                Y_array.append(dir_Y_values)
                Group.append(group_value)

# Pad the sequences to ensure they all have the same length
max_length = max(len(seq) for dir_seq in Y_array for seq in dir_seq)
Y_array_padded = [pad_sequences(dir_seq, maxlen=max_length, padding='post', dtype='float32') for dir_seq in Y_array]

# Define the output directory
output_dir = "C:\\Users\\Hong\\Downloads\\MediaPipe_Testing\\MediaPipePoseEstimation\\Deep_Learning"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the Y_array and Group as pickle files in the specified directory
with open(os.path.join(output_dir, 'Z_array_2.pkl'), 'wb') as f:
    pickle.dump(Y_array_padded, f)

with open(os.path.join(output_dir, 'Group_2.pkl'), 'wb') as f:
    pickle.dump(Group, f)

print("Z_array and Group have been saved as pickle files in the specified directory.")
