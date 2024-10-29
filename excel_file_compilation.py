import pandas as pd
import tsfel
import os

# Directories containing the folders with Excel files
directories = {
    "Normal_Excel_Files": r"C:\Users\Hong\Downloads\MediaPipe_Testing\MediaPipePoseEstimation\Normal_1_Excel_Files",
    "PD_ML_Excel_Files": r"C:\Users\Hong\Downloads\MediaPipe_Testing\MediaPipePoseEstimation\PD_ML_1_Excel_Files",
    "PD_MD_Excel_Files": r"C:\Users\Hong\Downloads\MediaPipe_Testing\MediaPipePoseEstimation\PD_MD_1_Excel_Files"
}

# Directory to save the compiled analysis Excel files
analysis_save_dir = r"C:\Users\Hong\Downloads\MediaPipe_Testing\MediaPipePoseEstimation\Analysis_of_Excel_Files_1"
os.makedirs(analysis_save_dir, exist_ok=True)

# Dictionary to store the features for each Excel file across directories
file_features = {}

# Set up the default configuration using statistical, temporal, and spectral feature sets
cfg = tsfel.get_features_by_domain()

for dir_name, dir_path in directories.items():
    for folder in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".xlsx"):
                    file_path = os.path.join(folder_path, file_name)
                    df = pd.read_excel(file_path)
                    y_series = df['Y']
                    y_df = pd.DataFrame(y_series)
                    X = tsfel.time_series_features_extractor(cfg, y_df)
                    
                    group = 0 if dir_name == "Normal_Excel_Files" else 1 if dir_name == "PD_ML_Excel_Files" else 2
                    
                    feature_row = {"folder_name": folder, "file_name": file_name, "group": group}
                    feature_row.update(X.mean(axis=0).to_dict())
                    
                    if file_name not in file_features:
                        file_features[file_name] = []
                    
                    file_features[file_name].append(feature_row)

# Create a DataFrame for each Excel file name and save it to an Excel file
for file_name, features in file_features.items():
    df = pd.DataFrame(features)
    
    # Save the DataFrame to an Excel file with the name of the Excel file + compiled
    compiled_file_path = os.path.join(analysis_save_dir, f"{file_name}_compiled_Y.xlsx")
    df.to_excel(compiled_file_path, index=False)

print("Compiled analysis Excel files have been saved successfully.")
