import pandas as pd
import tsfel
import matplotlib.pyplot as plt
import os

# Directories containing the folders with Excel files
directories = {
    "Normal_Excel_Files": r"C:\Users\Hong\Downloads\MediaPipe_Testing\MediaPipePoseEstimation\Normal_Excel_Files",
    "PD_ML_Excel_Files": r"C:\Users\Hong\Downloads\MediaPipe_Testing\MediaPipePoseEstimation\PD_ML_Excel_Files",
    "PD_MD_Excel_Files": r"C:\Users\Hong\Downloads\MediaPipe_Testing\MediaPipePoseEstimation\PD_MD_Excel_Files"
}

# Base directory to save the figures
base_save_dir = r"C:\Users\Hong\Downloads\MediaPipe_Testing\MediaPipePoseEstimation\024_NM_02"

# Create the base directory if it doesn't exist
os.makedirs(base_save_dir, exist_ok=True)

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
                    
                    if file_name not in file_features:
                        file_features[file_name] = {"Normal_Excel_Files": None, "PD_ML_Excel_Files": None, "PD_MD_Excel_Files": None}
                    
                    file_features[file_name][dir_name] = X.mean(axis=0)  # Average features for each file

# Plotting the bar graph for each feature in each Excel file
for file_name, features in file_features.items():
    # Create a directory for each file name
    save_dir = os.path.join(base_save_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.DataFrame(features)
    
    for feature in df.index:
        plt.figure(figsize=(15, 10))
        bars = df.loc[feature].plot(kind='bar', color=['red', 'blue', 'green'])
        plt.xlabel('Categories')
        plt.ylabel('Average Value')
        title = f'Average Feature Values for {feature} in {file_name}'
        plt.title(title)
        
        # Set x-axis labels to Normal, PD_ML, and PD_MD
        bars.set_xticklabels(["Normal", "PD_ML", "PD_MD"])
        
        plt.grid(True)
        
        # Save the figure with the feature name attached to the file name
        save_path = os.path.join(save_dir, f"{file_name}_{feature}.png")
        plt.savefig(save_path)

        # Close the figure to free up memory
        plt.close()
        
        # Show the plot (optional)
        # plt.show()

print("Figures have been saved successfully.")
