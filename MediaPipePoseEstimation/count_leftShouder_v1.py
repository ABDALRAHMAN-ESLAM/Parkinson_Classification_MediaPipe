import cv2
import mediapipe as mp
import pandas as pd
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Directory containing .MOV files
input_directory = r"C:\Users\Hong\Downloads\MediaPipe_Testing\MediaPipePoseEstimation\PD_MD_1"
output_base_directory = r"C:\Users\Hong\Downloads\MediaPipe_Testing\MediaPipePoseEstimation\PD_MD_1_Excel_Files"

# Create the output base directory if it doesn't exist
os.makedirs(output_base_directory, exist_ok=True)

# Get list of all .MOV files in the input directory
mov_files = [f for f in os.listdir(input_directory) if f.endswith('.MOV')]

# List to keep track of failed files
failed_files = []

for mov_file in mov_files:
    try:
        # Create a VideoCapture object for each .MOV file
        cap = cv2.VideoCapture(os.path.join(input_directory, mov_file))
        
        # Create a dictionary to hold DataFrames for each landmark
        landmark_data = {mp_pose.PoseLandmark(i).name: pd.DataFrame(columns=['Time Interval', 'X', 'Y', 'Z']) for i in range(33)}
        
        # Resize parameters
        resize_width = 640
        resize_height = 480
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
        frame_duration = 1 / fps  # Duration of each frame in seconds
        
        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            time_counter = 0.0  # Initialize time counter as a float
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Increment the time counter
                time_counter += frame_duration
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
              
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Collect data for each landmark
                    for i in range(33):
                        landmark_name = mp_pose.PoseLandmark(i).name
                        x = landmarks[i].x
                        y = landmarks[i].y
                        z = landmarks[i].z
                        
                        # Create a new DataFrame for the new row
                        new_row = pd.DataFrame({
                            'Time Interval': [time_counter],  # Use the continuously increasing time counter
                            'X': [x],
                            'Y': [y],
                            'Z': [z]
                        })

                        # Concatenate the new row to the existing DataFrame
                        landmark_data[landmark_name] = pd.concat([landmark_data[landmark_name], new_row], ignore_index=True)

                        # Print out time interval and landmark data
                        print(f'{landmark_name}: (X: {x:.9f}, Y: {y:.9f}, Z: {z:.9f})')
                        print(f'Time interval: {time_counter:.9f} seconds')
                
                except Exception as e:
                    print(f'Error processing landmarks: {e}')
                
                # Draw landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                
                # Resize frame
                image_resized = cv2.resize(image, (resize_width, resize_height))
                
                # Display resized frame
                cv2.imshow('Mediapipe Feed', image_resized)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        
        # Create a new folder for each .MOV file to store the Excel files
        output_directory = os.path.join(output_base_directory, os.path.splitext(mov_file)[0])
        os.makedirs(output_directory, exist_ok=True)
        
        # Save each landmark's DataFrame to its own Excel file in the specified directory
        for landmark_name, df in landmark_data.items():
            df.to_excel(os.path.join(output_directory, f'{landmark_name}_data.xlsx'), index=False)

        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f'Failed to process {mov_file}: {e}')
        failed_files.append(mov_file)

# Print the names of the failed processed .MOV files
if failed_files:
    print("Failed to process the following .MOV files:")
    for file in failed_files:
        print(file)
else:
    print("All .MOV files processed successfully.")
