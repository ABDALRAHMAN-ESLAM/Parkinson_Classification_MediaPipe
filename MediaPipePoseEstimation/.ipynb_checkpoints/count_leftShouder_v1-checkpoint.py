import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(r"C:\Users\Hong\Downloads\MediaPipe_Testing\MediaPipePoseEstimation\input\024_NM_02.MOV")

#def calculate_angle(a, b, c):
    #a = np.array(a)  # First
    #b = np.array(b)  # Mid
    #c = np.array(c)  # End
    
    #radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    #angle = np.abs(radians * 180.0 / np.pi)
    
    #if angle > 180.0:
        #angle = 360 - angle
    #return angle 

# Curl counter variables
counter = 0 
stage = None

# Resize parameters
resize_width = 640  # Width of the resized frame
resize_height = 480  # Height of the resized frame

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
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
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            print(f'Shoulder landmark: {shoulder}')
            print(f'Elbow landmark: {elbow}')
            print(f'Wrist landmark: {wrist}')
            print(f'Hip landmark: {hip}')
 
            # Calculate angle # hip
            #target = shoulder
            #angle = calculate_angle(hip, target, elbow)

            # Curl counter logic
            #if angle > 70:
                #stage = "up"
            #if angle < 30 and stage == 'up':
                #stage = "down"
                #counter += 1
                #print(counter)

            # Visualize angle
            #if angle > 90:
                #color = (0, 0, 255)  # Red for large angles
            #else:
                #color = (120, 120, 120)  # Grey for smaller angles
            #cv2.putText(image, str(angle), 
                           #tuple(np.multiply(target, [640, 480]).astype(int)), 
                           #cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
        except:
            pass
        
        # Render curl counter
        #cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        
        # Rep data
        #cv2.putText(image, 'REPS', (15, 12), 
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        #cv2.putText(image, str(counter), 
                    #(10, 60), 
                    #cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Stage data
        #cv2.putText(image, 'STAGE', (120, 12), 
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        #cv2.putText(image, stage, 
                    #(120, 60), 
                    #cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
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
    cv2.destroyAllWindows()
