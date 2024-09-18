import mediapipe as mp  # Import mediapipe
import cv2  # Import opencv
import csv
import numpy as np

# Initialize Mediapipe drawing and BlazePose models
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Replace 'video_file.mp4' with your video file path
cap = cv2.VideoCapture(r'/Users/banyaponpumipol/Desktop/project /dataset/fullbody video/prone.mov')

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a CSV file and write the header
with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    landmarks = ['class']
    num_coords = 33  # BlazePose landmarks have 33 points
    for val in range(1, num_coords + 1):
        landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
    csv_writer.writerow(landmarks)

# Buffer to store data before writing to CSV
buffer = []

# Initiate BlazePose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    class_name = "prone"  # Adjust this as needed
    processed_frames = 0  # To keep track of processed frames
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = pose.process(image)
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Pose Detections
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                     )
            # Extract Pose landmarks
            pose_landmarks = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks]).flatten())
            
            # Concatenate row with class name
            row = [class_name] + pose_row
            
            # Store row in buffer
            buffer.append(row)
            
            # Write to CSV in bulk (e.g., every 100 frames)
            if len(buffer) >= 100:
                with open('coords.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerows(buffer)
                buffer = []  # Clear buffer after writing
        
        # Update progress
        processed_frames += 1
        percent_complete = (processed_frames / total_frames) * 100
        print(f'Processing: {percent_complete:.2f}% complete', end='\r')
        
        # Optionally display the video feed (comment out if not needed)
        cv2.imshow('Video Feed', image)

        if cv2.waitKey(10) & 0xFF == 27:  # Press 'esc' to exit
            break

# Write remaining data in buffer to CSV
if buffer:
    with open('coords.csv', mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(buffer)

# Release resources
cap.release()
cv2.destroyAllWindows()

print("CSV export complete!")
