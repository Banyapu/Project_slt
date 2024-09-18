import cv2
import numpy as np
import tensorflow as tf
from mediapipe import solutions
from sklearn.preprocessing import LabelEncoder

# โหลดโมเดลที่ฝึกสอนมาแล้ว
model = tf.keras.models.load_model('posture_model.keras')

# Initialize Mediapipe Pose
mp_pose = solutions.pose
mp_drawing = solutions.drawing_utils

# เปิดไฟล์วิดีโอ
cap = cv2.VideoCapture('dataset/withbolster video/lyingR.mov')

# สร้าง LabelEncoder เดิมที่ใช้ในการฝึกโมเดล
label_encoder = LabelEncoder()
labels = ['Supine', 'Prone', 'Right']
label_encoder.fit(labels)

# ตัวแปรสำหรับ Threshold-based Stability
stable_prediction = None  # ท่าทางที่เสถียร
stable_count = 0  # จำนวนครั้งที่ทำนายเหมือนกัน
threshold = 10  # ต้องทำนายเหมือนกัน 3 ครั้งติดกันถึงจะเปลี่ยนท่าทาง

# ใช้ Mediapipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # เปลี่ยนสีภาพเป็น RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # ตรวจจับ Landmark
        results = pose.process(image)

        # ตั้งค่าเริ่มต้นของ predicted_label เพื่อแสดงผลลัพธ์ในทุกเฟรม
        if stable_prediction is not None:
            predicted_label = label_encoder.inverse_transform([stable_prediction])[0]
        else:
            predicted_label = "Unknown"  # ถ้ายังไม่มีการทำนายตั้งแต่เริ่มต้น

        # ถ้าพบ Pose landmarks
        if results.pose_landmarks:
            # วาด landmarks บนภาพ
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # ดึงข้อมูล features (landmarks) จากเฟรมวิดีโอ
            pose_landmarks = results.pose_landmarks.landmark
            pose_row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks]).flatten()

            # Reshape ข้อมูลให้เข้ากับโมเดล
            pose_row = pose_row.reshape(1, -1, 1)  # 1 instance, 99 features (33*3), 1 timestep

            # ทำการทำนาย
            prediction = model.predict(pose_row)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # ตรวจสอบว่าผลทำนายเหมือนกับท่าทางก่อนหน้าหรือไม่
            if predicted_class == stable_prediction:
                stable_count += 1  # ถ้าทำนายเหมือนกัน ให้เพิ่มจำนวนครั้งที่ทำนายตรงกัน
            else:
                stable_prediction = predicted_class  # อัปเดตท่าทางใหม่
                stable_count = 1  # รีเซ็ตตัวนับ

            # ถ้าทำนายเหมือนกันครบ threshold แล้ว ให้แสดงผลท่าทางใหม่
            if stable_count >= threshold:
                predicted_label = label_encoder.inverse_transform([stable_prediction])[0]

        # แสดงผลลัพธ์บนวิดีโอ
        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # แสดงเฟรมวิดีโอที่มีการวาด landmarks และผลลัพธ์ทำนาย
        cv2.imshow('Pose Estimation', frame)

        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# ปิดวิดีโอ
cap.release()
cv2.destroyAllWindows()
