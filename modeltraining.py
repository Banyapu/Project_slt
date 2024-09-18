import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# 1. โหลดไฟล์ CSV ที่รวมข้อมูลทุกท่า
combined_data = pd.read_csv('combined_postures.csv')

# 2. แยก features และ labels
X = combined_data.drop('class', axis=1)  # ลบคอลัมน์ 'class' สำหรับ features
y = combined_data['class']  # ใช้คอลัมน์ 'class' สำหรับ labels

# 3. แปลง labels เป็น one-hot encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

# 4. แบ่งข้อมูล train และ test
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# 5. Reshape ข้อมูล X ให้เหมาะกับการใช้ใน LSTM
X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# 6. สร้าง Callback สำหรับ Progress Percent
class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.params['epochs'] * 100
        print(f"Epoch {epoch + 1}/{self.params['epochs']} - Progress: {progress:.2f}%")

# 7. สร้างโมเดล LSTM ด้วยการใช้ Input layer
model = Sequential()
model.add(Input(shape=(X_train.shape[1], 1)))  # ใช้ Input layer แทน input_shape ใน LSTM
model.add(LSTM(128, return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

# 8. Compile โมเดล
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 9. ฝึกสอนโมเดลพร้อมกับ Callback
progress_callback = ProgressCallback()
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[progress_callback])

# 10. บันทึกโมเดลที่ฝึกสอนแล้ว
model.save('posture_model.keras')

print("โมเดลได้รับการฝึกสอนและบันทึกเรียบร้อยแล้ว!")
