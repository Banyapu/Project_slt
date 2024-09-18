import pandas as pd

# 1. โหลดไฟล์ CSV แต่ละท่า (ข้อมูลมีคอลัมน์ 'class' อยู่แล้ว)
supine_data = pd.read_csv('coords_dataset/coords_fullbody_supine.csv')
prone_data = pd.read_csv('coords_dataset/coords_fullbody_prone.csv')
right_data = pd.read_csv('coords_dataset/coords_fullbody_right.csv')

# 2. รวมข้อมูลทั้งหมดเข้าด้วยกัน
combined_data = pd.concat([supine_data, prone_data, right_data], axis=0)

# 3. บันทึกข้อมูลที่รวมแล้วลงในไฟล์ CSV ใหม่
combined_data.to_csv('combined_postures.csv', index=False)

print("ข้อมูลถูกบันทึกลงในไฟล์ combined_postures.csv เรียบร้อยแล้ว!")
