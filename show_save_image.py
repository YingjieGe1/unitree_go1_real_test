import sys
sys.path.append('unitree_legged_sdk/lib/python/amd64')
import cv2
import numpy as np
import time
import os

# ====== 图像保存参数 ======
SAVE_DIR = "saved_frames_forward"
FRAME_PATH = "tmp/go1_frame.jpg"
frame_id = 0

# 创建图像保存文件夹
os.makedirs(SAVE_DIR, exist_ok=True)

def preprocess_and_save_frame(frame, frame_id, save_dir):
    rotated = cv2.rotate(frame, cv2.ROTATE_180)
    h = rotated.shape[0]
    start = int(h * 4 / 7)
    cropped = rotated[start:, :, :]
    img = cv2.resize(cropped, (84, 84))
    cv2.imwrite(f"{save_dir}/frame_{frame_id:05d}.jpg", img)
    return img

# ====== 主循环 ======
print("🎯 开始保存图像帧 (ESC 退出)")

try:
    while True:
        if not os.path.exists(FRAME_PATH):
            print("⚠️ 等待图像帧...")
            time.sleep(0.05)
            continue

        frame = cv2.imread(FRAME_PATH)
        if frame is None:
            continue

        processed_img = preprocess_and_save_frame(frame, frame_id, SAVE_DIR)
        frame_id += 1

        cv2.imshow("Processed Frame", processed_img)
        if cv2.waitKey(1) == 27:  # ESC键
            break

except KeyboardInterrupt:
    print("🔚 用户中断")

finally:
    cv2.destroyAllWindows()
