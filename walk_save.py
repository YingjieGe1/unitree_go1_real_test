import sys
sys.path.append('unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk
import cv2
import numpy as np
import time
import os

# ====== 控制参数 ======
ACTION_DURATION = 0.7  # 每个动作持续时间
CONTROL_FREQ = 20      # 控制频率 (Hz)
SAVE_DIR = "saved_frames_forward"

# ====== 初始化高层控制接口 ======
HIGHLEVEL = 0xee
udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)
cmd = sdk.HighCmd()
state = sdk.HighState()
udp.InitCmdData(cmd)

# 创建图像保存文件夹
os.makedirs(SAVE_DIR, exist_ok=True)

def send_forward_action(duration=0.7, freq=20):
    interval = 1.0 / freq
    steps = int(duration * freq)

    cmd.mode = 2
    cmd.gaitType = 0
    cmd.speedLevel = 0
    cmd.footRaiseHeight = 0.08
    cmd.bodyHeight = 0.0
    cmd.euler = [0, 0, 0]
    cmd.velocity = [0.2, 0]  # 直线前进
    cmd.yawSpeed = 0
    cmd.reserve = 0

    for _ in range(steps):
        udp.SetSend(cmd)
        udp.Send()
        time.sleep(interval)

def preprocess_and_save_frame(frame, frame_id, save_dir):
    rotated = cv2.rotate(frame, cv2.ROTATE_180)
    h = rotated.shape[0]
    start = int(h * 4 / 7)
    cropped = rotated[start:, :, :]
    img = cv2.resize(cropped, (84, 84))
    cv2.imwrite(f"{save_dir}/frame_{frame_id:05d}.jpg", img)
    return img

# ====== 主循环 ======
print("🎯 开始直线前进并保存图像 (ESC 退出)")
FRAME_PATH = "tmp/go1_frame.jpg"
frame_id = 0

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

        send_forward_action(duration=ACTION_DURATION, freq=CONTROL_FREQ)

        cv2.imshow("Processed Frame", processed_img)
        if cv2.waitKey(1) == 27:  # ESC键
            break

except KeyboardInterrupt:
    print("🔚 用户中断")

finally:
    # 停止机器人
    cmd.velocity = [0.0, 0.0]
    udp.Send()
    cv2.destroyAllWindows()
