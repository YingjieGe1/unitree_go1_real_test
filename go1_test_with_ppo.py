import sys
sys.path.append('unitree_legged_sdk/lib/python/amd64')  # ⚡ 根据你的SDK路径调整
import robot_interface as sdk
import cv2
import numpy as np
from stable_baselines3 import PPO
import time
import os

# ====== 配置路径 ======
MODEL_PATH = "Master_project/final_model_0731"
FRAME_PATH = "tmp_1/go1_frame.jpg"

# ====== 控制参数 ======
FORWARD_SPEED = 0.2  # m/s
TURN_SPEED = 0.5     # rad/s
ACTION_DURATION = 0.7  # 每个动作持续时间
CONTROL_FREQ = 20      # 控制频率 (Hz)
step_count = 0
max_episode_steps = 90

# ====== 初始化高层控制接口 ======
HIGHLEVEL = 0xee
udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)
cmd = sdk.HighCmd()
state = sdk.HighState()
udp.InitCmdData(cmd)

# ====== 加载 PPO 模型 ======
model = PPO.load(MODEL_PATH)


def preprocess_frame(frame, frame_id=None, save_dir="saved_frames"):
    # 创建保存文件夹
    os.makedirs(save_dir, exist_ok=True)

    # 🔄 旋转 180°
    rotated = cv2.rotate(frame, cv2.ROTATE_180)

    # ✂️ 裁掉上半部分（保留下 1/2）
    h = rotated.shape[0]
    start = int(h * 3 / 5)
    cropped = rotated[start:, :, :]

    # 📝 预处理（送入模型）
    img = cv2.resize(cropped, (84, 84))

    # 💾 保存裁剪图像
    if frame_id is not None:
        filename = f"{save_dir}/frame_{frame_id:05d}.jpg"
        cv2.imwrite(filename, img)

    return img, cropped




def send_action(action, duration=0.7, freq=20):
    interval = 1.0 / freq
    steps = int(duration * freq)

    cmd.mode = 2           # Walk mode
    cmd.gaitType = 0       # 步态
    cmd.velocity = [0.0, 0.0]  # Reset velocity

    if action == 0:
        cmd.mode = 2  # 2: 连续行走模式
        cmd.gaitType = 0  # 1: 小跑 gait（可改为0慢走，2跳跃）
        cmd.speedLevel = 1  # 速度等级：0-2
        cmd.footRaiseHeight = 0.08  # 抬脚高度，0.05~0.15
        cmd.bodyHeight = 0.0  # 身体高度调整（0~0.2）
        cmd.euler = [0, 0, 0]  # 无身体姿态倾斜
        cmd.velocity = [0.3, 0]  # x=前进速度(-1~1)，y=横移速度
        cmd.yawSpeed = 0  # 无转向
        cmd.reserve = 0  # 右转
        # 前进
        print("forward")
    elif action == 1:
        cmd.mode = 2  # 2: 连续行走模式
        cmd.gaitType = 0  # 1: 小跑 gait（可改为0慢走，2跳跃）
        cmd.speedLevel = 1  # 速度等级：0-2
        cmd.footRaiseHeight = 0.08  # 抬脚高度，0.05~0.15
        cmd.bodyHeight = 0.0  # 身体高度调整（0~0.2）
        cmd.euler = [0, 0, 0]  # 无身体姿态倾斜
        cmd.velocity = [0, 0]  # x=前进速度(-1~1)，y=横移速度
        cmd.yawSpeed = 0.6  # 无转向
        cmd.reserve = 0  # 右转
        # 左转
        print("turn left")
    elif action == 2:
        cmd.mode = 2  # 2: 连续行走模式
        cmd.gaitType = 0  # 1: 小跑 gait（可改为0慢走，2跳跃）
        cmd.speedLevel = 1  # 速度等级：0-2
        cmd.footRaiseHeight = 0.08  # 抬脚高度，0.05~0.15
        cmd.bodyHeight = 0.0  # 身体高度调整（0~0.2）
        cmd.euler = [0, 0, 0]  # 无身体姿态倾斜
        cmd.velocity = [0, 0]  # x=前进速度(-1~1)，y=横移速度
        cmd.yawSpeed = -0.6  # 无转向
        cmd.reserve = 0
        # 右转
        print("turn right")

    for _ in range(steps):
        udp.SetSend(cmd)
        udp.Send()
        time.sleep(interval)




print("🎯 开始接收帧并控制 Go1 (按 ESC 退出)")
try:
    frame_id = 0
    while True:
        if not os.path.exists(FRAME_PATH):
            print("⚠️ 等待帧...")
            time.sleep(0.05)
            continue

        frame = cv2.imread(FRAME_PATH)
        if frame is None:
            continue

        obs, processed_frame = preprocess_frame(frame, frame_id = frame_id)
        frame_id = frame_id + 1
        action, _ = model.predict(obs, deterministic=True)
        step_count += 1

        send_action(action, duration=0.2, freq=20)

        cv2.imshow("Go1 Camera Stream (Rotated+Cropped)", obs)

        time.sleep(6.0)

        if cv2.waitKey(1) == 27:  # ESC退出
            break
        elif step_count == max_episode_steps:
            break

except KeyboardInterrupt:
    print("🔚 用户中断")
finally:
    # 发送停止指令
    cmd.velocity = [0.0, 0.0]
    udp.Send()

    cv2.destroyAllWindows()
