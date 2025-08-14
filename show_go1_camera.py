import cv2
import time

while True:
    frame = cv2.imread("temp/go1_frame.jpg")
    if frame is None:
        time.sleep(0.05)  # 等待新帧
        continue
    cv2.imshow("Go1 Camera Frame", frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
