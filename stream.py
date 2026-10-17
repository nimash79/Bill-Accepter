import cv2
import time
from config import IP

rtsp_url = f"rtsp://{IP}:8080/h264.sdp"

def open_rtsp(url):
    while True:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print("Connected")
            return cap
        print("Reconnecting...")
        time.sleep(2)

cap = open_rtsp(rtsp_url)

cv2.namedWindow("RTSP Stream", cv2.WINDOW_NORMAL)

# اندازه بزرگ‌تر (مثلاً 960x720)
cv2.resizeWindow("RTSP Stream", 800, 400)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lost connection, reconnecting...")
        cap.release()
        cap = open_rtsp(rtsp_url)
        continue

    # ✅ چرخش ۱۸۰ درجه
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    cv2.imshow("RTSP Stream", frame)

    key = cv2.waitKey(1) & 0xFF

    # ✅ گرفتن عکس با Space
    if key == 32:
        filename = f"capture_{frame_id}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        frame_id += 1

    # خروج با q
    if key == ord('q'):
        break

    # خروج با ضربدر
    if cv2.getWindowProperty("RTSP Stream", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
