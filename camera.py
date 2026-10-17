import cv2
from config import IP

rtsp_url = f"rtsp://{IP}:8080/h264.sdp"
# rtsp_url = f"http://{IP}:8080/video"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

for i in range(30):
    ret, frame = cap.read()

if ret:
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    cv2.imwrite("frame.jpg", frame)
    print("Saved frame.jpg")

cap.release()
