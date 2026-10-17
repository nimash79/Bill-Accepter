import cv2
import numpy as np


def extract_roi(img, roi):
    x1, x2, y1, y2 = roi
    return img[y1:y2, x1:x2]


def avg_hue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 0])


def check_color_shift(img1, img2, threshold=15):
    roi_100 = (1330, 1800, 600, 750)
    roi_bell = (1030, 1330, 500, 710)

    if img1 is None or img2 is None:
        raise ValueError("image not loaded")

    roi100_img1 = extract_roi(img1, roi_100)
    roi100_img2 = extract_roi(img2, roi_100)

    bell_img1 = extract_roi(img1, roi_bell)
    bell_img2 = extract_roi(img2, roi_bell)

    h100_1 = avg_hue(roi100_img1)
    h100_2 = avg_hue(roi100_img2)

    hbell_1 = avg_hue(bell_img1)
    hbell_2 = avg_hue(bell_img2)

    shift100 = abs(h100_2 - h100_1)
    shiftbell = abs(hbell_2 - hbell_1)

    print("shift 100:", shift100)
    print("shift bell:", shiftbell)

    if shift100 > threshold or shiftbell > threshold:
        return True
    else:
        return False


def visualize_rois(img, scale=0.6):
    roi_100 = (1310, 1790, 600, 750)
    roi_bell = (1010, 1310, 500, 710)
    vis = img.copy()

    # رسم محدوده‌ها
    x1,x2,y1,y2 = roi_100
    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),3)
    cv2.putText(vis,"100",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    x1,x2,y1,y2 = roi_bell
    cv2.rectangle(vis,(x1,y1),(x2,y2),(255,0,0),3)
    cv2.putText(vis,"Bell",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    # کوچک‌سازی تصویر قبل از نمایش
    h, w = vis.shape[:2]
    vis_resized = cv2.resize(vis, (int(w * scale), int(h * scale)))

    cv2.imshow("ROI Check", vis_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

roi_100 = (1310, 1790, 600, 750)
roi_bell = (1010, 1310, 500, 710)
img1 = cv2.imread("cropped1.jpg")
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("cropped2.jpg")
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
visualize_rois(img1)

result = check_color_shift(img1, img2)

print("REAL" if result else "FAKE")
