import cv2
import easyocr
import re
import numpy as np

SERIAL_ROIS = {
    "1": {
        "right": (1140, 1610, 180, 280),
        "left": ()
    },
    "5": {
        "right": (1070, 1560, 460, 570),
        "left": (40, 550, 120, 210),
    },
    "10": {
        "right": (1140, 1620, 450, 580),
        "left": (80, 590, 140, 220),
    },
    "20": {
        "right": (1160, 1610, 480, 580),
        "left": (60, 570, 150, 230),
    },
    "100": {
        "right": (1260, 1720, 500, 630),
        "left": (40, 600, 170, 290),
    }
}

reader = easyocr.Reader(
    ["en"],
    model_storage_directory="models"
)

def crop_serial_roi(img, bill_type, position="right"):
    h, w = img.shape[:2]
    x1, x2, y1, y2 = SERIAL_ROIS[bill_type][position]
    return img[y1:y2, x1:x2]

def optimize_serial_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("morph.jpg", img)

    results = reader.readtext(
        img, 
        allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        detail=0,
    )
    
    raw_text = "".join(results).replace(" ", "")
    print(f"Raw OCR Output: {raw_text}")

    pattern = r'[A-Z]{1,2}[0-9]{8}[A-Z]'
    match = re.search(pattern, raw_text)

    if match:
        return match.group(0)
    else:
        clean_str = re.sub(r'[^A-Z0-9]', '', raw_text)
        return f"Not Found (Partial: {clean_str})"

def clean_serial(text):

    text = text.replace(" ", "")
    text = text.upper()

    # text = re.sub(r'[^A-Z0-9]', '', text)

    match = re.search(r'[A-Z]{1,2}[0-9]{8}[A-Z]', text)

    if match:
        return match.group(0)

    return text

def read_serial(img, bill_type, position):
    roi = crop_serial_roi(img, str(bill_type), position)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 8
    )
    cv2.imwrite("threshold.jpg", th)
    result = reader.readtext(th, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', detail=0)
    # joined = "".join(result).replace(" ", "")
    # m = re.findall(r"[A-Z0-9]{10,12}", joined)
    # return m[0] if m else joined
    text = "".join(result)
    serial = clean_serial(text)
    return serial

def extract_serial(img, bill_type, position="right"):
    serial_roi = crop_serial_roi(img, str(bill_type), position)
    serial = optimize_serial_ocr(serial_roi)
    return serial

def is_back_side(img):
    for angle in [0, 180]:
        rotated = img if angle == 0 else cv2.rotate(img, cv2.ROTATE_180)
        results = reader.readtext(rotated, detail=0)
        for t in results:
            text = t.replace(" ", "")
            if "TRUST" in text:
                return True
    return False

