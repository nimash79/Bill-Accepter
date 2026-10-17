import cv2

img = cv2.imread("dataset/100.jpg")

if img is None:
    raise ValueError("تصویر یافت نشد!")

h, w = img.shape[:2]

x1 = 0
x2 = 350

y1 = 0
y2 = 200

print((x1, x2, y1, y2))


cropped = img[y1:y2, x1:x2]

cv2.imwrite("cropped.jpg", cropped)

print("Done!")
