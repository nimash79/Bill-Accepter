import cv2
import numpy as np

# مسیر عکس‌ها
template_paths = {
    1: "dataset/1.jpg",
    5: "dataset/5.jpg",
    10: "dataset/10.jpg",
    20: "dataset/20.jpg",
    100: "dataset/100.jpg"
}


templates = {}
for value, path in template_paths.items():
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load template image: {path}")
    templates[value] = img


def match_score(input_img, template_img):
    res = cv2.matchTemplate(input_img, template_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val


def algorithm(input_img, threshold=0.35):
    results = []
    print("=== Matching Scores ===")
    for angle in [0, 180]:
        # چرخش ۱۸۰ درجه در صورت نیاز
        rotated = input_img if angle == 0 else cv2.rotate(input_img, cv2.ROTATE_180)

        print(f"\n--- Rotation: {angle} degrees ---")

        for value, tmpl in templates.items():
            score = match_score(rotated, tmpl)
            print(f"Value ${value} → Score = {score:.4f}")
            results.append((value, angle, score))


    # پیدا کردن بهترین score
    best_value, best_angle, best_score = max(results, key=lambda x: x[2])

    print("\n===============================")
    if best_score < threshold:
        print(f"NO MATCH FOUND (Best score {best_score:.4f} < threshold {threshold})")
        print("===============================")
        return None, None, best_score

    print(f"BEST MATCH → ${best_value}  (Rotation: {best_angle}°)")
    print(f"SCORE: {best_score:.4f}")
    print("===============================")

    return best_value, best_angle, best_score