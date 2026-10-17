import cv2
import numpy as np
import os
from skimage.exposure import equalize_adapthist
from collections import Counter
from scipy.spatial.distance import euclidean
import datetime

directory = "dataset"
clean_images = os.listdir(directory)
clean_images = [f"{directory}/{image}" for image in clean_images]

def extractFeatures(img):
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    histogram_features = hist.flatten()

    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hue_features = cv2.HuMoments(cv2.moments(image_gray)).flatten()
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(image_hsv)
    h_hist = cv2.calcHist([h], [0], None, [50], [0, 180])  # Hue ranges from 0 to 180 in OpenCV
    s_hist = cv2.calcHist([s], [0], None, [50], [0, 256])  # Saturation ranges from 0 to 256
    v_hist = cv2.calcHist([v], [0], None, [50], [0, 256])  # Value ranges from 0 to 256
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    hsv_features = np.concatenate([h_hist, s_hist, v_hist])
    
    image_clahe = equalize_adapthist(image_gray, clip_limit = 0.02)
    # Rescaling img2 from 0 to 255.
    image_clahe = image_clahe*255.0
    clahe_features = image_clahe.flatten()
    
    features = (np.hstack([histogram_features, clahe_features, hsv_features]))
    features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)  # Avoid division by zero
    return features

def divide_image(image):
    height, width, _ = image.shape
    part_height = height // 4
    part_width = width // 4
    parts = []
    for i in range(4):
        for j in range(4):
            x_start = j * part_width
            x_end = x_start + part_width
            y_start = i * part_height
            y_end = y_start + part_height
            part = image[y_start:y_end, x_start:x_end]
            parts.append(part)
    return parts

def display_result(input_image, result_image_path):
    cv2.imshow("input image", input_image)
    result_image = cv2.imread(result_image_path)
    result_image = cv2.resize(result_image, (500, 200))
    cv2.imshow("result image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def algorithm(input_img):
    dataset_features = []
    for image_path in clean_images:
        img = cv2.imread(image_path)
        parts = divide_image(img)
        features_parts = []
        for part in parts:
            features_part = extractFeatures(part)
            features_parts.append(features_part)
        features = features_parts
        dataset_features.append(features)
        
    parts = divide_image(input_img)

    input_parts = []
    for part in parts:
        features_part = extractFeatures(part)
        input_parts.append(features_part)

    min_distance = [float("inf") for i in range(16)]
    best_match = [None for i in range(16)]

    for i in range(len(dataset_features)):
        dataset_parts = dataset_features[i]
        
        distance_parts = [0 for i in range(16)]
        for j in range(len(dataset_parts)):
            dataset_part = dataset_parts[j]
            input_part = input_parts[j]
            distance_parts[j] = euclidean(input_part, dataset_part)
        for k in range(len(distance_parts)):
            if distance_parts[k] < min_distance[k]:
                min_distance[k] = distance_parts[k]
                best_match[k] = clean_images[i]


    counter = Counter(best_match)
    print(counter)
    max_count = max(counter.values())
    best_matches = [string for string, count in counter.items() if count == max_count]
    prev_best_matches = None
    while len(best_matches) > 1 or max_count < 4:
        if prev_best_matches == best_matches:
            break
        prev_best_matches = best_matches
        if max_count < 4:
            best_matches = [string for string, count in counter.items() if count != 1]
        dataset_features = []
        for image_path in best_matches:
            img = cv2.imread(image_path)
            parts = divide_image(img)
            features_parts = []
            for part in parts:
                features_part = extractFeatures(part)
                features_parts.append(features_part)
            features = features_parts
            dataset_features.append(features)

        min_distance = [float("inf") for i in range(16)]
        best_match = [None for i in range(16)]
        for i in range(len(dataset_features)):
            dataset_parts = dataset_features[i]
        
            distance_parts = [0 for i in range(16)]
            for j in range(len(dataset_parts)):
                dataset_part = dataset_parts[j]
                input_part = input_parts[j]
                distance_parts[j] = euclidean(input_part, dataset_part)
            for k in range(len(distance_parts)):
                if distance_parts[k] < min_distance[k]:
                    min_distance[k] = distance_parts[k]
                    best_match[k] = best_matches[i]
        counter = Counter(best_match)
        max_count = max(counter.values())

        best_matches = [string for string, count in counter.items() if count == max_count]

    return best_matches[0]