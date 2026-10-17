import cv2
import numpy as np
import os
import datetime

print(datetime.datetime.now())
directory = "dollar_train"
clean_images = os.listdir(directory)
clean_images = [f"{directory}/{image}" for image in clean_images]

crumpled_image = cv2.imread("test10.jpg")
crumpled_image_gray = cv2.cvtColor(crumpled_image, cv2.COLOR_RGB2GRAY)

def identify_banknote(clean_images, crumpled_image):
    # Load the crumpled image
    if crumpled_image is None:
        raise ValueError("Could not load crumpled image.")
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and descriptors in the crumpled image
    kp2, des2 = sift.detectAndCompute(crumpled_image, None)
    
    # Initialize the BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    max_inliers = 0
    identified_image = None

    for clean_image_path in clean_images:
        # Load and process each clean image
        clean_image = cv2.imread(clean_image_path, cv2.IMREAD_GRAYSCALE)
        if clean_image is None:
            print(f"Could not load {clean_image_path}. Skipping.")
            continue
        
        # Detect keypoints and descriptors in the clean image
        kp1, des1 = sift.detectAndCompute(clean_image, None)
        
        # Match features
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        if len(good_matches) > 4:
            # Extract locations of good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Estimate homography using RANSAC
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = np.sum(mask)
            
            if inliers > max_inliers:
                max_inliers = inliers
                identified_image = clean_image_path
    
    return identified_image

# Example usage:

result = identify_banknote(clean_images, crumpled_image_gray)

if result:
    print(f"The crumpled banknote matches with: {result}")
    print(datetime.datetime.now())
    cv2.imshow("original banknote", cv2.resize(cv2.imread(result), (500, 200)))
    cv2.imshow("crumpled banknote", cv2.resize(crumpled_image, (500, 200)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No matching banknote found.")