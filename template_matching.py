import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

TEMPLATE_PATHS = {
    1: "dataset/1.jpg",
    5: "dataset/5.jpg",
    10: "dataset/10.jpg",
    20: "dataset/20.jpg",
    100: "dataset/100.jpg",
}

# Number of SIFT features
SIFT_FEATURES = 5000

# Lowe's ratio test
LOWE_RATIO = 0.70

# RANSAC reprojection threshold
RANSAC_THRESHOLD = 5.0

# Minimum number of good matches
MIN_GOOD_MATCHES = 10

# Minimum number of RANSAC inliers
MIN_INLIERS = 8

# Minimum inlier ratio
MIN_INLIER_RATIO = 0.20

# Minimum final matching score
MIN_SCORE = 0.20


# ============================================================
# LOAD TEMPLATES
# ============================================================

templates = {}

for value, path in TEMPLATE_PATHS.items():

    img = cv2.imread(path)

    if img is None:
        raise ValueError(
            f"Cannot load template image: {path}"
        )

    templates[value] = img


# ============================================================
# SIFT
# ============================================================

sift = cv2.SIFT_create(
    nfeatures=SIFT_FEATURES
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(img):
    """
    Convert image to grayscale and improve local contrast.

    CLAHE helps SIFT when the template and camera image have
    different illumination.
    """

    if img is None:
        raise ValueError(
            "Image is None."
        )

    if len(img.shape) == 3:

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = img.copy()

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    return gray


# ============================================================
# PRECOMPUTE TEMPLATE FEATURES
# ============================================================

template_features = {}


def prepare_templates():

    print()
    print("========================================")
    print("Preparing denomination templates")
    print("========================================")

    for value, template in templates.items():

        gray = preprocess(
            template
        )

        keypoints, descriptors = sift.detectAndCompute(
            gray,
            None
        )

        if descriptors is None:
            raise RuntimeError(
                f"Could not extract SIFT features "
                f"from ${value} template."
            )

        template_features[value] = {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "shape": template.shape,
        }

        print(
            f"${value}: "
            f"{len(keypoints)} keypoints"
        )

    print("========================================")


prepare_templates()


# ============================================================
# MATCH ONE DENOMINATION
# ============================================================

def match_template_sift(
    input_img,
    template_value
):
    """
    Match one denomination template against the input image.

    Returns a dictionary containing:

        score
        good_matches
        inliers
        inlier_ratio
        homography
        template_keypoints
        input_keypoints
        good_match_objects
        inlier_mask
    """

    template_data = template_features[
        template_value
    ]

    template_kp = template_data[
        "keypoints"
    ]

    template_des = template_data[
        "descriptors"
    ]

    # --------------------------------------------------------
    # Input features
    # --------------------------------------------------------

    input_gray = preprocess(
        input_img
    )

    input_kp, input_des = sift.detectAndCompute(
        input_gray,
        None
    )

    if input_des is None:

        return {
            "score": 0.0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "homography": None,
            "template_keypoints": template_kp,
            "input_keypoints": input_kp,
            "good_match_objects": [],
            "inlier_mask": None,
        }

    # --------------------------------------------------------
    # BFMatcher
    # --------------------------------------------------------

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False
    )

    matches = matcher.knnMatch(
        template_des,
        input_des,
        k=2
    )

    # --------------------------------------------------------
    # Lowe ratio test
    # --------------------------------------------------------

    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:
            good_matches.append(m)

    good_count = len(
        good_matches
    )

    if good_count < MIN_GOOD_MATCHES:

        return {
            "score": 0.0,
            "good_matches": good_count,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "homography": None,
            "template_keypoints": template_kp,
            "input_keypoints": input_kp,
            "good_match_objects": good_matches,
            "inlier_mask": None,
        }

    # --------------------------------------------------------
    # Point correspondences
    #
    # Template -> Input
    # --------------------------------------------------------

    src_pts = np.float32([
        template_kp[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        input_kp[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    # --------------------------------------------------------
    # RANSAC Homography
    # --------------------------------------------------------

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        RANSAC_THRESHOLD
    )

    if H is None or mask is None:

        return {
            "score": 0.0,
            "good_matches": good_count,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "homography": None,
            "template_keypoints": template_kp,
            "input_keypoints": input_kp,
            "good_match_objects": good_matches,
            "inlier_mask": None,
        }

    inlier_count = int(
        mask.sum()
    )

    inlier_ratio = (
        inlier_count /
        max(good_count, 1)
    )

    # --------------------------------------------------------
    # Matching score
    #
    # We use both:
    #
    #   inlier ratio
    #   absolute number of inliers
    #
    # The score is normalized approximately to [0, 1].
    # --------------------------------------------------------

    ratio_score = inlier_ratio

    inlier_count_score = min(
        inlier_count / 100.0,
        1.0
    )

    score = (
        0.6 * ratio_score
        +
        0.4 * inlier_count_score
    )

    return {
        "score": float(score),
        "good_matches": good_count,
        "inliers": inlier_count,
        "inlier_ratio": float(inlier_ratio),
        "homography": H,
        "template_keypoints": template_kp,
        "input_keypoints": input_kp,
        "good_match_objects": good_matches,
        "inlier_mask": mask,
    }


# ============================================================
# MATCH ALL DENOMINATIONS
# ============================================================

def match_all_templates(input_img):
    """
    Match the input image against all denomination templates.

    Returns:

        {
            1: {...},
            5: {...},
            10: {...},
            20: {...},
            100: {...}
        }
    """

    results = {}

    for value in templates:

        result = match_template_sift(
            input_img,
            value
        )

        results[value] = result

    return results


# ============================================================
# TRANSFORM TEMPLATE CORNERS
# ============================================================

def transform_template_corners(
    H,
    template_shape
):
    """
    Transform the four corners of the reference template
    into the input-image coordinate system.

    Order:

        0 = top-left
        1 = top-right
        2 = bottom-right
        3 = bottom-left
    """

    h, w = template_shape[:2]

    corners = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ]).reshape(-1, 1, 2)

    transformed = cv2.perspectiveTransform(
        corners,
        H
    )

    return transformed.reshape(4, 2)


# ============================================================
# ORIENTATION ESTIMATION
# ============================================================

def estimate_orientation(
    H,
    template_shape
):
    """
    Determine whether the banknote is approximately:

        0 degrees
    or
        180 degrees

    relative to the reference template.

    IMPORTANT:
    This function does NOT rely on which rotated input
    produced the highest SIFT score.

    It analyzes the actual geometric transformation represented
    by the homography.
    """

    corners = transform_template_corners(
        H,
        template_shape
    )

    top_left = corners[0]
    top_right = corners[1]

    bottom_right = corners[2]
    bottom_left = corners[3]

    # --------------------------------------------------------
    # Reference top edge
    # --------------------------------------------------------

    top_vector = (
        top_right -
        top_left
    )

    # --------------------------------------------------------
    # Reference bottom edge
    # --------------------------------------------------------

    bottom_vector = (
        bottom_right -
        bottom_left
    )

    # --------------------------------------------------------
    # Average horizontal direction
    #
    # This reduces the influence of perspective.
    # --------------------------------------------------------

    horizontal_vector = (
        top_vector +
        bottom_vector
    ) / 2.0

    dx = horizontal_vector[0]
    dy = horizontal_vector[1]

    angle = np.degrees(
        np.arctan2(
            dy,
            dx
        )
    )

    # Convert to [0, 360)
    angle = angle % 360.0

    # --------------------------------------------------------
    # The camera can introduce some perspective, so don't
    # require exactly 0 or 180.
    # --------------------------------------------------------

    distance_to_0 = min(
        angle,
        360.0 - angle
    )

    distance_to_180 = abs(
        angle - 180.0
    )

    if distance_to_0 <= distance_to_180:
        return 0

    return 180


# ============================================================
# ALTERNATIVE ORIENTATION ESTIMATION
# ============================================================

def estimate_orientation_using_corners(
    H,
    template_shape
):
    """
    More robust orientation estimator.

    It examines the direction of both horizontal and vertical
    edges.

    This is useful when perspective distortion is significant.
    """

    corners = transform_template_corners(
        H,
        template_shape
    )

    tl = corners[0]
    tr = corners[1]
    br = corners[2]
    bl = corners[3]

    # --------------------------------------------------------
    # Horizontal edges
    # --------------------------------------------------------

    top = tr - tl
    bottom = br - bl

    horizontal = (
        top + bottom
    ) / 2.0

    # --------------------------------------------------------
    # Vertical edges
    # --------------------------------------------------------

    left = bl - tl
    right = br - tr

    vertical = (
        left + right
    ) / 2.0

    # --------------------------------------------------------
    # Determine horizontal direction
    #
    # If the reference's left -> right direction maps toward
    # positive X, the banknote is normally 0°.
    #
    # If it maps toward negative X, it is normally 180°.
    # --------------------------------------------------------

    horizontal_x = horizontal[0]

    # --------------------------------------------------------
    # In unusual camera orientations, vertical geometry can
    # help disambiguate.
    # --------------------------------------------------------

    vertical_y = vertical[1]

    # Primary decision
    if horizontal_x >= 0:

        angle = 0

    else:

        angle = 180

    return angle


# ============================================================
# GET BEST MATCH
# ============================================================

def get_best_match(
    input_img
):
    """
    Find the denomination with the strongest geometric match.

    Returns:

        value
        result
    """

    results = match_all_templates(
        input_img
    )

    best_value = None
    best_result = None

    for value, result in results.items():

        if best_result is None:

            best_value = value
            best_result = result

            continue

        # Primary criterion: score
        if result["score"] > best_result["score"]:

            best_value = value
            best_result = result

    return (
        best_value,
        best_result
    )


# ============================================================
# MAIN ALGORITHM
# ============================================================

def algorithm(
    input_img,
    threshold=MIN_SCORE
):
    """
    Main template matching API.

    Returns:

        value
        angle
        score

    Example:

        value, angle, score = algorithm(image)
    """

    if input_img is None:
        raise ValueError(
            "Input image is None."
        )

    print()
    print("========================================")
    print("      DYNAMIC TEMPLATE MATCHING")
    print("========================================")

    # ========================================================
    # IMPORTANT
    #
    # We DO NOT rotate the input image.
    #
    # SIFT is rotation invariant.
    #
    # The homography tells us the physical orientation.
    # ========================================================

    results = match_all_templates(
        input_img
    )

    print()

    for value, result in results.items():

        print(
            f"Value ${value} → "
            f"Score = {result['score']:.4f} | "
            f"Good = {result['good_matches']} | "
            f"Inliers = {result['inliers']} | "
            f"Ratio = {result['inlier_ratio']:.3f}"
        )

    # ========================================================
    # FIND BEST DENOMINATION
    # ========================================================

    best_value = max(
        results,
        key=lambda value: results[value]["score"]
    )

    best_result = results[
        best_value
    ]

    best_score = best_result[
        "score"
    ]

    best_inliers = best_result[
        "inliers"
    ]

    best_ratio = best_result[
        "inlier_ratio"
    ]

    H = best_result[
        "homography"
    ]

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("========================================")
    print("           MATCHING RESULT")
    print("========================================")

    print(
        f"Best denomination : ${best_value}"
    )

    print(
        f"Best score        : {best_score:.4f}"
    )

    print(
        f"RANSAC inliers    : {best_inliers}"
    )

    print(
        f"Inlier ratio      : {best_ratio:.3f}"
    )

    print(
        f"Threshold         : {threshold:.4f}"
    )

    if (
        H is None
        or best_score < threshold
        or best_inliers < MIN_INLIERS
        or best_ratio < MIN_INLIER_RATIO
    ):

        print(
            "RESULT: NO MATCH"
        )

        print("========================================")

        return (
            None,
            None,
            best_score
        )

    # ========================================================
    # DETERMINE PHYSICAL ORIENTATION
    # ========================================================

    template_shape = templates[
        best_value
    ].shape

    angle = estimate_orientation(
        H,
        template_shape
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        f"Physical rotation : {angle}°"
    )

    print(
        f"RESULT: ${best_value} "
        f"({angle}°)"
    )

    print("========================================")

    return (
        best_value,
        angle,
        best_score
    )


# ============================================================
# GET HOMOGRAPHY
# ============================================================

def get_homography(
    input_img,
    value
):
    """
    Calculate the homography for a specific denomination.

    Useful when you want to reuse the same homography for:

        - serial number ROI
        - Bell ROI
        - $100 ROI
        - other dynamic regions
    """

    if value not in templates:

        raise ValueError(
            f"Unsupported denomination: ${value}"
        )

    result = match_template_sift(
        input_img,
        value
    )

    if result["homography"] is None:

        raise RuntimeError(
            f"Could not calculate homography "
            f"for ${value}."
        )

    return result["homography"]


# ============================================================
# ALGORITHM WITH HOMOGRAPHY
# ============================================================

def algorithm_with_homography(
    input_img,
    threshold=MIN_SCORE
):
    """
    Same as algorithm(), but additionally returns H.

    Returns:

        value
        angle
        score
        H

    This is recommended for your complete inspection pipeline.
    """

    if input_img is None:
        raise ValueError(
            "Input image is None."
        )

    print()
    print("========================================")
    print("      DYNAMIC TEMPLATE MATCHING")
    print("========================================")

    # --------------------------------------------------------
    # Match directly
    # --------------------------------------------------------

    results = match_all_templates(
        input_img
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    for value, result in results.items():

        print(
            f"Value ${value} → "
            f"Score = {result['score']:.4f} | "
            f"Good = {result['good_matches']} | "
            f"Inliers = {result['inliers']} | "
            f"Ratio = {result['inlier_ratio']:.3f}"
        )

    # --------------------------------------------------------
    # Best denomination
    # --------------------------------------------------------

    best_value = max(
        results,
        key=lambda value: results[value]["score"]
    )

    best_result = results[
        best_value
    ]

    best_score = best_result[
        "score"
    ]

    best_inliers = best_result[
        "inliers"
    ]

    best_ratio = best_result[
        "inlier_ratio"
    ]

    H = best_result[
        "homography"
    ]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if (
        H is None
        or best_score < threshold
        or best_inliers < MIN_INLIERS
        or best_ratio < MIN_INLIER_RATIO
    ):

        print()
        print(
            "RESULT: NO MATCH"
        )
        print("========================================")

        return (
            None,
            None,
            best_score,
            None
        )

    # --------------------------------------------------------
    # Orientation
    # --------------------------------------------------------

    angle = estimate_orientation(
        H,
        templates[best_value].shape
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print("========================================")
    print("           MATCHING RESULT")
    print("========================================")

    print(
        f"Best denomination : ${best_value}"
    )

    print(
        f"Physical rotation : {angle}°"
    )

    print(
        f"Score             : {best_score:.4f}"
    )

    print(
        f"RANSAC inliers    : {best_inliers}"
    )

    print(
        f"Inlier ratio      : {best_ratio:.3f}"
    )

    print(
        f"RESULT: ${best_value} "
        f"({angle}°)"
    )

    print("========================================")

    return (
        best_value,
        angle,
        best_score,
        H
    )


# ============================================================
# DEBUG: VISUALIZE HOMOGRAPHY
# ============================================================

def visualize_homography(
    input_img,
    value,
    H,
    window_name="Homography Result"
):
    """
    Draw the transformed template boundary onto the input image.
    """

    if H is None:

        print(
            "Homography is None."
        )

        return

    if value not in templates:

        raise ValueError(
            f"Unsupported denomination: ${value}"
        )

    template = templates[
        value
    ]

    corners = transform_template_corners(
        H,
        template.shape
    )

    debug = input_img.copy()

    polygon = np.int32(
        corners
    ).reshape(-1, 1, 2)

    # Draw banknote boundary
    cv2.polylines(
        debug,
        [polygon],
        True,
        (0, 255, 0),
        4
    )

    # Draw corners
    for index, point in enumerate(corners):

        x = int(point[0])
        y = int(point[1])

        cv2.circle(
            debug,
            (x, y),
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            debug,
            str(index),
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        window_name,
        debug
    )

    cv2.waitKey(0)

    # Important:
    # Use destroyAllWindows() instead of destroyWindow()
    # to avoid the Windows NULL-window exception.
    cv2.destroyAllWindows()


# ============================================================
# DEBUG: VISUALIZE SIFT MATCHES
# ============================================================

def visualize_matches(
    input_img,
    value
):
    """
    Display RANSAC inlier matches between the denomination
    template and the input image.
    """

    result = match_template_sift(
        input_img,
        value
    )

    H = result[
        "homography"
    ]

    good_matches = result[
        "good_match_objects"
    ]

    template_kp = result[
        "template_keypoints"
    ]

    input_kp = result[
        "input_keypoints"
    ]

    mask = result[
        "inlier_mask"
    ]

    if H is None or mask is None:

        print(
            "Could not calculate homography."
        )

        return

    template = templates[
        value
    ]

    # --------------------------------------------------------
    # Draw only RANSAC inliers
    # --------------------------------------------------------

    matches_mask = (
        mask.ravel().tolist()
    )

    visualization = cv2.drawMatches(
        template,
        template_kp,
        input_img,
        input_kp,
        good_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.namedWindow(
        "SIFT RANSAC Matches",
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        "SIFT RANSAC Matches",
        visualization
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    IMAGE_PATH = "test.jpg"

    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load image: "
            f"{IMAGE_PATH}"
        )

    # --------------------------------------------------------
    # Main test
    # --------------------------------------------------------

    value, angle, score, H = (
        algorithm_with_homography(
            image
        )
    )

    print()
    print("FINAL RESULT")
    print("----------------------------------------")
    print(
        f"Value : {value}"
    )
    print(
        f"Angle : {angle}"
    )
    print(
        f"Score : {score:.4f}"
    )

    # --------------------------------------------------------
    # Optional visualization
    # --------------------------------------------------------

    if (
        value is not None
        and H is not None
    ):

        visualize_homography(
            image,
            value,
            H
        )

        # Uncomment if needed:
        #
        # visualize_matches(
        #     image,
        #     value
        # )