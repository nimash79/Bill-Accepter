import cv2
import numpy as np
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

REFERENCE_IMAGE = "dataset/100.jpg"

# ROIs in the REFERENCE IMAGE
#
# Format:
#     (x1, x2, y1, y2)
#
REFERENCE_ROI_100 = (1330, 1800, 600, 750)
REFERENCE_ROI_BELL = (1030, 1330, 500, 710)


# ------------------------------------------------------------
# SIFT configuration
# ------------------------------------------------------------

SIFT_FEATURES = 5000

LOWE_RATIO = 0.70

RANSAC_THRESHOLD = 5.0

MIN_GOOD_MATCHES = 10

MIN_INLIERS = 8


# ------------------------------------------------------------
# ROI extraction size
# ------------------------------------------------------------

ROI_100_SIZE = (470, 150)

ROI_BELL_SIZE = (300, 210)


# ------------------------------------------------------------
# Authenticity thresholds
# ------------------------------------------------------------
#
# These are intentionally much more conservative than the
# previous mean-hue threshold.
#
# They should be calibrated using your real/fake samples.
# ------------------------------------------------------------

# Minimum percentage of pixels showing meaningful
# chromatic/color change.
MIN_COLOR_CHANGE_RATIO = 0.08

# Difference between normalized chromaticity.
MIN_CHROMATICITY_DIFF = 0.025

# Difference in hue for changed pixels.
MIN_HUE_DIFF = 5.0

# Difference in saturation.
MIN_SATURATION_DIFF = 8.0

# Histogram difference.
MIN_HISTOGRAM_DISTANCE = 0.08


# ------------------------------------------------------------
# Decision
# ------------------------------------------------------------

# Number of ROI tests that need to indicate a color change.
#
# Possible tests:
#
#   1. $100 chromaticity
#   2. $100 local color
#   3. Bell chromaticity
#   4. Bell local color
#   5. Histogram comparison
#
MIN_POSITIVE_TESTS = 2


# ============================================================
# Cached reference data
# ============================================================

_reference_image = None
_reference_keypoints = None
_reference_descriptors = None


# ============================================================
# Reference image
# ============================================================

def _load_reference():
    """
    Load the reference image and calculate SIFT descriptors.

    The descriptors are cached so that SIFT does not have to
    be calculated repeatedly for every captured image.
    """

    global _reference_image
    global _reference_keypoints
    global _reference_descriptors

    if _reference_descriptors is not None:
        return

    path = Path(REFERENCE_IMAGE)

    if not path.exists():
        raise FileNotFoundError(
            f"Reference image not found: {REFERENCE_IMAGE}"
        )

    _reference_image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if _reference_image is None:
        raise RuntimeError(
            f"Could not read reference image: {REFERENCE_IMAGE}"
        )

    gray = cv2.cvtColor(
        _reference_image,
        cv2.COLOR_BGR2GRAY
    )

    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES
    )

    (
        _reference_keypoints,
        _reference_descriptors
    ) = sift.detectAndCompute(
        gray,
        None
    )

    if _reference_descriptors is None:
        raise RuntimeError(
            "Could not calculate SIFT descriptors "
            "for reference image."
        )

    print(
        "[FAKE DETECTION] "
        f"Reference loaded: "
        f"{len(_reference_keypoints)} keypoints"
    )


# ============================================================
# ROI utilities
# ============================================================

def roi_to_points(roi):
    """
    Convert:

        (x1, x2, y1, y2)

    to four corner points.

    Order:

        top-left
        top-right
        bottom-right
        bottom-left
    """

    x1, x2, y1, y2 = roi

    return np.float32([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ])


def polygon_area(points):
    """
    Calculate polygon area.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    return abs(
        cv2.contourArea(
            points.reshape(-1, 1, 2)
        )
    )


def polygon_inside_image(
    polygon,
    image
):
    """
    Check whether the polygon overlaps the visible image.

    The whole ROI does NOT need to be visible.
    """

    h, w = image.shape[:2]

    polygon = np.asarray(
        polygon,
        dtype=np.float32
    )

    min_x = np.min(polygon[:, 0])
    max_x = np.max(polygon[:, 0])

    min_y = np.min(polygon[:, 1])
    max_y = np.max(polygon[:, 1])

    return not (
        max_x < 0 or
        max_y < 0 or
        min_x >= w or
        min_y >= h
    )


# ============================================================
# Homography
# ============================================================

def find_banknote_homography(image):
    """
    Find homography:

        reference image -> current image

    The entire banknote does not need to be visible.
    """

    if image is None:
        raise ValueError(
            "Input image is None."
        )

    _load_reference()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES
    )

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    if (
        descriptors is None
        or len(keypoints) < 4
    ):
        raise RuntimeError(
            "Could not find enough SIFT features "
            "in input image."
        )

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False
    )

    knn_matches = matcher.knnMatch(
        _reference_descriptors,
        descriptors,
        k=2
    )

    good_matches = []

    for pair in knn_matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:
            good_matches.append(m)

    print(
        "[FAKE DETECTION] "
        f"Good matches: {len(good_matches)}"
    )

    if len(good_matches) < MIN_GOOD_MATCHES:
        raise RuntimeError(
            "Could not find enough matching features "
            f"({len(good_matches)} < "
            f"{MIN_GOOD_MATCHES})."
        )

    src_points = np.float32([
        _reference_keypoints[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    dst_points = np.float32([
        keypoints[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        RANSAC_THRESHOLD
    )

    if H is None:
        raise RuntimeError(
            "Could not calculate banknote homography."
        )

    inlier_mask = mask.ravel().astype(bool)

    inliers = int(
        np.sum(inlier_mask)
    )

    print(
        "[FAKE DETECTION] "
        f"RANSAC inliers: "
        f"{inliers}/{len(good_matches)}"
    )

    if inliers < MIN_INLIERS:
        raise RuntimeError(
            "Homography is unreliable. "
            f"Only {inliers} inliers found."
        )

    return H, good_matches, inliers


# ============================================================
# Transform ROI
# ============================================================

def transform_roi(
    roi,
    H
):
    """
    Transform reference ROI into current-image coordinates.
    """

    points = roi_to_points(roi)

    transformed = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2),
        H
    )

    return transformed.reshape(4, 2)


# ============================================================
# Perspective ROI extraction
# ============================================================

def extract_transformed_roi(
    image,
    polygon,
    output_size
):
    """
    Extract a perspective-normalized ROI.
    """

    polygon = np.asarray(
        polygon,
        dtype=np.float32
    )

    if polygon.shape != (4, 2):
        raise ValueError(
            "Polygon must contain four points."
        )

    width, height = output_size

    destination = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])

    transform = cv2.getPerspectiveTransform(
        polygon,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped


# ============================================================
# Dynamic ROI detection
# ============================================================

def get_dynamic_rois(image):
    """
    Locate the banknote dynamically and extract:

        $100 region
        Bell region
    """

    if image is None:
        raise ValueError(
            "Input image is None."
        )

    H, matches, inliers = find_banknote_homography(
        image
    )

    roi100_polygon = transform_roi(
        REFERENCE_ROI_100,
        H
    )

    bell_polygon = transform_roi(
        REFERENCE_ROI_BELL,
        H
    )

    if not polygon_inside_image(
        roi100_polygon,
        image
    ):
        raise RuntimeError(
            "$100 ROI is outside the visible image."
        )

    if not polygon_inside_image(
        bell_polygon,
        image
    ):
        raise RuntimeError(
            "Bell ROI is outside the visible image."
        )

    if polygon_area(roi100_polygon) < 100:
        raise RuntimeError(
            "$100 ROI has invalid geometry."
        )

    if polygon_area(bell_polygon) < 100:
        raise RuntimeError(
            "Bell ROI has invalid geometry."
        )

    roi100 = extract_transformed_roi(
        image,
        roi100_polygon,
        ROI_100_SIZE
    )

    bell = extract_transformed_roi(
        image,
        bell_polygon,
        ROI_BELL_SIZE
    )

    return {
        "100": {
            "polygon": roi100_polygon,
            "image": roi100
        },

        "bell": {
            "polygon": bell_polygon,
            "image": bell
        },

        "homography": H,

        "matches": len(matches),

        "inliers": inliers
    }


# ============================================================
# Image preprocessing
# ============================================================

def normalize_color(image):
    """
    Normalize illumination while preserving chromatic
    information.

    We use LAB CLAHE on the L channel.
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    normalized = cv2.merge([
        l,
        a,
        b
    ])

    return cv2.cvtColor(
        normalized,
        cv2.COLOR_LAB2BGR
    )


# ============================================================
# Chromaticity
# ============================================================

def get_chromaticity(image):
    """
    Convert BGR into normalized chromaticity.

    This removes much of the influence of overall brightness.

        r = R / (R + G + B)
        g = G / (R + G + B)
        b = B / (R + G + B)
    """

    image = image.astype(
        np.float32
    )

    # OpenCV BGR
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    total = (
        r + g + b + 1e-6
    )

    chroma = np.stack([
        r / total,
        g / total,
        b / total
    ], axis=2)

    return chroma


# ============================================================
# Mean chromaticity difference
# ============================================================

def chromaticity_difference(
    image1,
    image2
):
    """
    Calculate mean chromaticity difference.
    """

    c1 = get_chromaticity(
        image1
    )

    c2 = get_chromaticity(
        image2
    )

    difference = np.linalg.norm(
        c1 - c2,
        axis=2
    )

    return float(
        np.mean(difference)
    )


# ============================================================
# HSV comparison
# ============================================================

def hsv_difference(
    image1,
    image2
):
    """
    Calculate hue and saturation differences.

    Hue is treated as circular.
    """

    hsv1 = cv2.cvtColor(
        image1,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    hsv2 = cv2.cvtColor(
        image2,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    h1 = hsv1[:, :, 0]
    h2 = hsv2[:, :, 0]

    # Circular hue distance
    hue_diff = np.abs(
        h1 - h2
    )

    hue_diff = np.minimum(
        hue_diff,
        180.0 - hue_diff
    )

    saturation_diff = np.abs(
        hsv1[:, :, 1]
        -
        hsv2[:, :, 1]
    )

    return (
        float(np.mean(hue_diff)),
        float(np.mean(saturation_diff))
    )


# ============================================================
# Local color change
# ============================================================

def local_color_change(
    image1,
    image2
):
    """
    Detect pixels whose chromaticity changes significantly.

    Returns:

        mean difference
        percentage of significantly changed pixels
    """

    c1 = get_chromaticity(
        image1
    )

    c2 = get_chromaticity(
        image2
    )

    difference = np.linalg.norm(
        c1 - c2,
        axis=2
    )

    mean_difference = float(
        np.mean(difference)
    )

    changed = (
        difference
        >
        MIN_CHROMATICITY_DIFF
    )

    change_ratio = float(
        np.mean(changed)
    )

    return (
        mean_difference,
        change_ratio
    )


# ============================================================
# Histogram comparison
# ============================================================

def calculate_hue_histogram(
    image
):
    """
    Calculate normalized hue histogram.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Ignore very dark / low saturation pixels.
    mask = cv2.inRange(
        hsv,
        np.array([0, 25, 20]),
        np.array([179, 255, 255])
    )

    histogram = cv2.calcHist(
        [hsv],
        [0],
        mask,
        [180],
        [0, 180]
    )

    histogram = cv2.normalize(
        histogram,
        histogram
    )

    return histogram


def hue_histogram_distance(
    image1,
    image2
):
    """
    Bhattacharyya distance between hue distributions.

    Higher means more different.
    """

    hist1 = calculate_hue_histogram(
        image1
    )

    hist2 = calculate_hue_histogram(
        image2
    )

    return float(
        cv2.compareHist(
            hist1,
            hist2,
            cv2.HISTCMP_BHATTACHARYYA
        )
    )


# ============================================================
# Spatial block comparison
# ============================================================

def block_color_analysis(
    image1,
    image2,
    rows=3,
    cols=4
):
    """
    Split the ROI into blocks.

    This is important because a security feature may occupy
    only a small portion of the ROI.

    Returns the maximum block chromaticity difference and
    average block difference.
    """

    h, w = image1.shape[:2]

    block_differences = []

    for row in range(rows):

        y1 = int(
            row * h / rows
        )

        y2 = int(
            (row + 1) * h / rows
        )

        for col in range(cols):

            x1 = int(
                col * w / cols
            )

            x2 = int(
                (col + 1) * w / cols
            )

            block1 = image1[
                y1:y2,
                x1:x2
            ]

            block2 = image2[
                y1:y2,
                x1:x2
            ]

            difference = chromaticity_difference(
                block1,
                block2
            )

            block_differences.append(
                difference
            )

    if not block_differences:
        return 0.0, 0.0

    return (
        float(np.max(block_differences)),
        float(np.mean(block_differences))
    )


# ============================================================
# Analyze one ROI
# ============================================================

def analyze_roi(
    image1,
    image2,
    name
):
    """
    Perform complete color-change analysis for one ROI.
    """

    if (
        image1 is None
        or image2 is None
    ):
        raise ValueError(
            f"{name} ROI is empty."
        )

    # --------------------------------------------------------
    # Resize if necessary
    # --------------------------------------------------------

    if image1.shape[:2] != image2.shape[:2]:

        image2 = cv2.resize(
            image2,
            (
                image1.shape[1],
                image1.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

    # --------------------------------------------------------
    # Normalize illumination
    # --------------------------------------------------------

    normalized1 = normalize_color(
        image1
    )

    normalized2 = normalize_color(
        image2
    )

    # --------------------------------------------------------
    # Chromaticity
    # --------------------------------------------------------

    chroma_diff = chromaticity_difference(
        normalized1,
        normalized2
    )

    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    hue_diff, saturation_diff = hsv_difference(
        normalized1,
        normalized2
    )

    # --------------------------------------------------------
    # Local change
    # --------------------------------------------------------

    local_diff, change_ratio = local_color_change(
        normalized1,
        normalized2
    )

    # --------------------------------------------------------
    # Hue histogram
    # --------------------------------------------------------

    histogram_distance = hue_histogram_distance(
        normalized1,
        normalized2
    )

    # --------------------------------------------------------
    # Block analysis
    # --------------------------------------------------------

    max_block_diff, mean_block_diff = (
        block_color_analysis(
            normalized1,
            normalized2
        )
    )

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    chromaticity_test = (
        chroma_diff
        >=
        MIN_CHROMATICITY_DIFF
    )

    local_change_test = (
        change_ratio
        >=
        MIN_COLOR_CHANGE_RATIO
    )

    hue_test = (
        hue_diff
        >=
        MIN_HUE_DIFF
    )

    saturation_test = (
        saturation_diff
        >=
        MIN_SATURATION_DIFF
    )

    histogram_test = (
        histogram_distance
        >=
        MIN_HISTOGRAM_DISTANCE
    )

    positive_tests = sum([
        chromaticity_test,
        local_change_test,
        hue_test,
        saturation_test,
        histogram_test
    ])

    # --------------------------------------------------------
    # Print diagnostics
    # --------------------------------------------------------

    print()
    print(
        f"[FAKE DETECTION] "
        f"--- {name} ROI ---"
    )

    print(
        f"[FAKE DETECTION] "
        f"Chromaticity difference: "
        f"{chroma_diff:.5f}"
    )

    print(
        f"[FAKE DETECTION] "
        f"Local color difference: "
        f"{local_diff:.5f}"
    )

    print(
        f"[FAKE DETECTION] "
        f"Changed pixels: "
        f"{change_ratio * 100:.2f}%"
    )

    print(
        f"[FAKE DETECTION] "
        f"Mean hue difference: "
        f"{hue_diff:.2f}"
    )

    print(
        f"[FAKE DETECTION] "
        f"Mean saturation difference: "
        f"{saturation_diff:.2f}"
    )

    print(
        f"[FAKE DETECTION] "
        f"Hue histogram distance: "
        f"{histogram_distance:.5f}"
    )

    print(
        f"[FAKE DETECTION] "
        f"Maximum block difference: "
        f"{max_block_diff:.5f}"
    )

    print(
        f"[FAKE DETECTION] "
        f"Mean block difference: "
        f"{mean_block_diff:.5f}"
    )

    print(
        f"[FAKE DETECTION] "
        f"Positive tests: "
        f"{positive_tests}/5"
    )

    return {
        "name": name,

        "chromaticity_difference":
            chroma_diff,

        "local_difference":
            local_diff,

        "change_ratio":
            change_ratio,

        "hue_difference":
            hue_diff,

        "saturation_difference":
            saturation_diff,

        "histogram_distance":
            histogram_distance,

        "max_block_difference":
            max_block_diff,

        "mean_block_difference":
            mean_block_diff,

        "chromaticity_test":
            chromaticity_test,

        "local_change_test":
            local_change_test,

        "hue_test":
            hue_test,

        "saturation_test":
            saturation_test,

        "histogram_test":
            histogram_test,

        "positive_tests":
            positive_tests
    }


# ============================================================
# Combined authenticity decision
# ============================================================

def check_color_shift(
    img1,
    img2,
    threshold=None
):
    """
    Determine whether a $100 banknote is REAL or FAKE.

    The old implementation only compared mean hue.

    This implementation combines several independent
    measurements.

    Returns:

        True  -> REAL
        False -> FAKE

    The optional `threshold` argument is retained for
    compatibility with your existing application, but is
    intentionally ignored.
    """

    if img1 is None:
        raise ValueError(
            "img1 is None."
        )

    if img2 is None:
        raise ValueError(
            "img2 is None."
        )

    print()
    print(
        "========================================"
    )

    print(
        "       FAKE DETECTION"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Image 1
    # --------------------------------------------------------

    print(
        "[FAKE DETECTION] "
        "Processing image 1..."
    )

    rois1 = get_dynamic_rois(
        img1
    )

    # --------------------------------------------------------
    # Image 2
    # --------------------------------------------------------

    print(
        "[FAKE DETECTION] "
        "Processing image 2..."
    )

    rois2 = get_dynamic_rois(
        img2
    )

    # --------------------------------------------------------
    # Analyze $100 region
    # --------------------------------------------------------

    result100 = analyze_roi(
        rois1["100"]["image"],
        rois2["100"]["image"],
        "$100"
    )

    # --------------------------------------------------------
    # Analyze Bell region
    # --------------------------------------------------------

    result_bell = analyze_roi(
        rois1["bell"]["image"],
        rois2["bell"]["image"],
        "Bell"
    )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    positive_tests = (
        result100["positive_tests"]
        +
        result_bell["positive_tests"]
    )

    total_tests = 10

    # We give slightly more importance to localized
    # chromaticity changes than generic histogram changes.
    #
    # The primary signal is:
    #
    #     local_change_test
    #     chromaticity_test
    #
    primary_signal_100 = (
        result100["local_change_test"]
        and
        result100["chromaticity_test"]
    )

    primary_signal_bell = (
        result_bell["local_change_test"]
        and
        result_bell["chromaticity_test"]
    )

    secondary_signal_100 = (
        result100["hue_test"]
        or
        result100["saturation_test"]
        or
        result100["histogram_test"]
    )

    secondary_signal_bell = (
        result_bell["hue_test"]
        or
        result_bell["saturation_test"]
        or
        result_bell["histogram_test"]
    )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------
    #
    # A strong local signal in either security feature is
    # sufficient.
    #
    # Otherwise require supporting evidence from both areas.
    # --------------------------------------------------------

    strong_signal = (
        primary_signal_100
        or
        primary_signal_bell
    )

    supporting_signal = (
        secondary_signal_100
        and
        secondary_signal_bell
    )

    is_real = (
        strong_signal
        or
        (
            positive_tests
            >=
            MIN_POSITIVE_TESTS
            and
            supporting_signal
        )
    )

    # --------------------------------------------------------
    # Print final diagnostics
    # --------------------------------------------------------

    print()
    print(
        "----------------------------------------"
    )

    print(
        "[FAKE DETECTION] "
        f"Total positive tests: "
        f"{positive_tests}/{total_tests}"
    )

    print(
        "[FAKE DETECTION] "
        f"$100 primary signal: "
        f"{primary_signal_100}"
    )

    print(
        "[FAKE DETECTION] "
        f"Bell primary signal: "
        f"{primary_signal_bell}"
    )

    print(
        "[FAKE DETECTION] "
        f"$100 secondary signal: "
        f"{secondary_signal_100}"
    )

    print(
        "[FAKE DETECTION] "
        f"Bell secondary signal: "
        f"{secondary_signal_bell}"
    )

    print(
        "----------------------------------------"
    )

    if is_real:

        print(
            "[FAKE DETECTION] "
            "RESULT: REAL"
        )

    else:

        print(
            "[FAKE DETECTION] "
            "RESULT: FAKE"
        )

    print(
        "========================================"
    )

    return is_real


# ============================================================
# Draw polygon
# ============================================================

def draw_polygon(
    image,
    polygon,
    color,
    label
):
    """
    Draw an ROI polygon.
    """

    polygon = np.asarray(
        polygon,
        dtype=np.int32
    )

    cv2.polylines(
        image,
        [
            polygon.reshape(-1, 1, 2)
        ],
        True,
        color,
        4
    )

    x = int(
        np.min(polygon[:, 0])
    )

    y = int(
        np.min(polygon[:, 1])
    )

    cv2.putText(
        image,
        label,
        (
            x,
            max(y - 10, 30)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3,
        cv2.LINE_AA
    )


# ============================================================
# Visualize dynamic ROIs
# ============================================================

def visualize_dynamic_rois(
    image,
    window_name="Dynamic ROI Detection",
    scale=0.5
):
    """
    Display dynamically detected ROIs.
    """

    result = get_dynamic_rois(image)

    vis = image.copy()

    draw_polygon(
        vis,
        result["100"]["polygon"],
        (0, 255, 0),
        "$100"
    )

    draw_polygon(
        vis,
        result["bell"]["polygon"],
        (255, 0, 0),
        "Bell"
    )

    cv2.putText(
        vis,
        f"Matches: {result['matches']}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        vis,
        f"Inliers: {result['inliers']}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    h, w = vis.shape[:2]

    display = cv2.resize(
        vis,
        (
            int(w * scale),
            int(h * scale)
        ),
        interpolation=cv2.INTER_AREA
    )

    # Create the window explicitly
    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        window_name,
        display
    )

    # Wait until user presses a key
    cv2.waitKey(0)

    # Safely destroy the window
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass

    return result


# ============================================================
# Visualize extracted ROIs
# ============================================================

def visualize_extracted_rois(
    image
):
    """
    Display the extracted $100 and Bell regions.
    """

    result = get_dynamic_rois(
        image
    )

    cv2.imshow(
        "$100 ROI",
        result["100"]["image"]
    )

    cv2.imshow(
        "Bell ROI",
        result["bell"]["image"]
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return result


# ============================================================
# Debug comparison
# ============================================================

def debug_compare(
    img1,
    img2
):
    """
    Comprehensive debugging visualization.
    """

    result1 = get_dynamic_rois(
        img1
    )

    result2 = get_dynamic_rois(
        img2
    )

    vis1 = img1.copy()
    vis2 = img2.copy()

    draw_polygon(
        vis1,
        result1["100"]["polygon"],
        (0, 255, 0),
        "$100"
    )

    draw_polygon(
        vis1,
        result1["bell"]["polygon"],
        (255, 0, 0),
        "Bell"
    )

    draw_polygon(
        vis2,
        result2["100"]["polygon"],
        (0, 255, 0),
        "$100"
    )

    draw_polygon(
        vis2,
        result2["bell"]["polygon"],
        (255, 0, 0),
        "Bell"
    )

    scale = 0.5

    display1 = cv2.resize(
        vis1,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    display2 = cv2.resize(
        vis2,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow(
        "Image 1 - ROIs",
        display1
    )

    cv2.imshow(
        "Image 2 - ROIs",
        display2
    )

    cv2.imshow(
        "Image 1 - $100",
        result1["100"]["image"]
    )

    cv2.imshow(
        "Image 2 - $100",
        result2["100"]["image"]
    )

    cv2.imshow(
        "Image 1 - Bell",
        result1["bell"]["image"]
    )

    cv2.imshow(
        "Image 2 - Bell",
        result2["bell"]["image"]
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
# Calibration helper
# ============================================================

def analyze_without_decision(
    img1,
    img2
):
    """
    Run all measurements without making a REAL/FAKE
    decision.

    This is useful for calibrating thresholds.

    Returns the complete measurements.
    """

    rois1 = get_dynamic_rois(
        img1
    )

    rois2 = get_dynamic_rois(
        img2
    )

    result100 = analyze_roi(
        rois1["100"]["image"],
        rois2["100"]["image"],
        "$100"
    )

    result_bell = analyze_roi(
        rois1["bell"]["image"],
        rois2["bell"]["image"],
        "Bell"
    )

    return {
        "100": result100,
        "bell": result_bell
    }


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    IMG1_PATH = "cropped1.jpg"
    IMG2_PATH = "cropped2.jpg"

    img1 = cv2.imread(
        IMG1_PATH,
        cv2.IMREAD_COLOR
    )

    img2 = cv2.imread(
        IMG2_PATH,
        cv2.IMREAD_COLOR
    )

    if img1 is None:
        raise FileNotFoundError(
            f"Could not read: {IMG1_PATH}"
        )

    if img2 is None:
        raise FileNotFoundError(
            f"Could not read: {IMG2_PATH}"
        )

    # --------------------------------------------------------
    # Verify dynamic localization
    # --------------------------------------------------------

    print()
    print(
        "Detecting ROIs in image 1..."
    )

    visualize_dynamic_rois(
        img1,
        "Image 1 - Dynamic ROIs"
    )

    print()
    print(
        "Detecting ROIs in image 2..."
    )

    visualize_dynamic_rois(
        img2,
        "Image 2 - Dynamic ROIs"
    )

    # --------------------------------------------------------
    # Run authenticity detection
    # --------------------------------------------------------

    result = check_color_shift(
        img1,
        img2
    )

    print()
    print(
        "=" * 50
    )

    if result:
        print(
            "RESULT: REAL"
        )
    else:
        print(
            "RESULT: FAKE"
        )

    print(
        "=" * 50
    )