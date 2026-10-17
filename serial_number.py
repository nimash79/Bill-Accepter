import cv2
import easyocr
import re
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

REFERENCE_IMAGE = "dataset/100.jpg"

SIFT_FEATURES = 5000
LOWE_RATIO = 0.70

RANSAC_THRESHOLD = 5.0
MIN_GOOD_MATCHES = 10
MIN_INLIERS = 8

# ------------------------------------------------------------
# IMPORTANT:
# These coordinates are in the REFERENCE IMAGE coordinate
# system, NOT the current camera image.
#
# Format:
# (x1, x2, y1, y2)
# ------------------------------------------------------------

SERIAL_ROIS = {
    "1": {
        "right": (1140, 1610, 180, 280),
        "left": None,
    },

    "5": {
        "right": (1070, 1560, 460, 570),
        "left": (40, 550, 120, 210),
    },

    "10": {
        "right": (1140, 1620, 450, 580),
        "left": (90, 600, 160, 240),
    },

    "20": {
        "right": (1160, 1610, 480, 580),
        "left": (60, 570, 150, 230),
    },

    "100": {
        "right": (1260, 1720, 500, 630),
        "left": (40, 600, 170, 290),
    },
}


# ============================================================
# OCR
# ============================================================

reader = easyocr.Reader(
    ["en"],
    model_storage_directory="models",
)


# ============================================================
# REFERENCE IMAGE / SIFT
# ============================================================

_reference_image = None
_reference_gray = None
_reference_keypoints = None
_reference_descriptors = None


def load_reference():
    """
    Load the reference banknote and calculate SIFT features.

    This is done only once and cached.
    """

    global \
        _reference_image, \
        _reference_gray, \
        _reference_keypoints, \
        _reference_descriptors

    if _reference_descriptors is not None:
        return

    reference = cv2.imread(REFERENCE_IMAGE)

    if reference is None:
        raise FileNotFoundError(
            f"Could not load reference image: {REFERENCE_IMAGE}"
        )

    _reference_image = reference
    _reference_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES
    )

    (
        _reference_keypoints,
        _reference_descriptors
    ) = sift.detectAndCompute(
        _reference_gray,
        None
    )

    if _reference_descriptors is None:
        raise RuntimeError(
            "Could not extract SIFT features from reference image."
        )

    print(
        f"[SERIAL] Reference loaded: "
        f"{len(_reference_keypoints)} keypoints"
    )


# ============================================================
# HOMOGRAPHY
# ============================================================

def find_homography(img):
    """
    Find the transformation from reference image coordinates
    to the current camera image coordinates.

    Returns:
        H
        good_matches
        inliers
    """

    load_reference()

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES
    )

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    if descriptors is None:
        raise RuntimeError(
            "Could not extract SIFT features from current image."
        )

    # --------------------------------------------------------
    # Match reference -> current image
    # --------------------------------------------------------

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    matches = matcher.knnMatch(
        _reference_descriptors,
        descriptors,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:
            good_matches.append(m)

    print(
        f"[SERIAL] Good matches: "
        f"{len(good_matches)}"
    )

    if len(good_matches) < MIN_GOOD_MATCHES:
        raise RuntimeError(
            f"Not enough good matches: "
            f"{len(good_matches)}"
        )

    # --------------------------------------------------------
    # Build point correspondences
    # --------------------------------------------------------

    src_pts = np.float32([
        _reference_keypoints[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        keypoints[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    # --------------------------------------------------------
    # RANSAC
    # --------------------------------------------------------

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        RANSAC_THRESHOLD
    )

    if H is None or mask is None:
        raise RuntimeError(
            "Could not calculate homography."
        )

    inlier_count = int(mask.sum())

    print(
        f"[SERIAL] RANSAC inliers: "
        f"{inlier_count}/{len(good_matches)}"
    )

    if inlier_count < MIN_INLIERS:
        raise RuntimeError(
            f"Not enough RANSAC inliers: "
            f"{inlier_count}"
        )

    return H, good_matches, inlier_count


# ============================================================
# ROI UTILITIES
# ============================================================

def roi_to_points(roi):
    """
    Convert:
        (x1, x2, y1, y2)

    into four corner points.
    """

    x1, x2, y1, y2 = roi

    return np.float32([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ]).reshape(-1, 1, 2)


def transform_roi(H, roi):
    """
    Transform a reference-image ROI into the current image.
    """

    points = roi_to_points(roi)

    transformed = cv2.perspectiveTransform(
        points,
        H
    )

    return transformed.reshape(4, 2)


def order_points(points):
    """
    Order four points as:

        top-left
        top-right
        bottom-right
        bottom-left
    """

    points = np.asarray(points, dtype=np.float32)

    result = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    s = points.sum(axis=1)

    result[0] = points[np.argmin(s)]
    result[2] = points[np.argmax(s)]

    diff = np.diff(
        points,
        axis=1
    )

    result[1] = points[np.argmin(diff)]
    result[3] = points[np.argmax(diff)]

    return result


def warp_roi(img, polygon):
    """
    Perspective-warp a transformed ROI into a rectangular image.
    """

    points = order_points(polygon)

    tl, tr, br, bl = points

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)

    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)

    width = int(
        max(
            width_top,
            width_bottom
        )
    )

    height = int(
        max(
            height_right,
            height_left
        )
    )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "Invalid transformed ROI size."
        )

    destination = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])

    M = cv2.getPerspectiveTransform(
        points,
        destination
    )

    warped = cv2.warpPerspective(
        img,
        M,
        (width, height)
    )

    return warped


# ============================================================
# DYNAMIC SERIAL ROI
# ============================================================

def get_dynamic_serial_roi(
    img,
    bill_type,
    position="right",
    H=None
):
    """
    Dynamically locate and extract the serial-number ROI.

    The coordinates in SERIAL_ROIS are reference-image
    coordinates.

    The returned image is perspective-normalized.
    """

    bill_type = str(bill_type)
    position = position.lower()

    if bill_type not in SERIAL_ROIS:
        raise ValueError(
            f"Unsupported bill type: {bill_type}"
        )

    if position not in SERIAL_ROIS[bill_type]:
        raise ValueError(
            f"Unsupported position: {position}"
        )

    roi = SERIAL_ROIS[bill_type][position]

    if roi is None or len(roi) != 4:
        raise ValueError(
            f"No serial ROI defined for "
            f"${bill_type} ({position})."
        )

    # --------------------------------------------------------
    # Calculate homography if caller didn't provide it
    # --------------------------------------------------------

    if H is None:
        H, _, _ = find_homography(img)

    # --------------------------------------------------------
    # Transform reference ROI -> current image
    # --------------------------------------------------------

    polygon = transform_roi(
        H,
        roi
    )

    print(
        f"[SERIAL] ${bill_type} {position} "
        f"dynamic polygon:"
    )

    for point in polygon:
        print(
            f"    ({point[0]:.1f}, "
            f"{point[1]:.1f})"
        )

    # --------------------------------------------------------
    # Perspective normalization
    # --------------------------------------------------------

    serial_roi = warp_roi(
        img,
        polygon
    )

    return serial_roi


# ============================================================
# OCR PREPROCESSING
# ============================================================

def prepare_ocr_images(img):
    """
    Generate several OCR-friendly versions of the ROI.

    Multiple versions make OCR more robust to illumination
    and printing differences.
    """

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Upscale
    scale = 2

    gray = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # Slight denoising
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        8
    )

    # Otsu threshold
    _, otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return [
        img,
        gray,
        adaptive,
        otsu,
    ]


# ============================================================
# SERIAL CLEANING
# ============================================================

def clean_serial(text):
    """
    Clean OCR output and try to find:

        AA12345678A

    or:

        A12345678A
    """

    text = text.upper()
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("-", "")

    # Remove everything except alphanumeric characters
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    # Expected format:
    #
    # 1-2 letters
    # 8 digits
    # 1 letter
    #
    pattern = r"[A-Z]{1,2}[0-9]{8}[A-Z]"

    match = re.search(
        pattern,
        text
    )

    if match:
        return match.group(0)

    return text


# ============================================================
# OCR
# ============================================================

def optimize_serial_ocr(img):
    """
    Run EasyOCR on several preprocessing variants and
    return the best detected serial.
    """

    images = prepare_ocr_images(img)

    candidates = []

    for index, image in enumerate(images):

        results = reader.readtext(
            image,
            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            detail=0,
            paragraph=False
        )

        raw_text = "".join(results)

        print(
            f"[SERIAL OCR] Version {index}: "
            f"{raw_text}"
        )

        cleaned = clean_serial(
            raw_text
        )

        # Exact serial found
        if re.fullmatch(
            r"[A-Z]{1,2}[0-9]{8}[A-Z]",
            cleaned
        ):
            print(
                f"[SERIAL OCR] Exact match: "
                f"{cleaned}"
            )

            return cleaned

        if cleaned:
            candidates.append(
                cleaned
            )

    if candidates:

        # Return the longest useful OCR result
        best = max(
            candidates,
            key=len
        )

        return f"Not Found (Partial: {best})"

    return "Not Found"


# ============================================================
# MAIN SERIAL EXTRACTION
# ============================================================

def extract_serial(
    img,
    bill_type,
    position,
    H,
):

    print()
    print("=" * 50)
    print(
        f"[SERIAL] Extracting serial "
        f"${bill_type} ({position})"
    )
    print("=" * 50)

    if img is None:
        raise ValueError(
            "Input image is None."
        )

    # --------------------------------------------------------
    # Calculate homography
    # --------------------------------------------------------

    # H, good_matches, inliers = find_homography(
    #     img
    # )

    # --------------------------------------------------------
    # Dynamically locate serial ROI
    # --------------------------------------------------------

    serial_roi = get_dynamic_serial_roi(
        img,
        bill_type,
        position,
        H=H
    )

    if serial_roi is None or serial_roi.size == 0:
        raise RuntimeError(
            "Dynamic serial ROI is empty."
        )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    serial = optimize_serial_ocr(
        serial_roi
    )

    print(
        f"[SERIAL] RESULT: {serial}"
    )

    return serial


# ============================================================
# BACK SIDE DETECTION
# ============================================================

def is_back_side(img):
    """
    Detect whether the banknote is the back side.

    This part does not require the dynamic serial ROI.
    """

    for angle in [0, 180]:

        if angle == 0:
            rotated = img
        else:
            rotated = cv2.rotate(
                img,
                cv2.ROTATE_180
            )

        results = reader.readtext(
            rotated,
            detail=0
        )

        for text in results:

            text = text.replace(
                " ",
                ""
            ).upper()

            if "TRUST" in text:
                return True

    return False


# ============================================================
# DEBUG VISUALIZATION
# ============================================================

def debug_serial_roi(
    img,
    bill_type,
    position="right"
):
    """
    Show where the dynamic serial ROI was located.

    This is intended for standalone debugging,
    NOT for the PyQt production UI.
    """

    H, _, _ = find_homography(img)

    roi = SERIAL_ROIS[
        str(bill_type)
    ][position]

    polygon = transform_roi(
        H,
        roi
    )

    debug = img.copy()

    polygon_int = np.int32(
        polygon
    ).reshape(-1, 1, 2)

    cv2.polylines(
        debug,
        [polygon_int],
        True,
        (0, 255, 0),
        3
    )

    # Draw corner points
    for i, point in enumerate(polygon):

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
            str(i),
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.namedWindow(
        "Dynamic Serial ROI",
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        "Dynamic Serial ROI",
        debug
    )

    cv2.waitKey(0)

    # Avoid destroyWindow() because it can throw
    # a NULL-window exception on Windows.
    cv2.destroyAllWindows()

    return polygon


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    image_path = "cropped1.jpg"

    img = cv2.imread(
        image_path
    )

    if img is None:
        raise FileNotFoundError(
            f"Could not load {image_path}"
        )

    # --------------------------------------------------------
    # Example: $100 right serial
    # --------------------------------------------------------

    serial = extract_serial(
        img,
        bill_type=10,
        position="right"
    )

    print()
    print(
        "FINAL SERIAL:",
        serial
    )

    # --------------------------------------------------------
    # Optional visualization
    # --------------------------------------------------------

    debug_serial_roi(
        img,
        bill_type=10,
        position="right"
    )