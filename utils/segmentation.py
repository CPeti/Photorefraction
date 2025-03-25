import numpy as np
import random
import cv2
import dlib
import math
# suppress warnings
import warnings
warnings.filterwarnings("ignore")

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("haarcascades/shape_predictor_68_face_landmarks.dat")


def detect_eyes(img, resample=0):
    faces = dlib.full_object_detections()
    detections = detector(img, resample)
    if len(detections) == 0:
        return None, None
    for det in detections:
        faces.append(sp(img, det))
    try:
        right_eye = [faces[0].part(i) for i in range(36, 42)]
        left_eye = [faces[0].part(i) for i in range(42, 48)]
    except:
        return None, None

    # find min and max x and y points of the eyes
    right_eye_x = [point.x for point in right_eye]
    right_eye_y = [point.y for point in right_eye]
    left_eye_x = [point.x for point in left_eye]
    left_eye_y = [point.y for point in left_eye]

    eps = 25
    right_eye_x_min = min(right_eye_x)
    right_eye_x_max = max(right_eye_x)
    right_eye_y_min = min(right_eye_y) - eps
    right_eye_y_max = max(right_eye_y) + eps
    left_eye_x_min = min(left_eye_x)
    left_eye_x_max = max(left_eye_x)
    left_eye_y_min = min(left_eye_y) - eps
    left_eye_y_max = max(left_eye_y) + eps

    right_eye_img = img[right_eye_y_min:right_eye_y_max, right_eye_x_min:right_eye_x_max]
    left_eye_img = img[left_eye_y_min:left_eye_y_max, left_eye_x_min:left_eye_x_max]

    return right_eye_img, left_eye_img

def calculate_circle_from_points(p1, p2, p3):
    """Calculate the circle passing through three points p1, p2, p3."""
    # Unpack points
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Calculate the circle center and radius using linear algebra
    A = np.array([[x1 - x2, y1 - y2], [x1 - x3, y1 - y3]])
    B = np.array([
        (x1 ** 2 - x2 ** 2 + y1 ** 2 - y2 ** 2) / 2,
        (x1 ** 2 - x3 ** 2 + y1 ** 2 - y3 ** 2) / 2
    ])

    try:
        # Solve for center (cx, cy)
        cx, cy = np.linalg.solve(A, B)
        radius = np.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
        return (cx, cy), radius
    except np.linalg.LinAlgError:
        # If points are collinear or close to it, return None
        return None, None

def ransac_circle(points, num_iterations=10000, threshold=1.00):
    """Fit a circle to the given points using RANSAC."""
    if len(points) < 3:
        return None
    best_circle = None
    best_inliers = 0

    for _ in range(num_iterations):
        # Randomly sample 3 points
        sample_points = random.sample(points, 3)
        center, radius = calculate_circle_from_points(*sample_points)

        if center is None:
            continue  # Skip if points are collinear

        # Calculate inliers
        inliers = 0
        for p in points:
            dist = np.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)
            if abs(dist - radius) <= threshold:
                inliers += 1

        # Update best circle if current one has more inliers
        if inliers > best_inliers:
            best_inliers = inliers
            best_circle = (center, radius)

    return best_circle

def get_circle(img):
    points = np.argwhere(img == 255)
    points = [tuple(point) for point in points]

    circle = ransac_circle(points, 100, 1.0)
    if circle:
        center, radius = circle
    else:
        return None, None

    center = (round(center[1]), round(center[0]))
    radius = round(radius)
    return center, radius

def find_left_activations(imgo):
    img = imgo.copy()
    output = []
    for i in range(img.shape[0]):
        row = img[i]
        first_nonzero = np.argmax(row)
        row[:first_nonzero] = 0
        row[first_nonzero:] = 0
        row[first_nonzero] = 255
        output.append(row)
    return remove_border(np.array(output))

def remove_border(img):
    img[:1] = 0
    img[-1:] = 0
    img[:, :1] = 0
    img[:, -1:] = 0
    return img

def get_cicle_points(input_img, threshold=60, blur='gaussian', channel=None, mask_center=None, mask_radius=None, edge_detection='normal', use_top=True, use_bottom=True, kernel='sobel'):
    if input_img is None:
        return None
    img = input_img.copy()
    if channel == 'single':
        pass
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur
    if blur == 'gaussian':
        img = cv2.GaussianBlur(img, (5, 5), 0)
    elif blur == 'median':
        img = cv2.medianBlur(img, 9)
    elif blur == 'bilateral':
        img = cv2.bilateralFilter(img, 9, 15, 15)
    # right sobel
    if kernel == 'sobel':
        kernel = np.array([
            [-1, 0, 1], 
            [-2, 0, 2], 
            [-1, 0, 1]
        ])
    elif kernel == 'prewitt':
        kernel = np.array([
            [-1, 0, 1], 
            [-1, 0, 1], 
            [-1, 0, 1]
        ])
    elif kernel == 'scharr':
        kernel = np.array([
            [-3, 0, 3], 
            [-10, 0, 10], 
            [-3, 0, 3]
        ])

    # convolve
    sobel_left = cv2.filter2D(img, -1, kernel)
    sobel_right = cv2.filter2D(img, -1, kernel*-1)
    sobel_top = cv2.filter2D(img, -1, kernel.T)
    sobel_bottom = cv2.filter2D(img, -1, (kernel*-1).T)

    if mask_center and mask_radius and mask_radius > 1:
        mask = np.zeros_like(img)
        cv2.circle(mask, mask_center, mask_radius, (255, 255, 255), -1)
        sobel_left = cv2.bitwise_and(sobel_left, mask)
        sobel_right = cv2.bitwise_and(sobel_right, mask)
        sobel_top = cv2.bitwise_and(sobel_top, mask)
        sobel_bottom = cv2.bitwise_and(sobel_bottom, mask)

    # threshold
    sobel_left = cv2.threshold(sobel_left, threshold, 255, cv2.THRESH_BINARY)[1]
    sobel_right = cv2.threshold(sobel_right, threshold, 255, cv2.THRESH_BINARY)[1]
    sobel_top = cv2.threshold(sobel_top, threshold, 255, cv2.THRESH_BINARY)[1]
    sobel_bottom = cv2.threshold(sobel_bottom, threshold, 255, cv2.THRESH_BINARY)[1]

    ## for each row in image, find the first non-zero element - everything else is set to 0
    if edge_detection == 'normal':
        edge_left = find_left_activations(cv2.flip(sobel_left, 1))
        edge_left = cv2.flip(edge_left, 1)

        edge_right = find_left_activations(sobel_right)

        edge_top = find_left_activations(np.rot90(sobel_top, 3))
        edge_top = np.rot90(edge_top, 1)

        edge_bottom = find_left_activations(np.rot90(sobel_bottom, 1))
        edge_bottom = np.rot90(edge_bottom, 3)

        mean_right = np.mean(edge_right.nonzero()[1])
        mean_left = np.mean(edge_left.nonzero()[1])
        # check if nan
        if math.isnan(mean_left):
            mean_left = 0
        if math.isnan(mean_right):
            mean_right = 0
        edge_top[:, :int(mean_right)] = 0
        edge_top[:, int(mean_left):] = 0
        edge_bottom[:, :int(mean_right)] = 0
        edge_bottom[:, int(mean_left):] = 0

    elif edge_detection == 'inverse':
        edge_left = find_left_activations(sobel_left)

        edge_right = find_left_activations(cv2.flip(sobel_right, 1))
        edge_right = cv2.flip(edge_right, 1)

        edge_top = find_left_activations(np.rot90(sobel_top, 1))
        edge_top = np.rot90(edge_top, 3)

        edge_bottom = find_left_activations(np.rot90(sobel_bottom, 3))
        edge_bottom = np.rot90(edge_bottom, 1)

        mean_left = np.mean(edge_left.nonzero()[1])
        mean_right = np.mean(edge_right.nonzero()[1])
        if math.isnan(mean_left):
            mean_left = 0
        if math.isnan(mean_right):
            mean_right = 0
        edge_left[:, int(mean_left):] = 0
        edge_right[:, :int(mean_right)] = 0

        mean_left = np.mean(edge_left.nonzero()[1])
        mean_right = np.mean(edge_right.nonzero()[1])

        if math.isnan(mean_left):
            mean_left = 0
        if math.isnan(mean_right):
            mean_right = 0
    
        edge_top[:, int(mean_right):] = 0
        edge_top[:, :int(mean_left)] = 0
        edge_bottom[:, int(mean_right):] = 0
        edge_bottom[:, :int(mean_left)] = 0

    # merge
    iris = cv2.bitwise_or(edge_left, edge_right)
    if use_top:
        iris = cv2.bitwise_or(iris, edge_top)
    if use_bottom:
        iris = cv2.bitwise_or(iris, edge_bottom)
    return iris

def crop_circle(img, center, radius, eps=0):
    # Define the bounding box for the iris
    radius = radius + eps
    x_min = max(center[0] - radius, 0)
    x_max = min(center[0] + radius, img.shape[1])
    y_min = max(center[1] - radius, 0)
    y_max = min(center[1] + radius, img.shape[0])

    center_crop = (center[0] - x_min, center[1] - y_min)
    radius_crop = radius

    # Crop the iris from the input image
    img_crop = img[y_min:y_max, x_min:x_max]

    # set pixels outside the iris to black
    mask = np.zeros_like(img_crop)
    mask = cv2.circle(mask, center_crop, radius_crop, (255, 255, 255), -1)
    img_crop = cv2.bitwise_and(img_crop, mask)

    return img_crop, center_crop, radius_crop

def detect_iris(img, crop=True):
    iris = get_cicle_points(img, blur='median', threshold=40, use_bottom=False, use_top=False)
    center, radius = get_circle(iris)
    if img is None or center is None or radius is None:
        return None, None, None
    if not crop:
        return img, center, radius
    iris_crop, center_crop, radius_crop = crop_circle(img, center, radius)

    return iris_crop, center_crop, radius_crop

def detect_pupil(iris, iris_center, iris_radius, crop=True, eps=1):
    if iris is None or iris_center is None or iris_radius is None:
        return None, None, None
    channels = get_pupil_channels(iris)
    scores = []
    fixed_channels = []
    for channel in channels:
        channel, score = feature_score(channel, iris_center, iris_radius)
        scores.append(score)
        fixed_channels.append(channel)
    # ugly fix to prio reds and hue
    scores[6] = scores[6] * 2
    best_channel = np.argmax(scores)
    input_channel = fixed_channels[best_channel]
    pupil = get_cicle_points(input_channel, kernel='sobel',blur='median',threshold=100, channel='single', mask_center=iris_center, mask_radius=iris_radius-20, edge_detection='inverse', use_top=True)
    center, radius = get_circle(pupil)
    if center is None or radius is None:
        return None, None, None
    if not crop:
        return iris, center, radius
    pupil_crop, center_crop, radius_crop = crop_circle(iris, center, radius, eps)
    return pupil_crop, center_crop, radius_crop

def detect_pupil_todo(iris, iris_center, iris_radius, crop=True, eps=1):
    # to hsv
    hsv = cv2.cvtColor(iris, cv2.COLOR_RGB2HSV)
    h_img = hsv[:,:,2]
    blurred = cv2.medianBlur(h_img, 5)
    # threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # invert
    #thresh = cv2.bitwise_not(thresh)
    # fill holes from corners
    cv2.floodFill(thresh, None, (0, 0), 0)
    cv2.floodFill(thresh, None, (0, thresh.shape[0]-1), 0)
    cv2.floodFill(thresh, None, (thresh.shape[1]-1, 0), 0)
    cv2.floodFill(thresh, None, (thresh.shape[1]-1, thresh.shape[0]-1), 0)

    pupil = get_cicle_points(thresh, kernel='sobel',blur='median',threshold=15, channel='single', mask_center=iris_center, mask_radius=iris_radius-10, edge_detection='inverse', use_top=False)
    center, radius = get_circle(pupil)
    if center is None or radius is None:
        return None, None, None
    if not crop:
        return iris, center, radius
    pupil_crop, center_crop, radius_crop = crop_circle(iris, center, radius, eps)
    return pupil_crop, center_crop, radius_crop

def pipe(img, resample=0, segment='pupil'):

    right_eye, left_eye = detect_eyes(img, resample=resample)
    right_iris, right_center, right_radius = detect_iris(right_eye)
    left_iris, left_center, left_radius = detect_iris(left_eye)
    if segment == 'iris':
        return right_iris, left_iris, right_radius, left_radius
    right_pupil, right_pupil_center, right_pupil_radius = detect_pupil(right_iris, right_center, right_radius)
    left_pupil, left_pupil_center, left_pupil_radius = detect_pupil(left_iris, left_center, left_radius)

    if right_pupil is None:
        print("Right pupil not found")
        right_pupil = np.zeros_like(right_eye)
    if left_pupil is None:
        print("Left pupil not found")
        left_pupil = np.zeros_like(left_eye)

    return right_pupil, left_pupil

def feature_score(feature, iris_center, iris_radius):
    pupil_r = iris_radius // 2 - 10
    feature_mask = np.zeros_like(feature, dtype=np.uint8)
    cv2.circle(feature_mask, iris_center, iris_radius, 255, thickness=-1)
    # set pixels outside the circle to 0
    feature = cv2.bitwise_and(feature, feature_mask)
    mask = np.zeros_like(feature, dtype=np.uint8)
    try:
        cv2.circle(mask, iris_center, pupil_r, 255, thickness=-1)
    except:
        return feature, 0
    center_mask = np.zeros_like(feature, dtype=np.uint8)
    cv2.circle(center_mask, iris_center, 5, 255, thickness=-1)
    center_mask = cv2.bitwise_not(center_mask)
    mask = cv2.bitwise_and(mask, center_mask)

    inverted_mask = cv2.bitwise_not(mask)
    inverted_mask = cv2.bitwise_and(inverted_mask, feature_mask)
    inverted_mask = cv2.bitwise_and(inverted_mask, center_mask)

    mean_intensity_pupil = cv2.mean(feature, mask=mask)[0]
    mean_intensity_iris = cv2.mean(feature, mask=inverted_mask)[0]
    score = round((mean_intensity_pupil - mean_intensity_iris) ** 2)
    if mean_intensity_pupil < mean_intensity_iris:
        feature = cv2.bitwise_not(feature)
        feature = cv2.bitwise_and(feature, feature_mask)
    return feature, score

def get_pupil_channels(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    r, g, bl = cv2.split(img)
    y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb))
    channels = [l, a, b, h, s, v, r, g, bl, y, cr, cb]
    channels = [cv2.GaussianBlur(channel, (5, 5), 0) for channel in channels]
    channels = [cv2.equalizeHist(channel) for channel in channels]
    return channels

def ransac_ellipse(points, num_iterations=10000, threshold=1.00):
    """Fit an ellipse to the given points using RANSAC."""
    if len(points) < 5:
        return None
    best_ellipse = None
    best_inliers = 0

    for _ in range(num_iterations):
        # Randomly sample 5 points
        sample_points = random.sample(points, 5)
        ellipse = cv2.fitEllipse(np.array(sample_points))

        # Calculate inliers
        inliers = 0
        for p in points:
            dist = cv2.pointPolygonTest(ellipse, p, True)
            if abs(dist) <= threshold:
                inliers += 1

        # Update best ellipse if current one has more inliers
        if inliers > best_inliers:
            best_inliers = inliers
            best_ellipse = ellipse

    return best_ellipse

def get_ellipse(img):
    points = np.argwhere(img == 255)
    points = [tuple(point) for point in points]

    ellipse = ransac_ellipse(points)
    if ellipse:
        return ellipse
    else:
        return None