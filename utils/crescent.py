from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.measure import label, regionprops

import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

def dist_feature(img_in):
    img = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
    center = (img.shape[0]//2, img.shape[1]//2)
    r = img
    dist = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dist[i, j] = np.linalg.norm((i-center[0], j-center[1])) 
    return dist

def intensity_features(img):
    return img[:, :, 0], img[:, :, 1], img[:, :, 2]

def intensity_gradient_features(img, mode="rgb"):
    if mode == "gray":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return np.gradient(gray)[0], np.gradient(gray)[1]
    else:
        r, g, b = intensity_features(img)
        return np.gradient(r)[0], np.gradient(r)[1], np.gradient(g)[0], np.gradient(g)[1], np.gradient(b)[0], np.gradient(b)[1]
    
def get_features(img):
    features = []
    features.append(dist_feature(img))
    features.extend(intensity_features(img))
    features.extend(intensity_gradient_features(img))
    return np.array(features)

def largest_connected_component(mask):
    labeled = label(mask)
    props = regionprops(labeled)
    areas = [p.area for p in props]
    largest_idx = np.argmax(areas)
    return labeled == largest_idx + 1

def get_short_axis(ellipse):
    a = ellipse[1][0] / 2
    b = ellipse[1][1] / 2
    return np.sqrt(b**2 - a**2)

# calculate angle between vertical line, and center of image to center of ellipse
def get_angle(ellipse, img, orientation="H"):
    center = (img.shape[0]//2, img.shape[1]//2)
    angle = np.arctan2(ellipse[0][1] - center[1], ellipse[0][0] - center[0])
    if orientation == "V":
        angle += np.pi/2
    # convert to degrees
    if angle > np.pi / 2:
        angle -= np.pi
    elif angle < -np.pi / 2:
        angle += np.pi
    angle = np.degrees(angle)
    return angle

def ellipse_to_point_dist(ellipse, point):
    center = ellipse[0]
    angle = ellipse[2]
    a = ellipse[1][0] / 2
    b = ellipse[1][1] / 2
    x = point[0] - center[0]
    y = point[1] - center[1]
    dist = (x*np.cos(angle) + y*np.sin(angle))**2 / a**2 + (x*np.sin(angle) - y*np.cos(angle))**2 / b**2
    return dist

def ransac_ellipse(points, max_iter, threshold=5, sample_size=20):
    if len(points) < 5:
        return None
    best_model = None
    best_ic = 0
    for epoch in range(max_iter):
        sample = random.sample(points, sample_size)
        model = cv2.fitEllipse(np.array(sample))
        ic = 0
        for p in points:
            if ellipse_to_point_dist(model, p) < threshold:
                ic += 1
        if ic > best_ic:
            best_ic = ic
            best_model = model
    return best_model

def get_mask(img, model):
    features = get_features(img)
    features = features.reshape(features.shape[0], -1)
    mask = model.predict(features.T)
    mask = mask.reshape(img.shape[0], img.shape[1])
    # check if mask is empty
    if np.sum(mask) == 0:
        return mask
    mask = largest_connected_component(mask)
    mask = mask.astype(np.uint8) * 255
    return mask

def pipe(img, model, mode="least_squares", threshold=6.0, sample_size=20, max_iter=5000, orientation="H"):
    mask = get_mask(img, model)
    if np.sum(mask) < 0:
        return None, None, None
    border = mask.copy()
    border = cv2.Canny(border.astype(np.uint8), 0, 1)
    border_points = np.argwhere(border)
    border_points = border_points[:, ::-1]
    if len(border_points) < 5:
        return None, None, None
    if mode == "ransac":
        best_ellipse = ransac_ellipse(border_points.tolist(), max_iter, threshold=threshold, sample_size=sample_size)
    elif mode == "least_squares":
        best_ellipse = cv2.fitEllipse(border_points)
    short_axis = get_short_axis(best_ellipse)
    angle = get_angle(best_ellipse, img, orientation)
    # if mode is H: sign is positive if ellipse is on the right side of the image
    # if mode is V: sign is positive if ellipse is on the top side of the image
    sign = 1
    if orientation == "H":
        if best_ellipse[0][0] < img.shape[0] // 2:
            sign = -1
    elif orientation == "V":
        if best_ellipse[0][1] > img.shape[1] // 2:
            sign = -1
    return short_axis, angle, sign