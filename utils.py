# utils.py
import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img

def save_image(path, img):
    cv2.imwrite(path, img)

def draw_keypoints(img, keypoints):
    out = img.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(out, (x, y), 10, (0,255,0), 2)
    return out

def draw_contours_centers(img, centers):
    out = img.copy()
    for (x, y) in centers:
        cv2.circle(out, (int(x), int(y)), 10, (0,255,0), 2)
    return out
