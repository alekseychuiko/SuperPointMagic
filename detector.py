import cv2
import numpy as np

def detect(img, objpoints, imagepoints, objdesc, imgdesc, obj, showKeypoints):
    
    if showKeypoints:
        out = cv2.drawKeypoints(img, imagepoints, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        out = img.copy()
    return out