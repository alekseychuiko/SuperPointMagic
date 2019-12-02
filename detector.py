import cv2
import numpy as np
import matcher

class Detector:
    def __init__(self, matchMultiplier, nn_thresh, method, reprThreshold, maxIter, confidence):
        self.matchMultiplier = matchMultiplier
        self.method = method
        self.reprThreshold = reprThreshold
        self.maxIter = maxIter
        self.confidence = confidence
        self.nn_thresh = nn_thresh
        
    def rectPerspectiveTransform(self, x, y, width, height, H):
        corners = np.empty((4,1,2), dtype=np.float32)
        corners[0,0,0] = x
        corners[0,0,1] = y
        corners[1,0,0] = x + width
        corners[1,0,1] = y
        corners[2,0,0] = x + width
        corners[2,0,1] = y + height
        corners[3,0,0] = x
        corners[3,0,1] = y + height
        return cv2.perspectiveTransform(corners, H)
        
    def drawTransformedRect(self, image, corners, color, thickness):
        cv2.line(image, (corners[0,0,0], corners[0,0,1]), (corners[1,0,0], corners[1,0,1]), color, thickness)
        cv2.line(image, (corners[1,0,0], corners[1,0,1]), (corners[2,0,0], corners[2,0,1]), color, thickness)
        cv2.line(image, (corners[2,0,0], corners[2,0,1]), (corners[3,0,0], corners[3,0,1]), color, thickness)
        cv2.line(image, (corners[3,0,0], corners[3,0,1]), (corners[0,0,0], corners[0,0,1]), color, thickness)
    
    def checkTransformedRect(self, image, corners):
        width = image.shape[1];
        height = image.shape[0];  
        result = True;
        for corner in corners:
            result = result and (corner[0,0] < width) and (corner[0,0] > 0) and (corner[0,1] < height) and (corner[0,1] > 0)
        return result
    
    def detect(self, img, objpoints, imagepoints, objdesc, imgdesc, deviceObject, showKeypoints):
        
        good_matches = matcher.nn_match_two_way(imgdesc, objdesc, self.nn_thresh)
        if (len(good_matches[0]) >= 4 ):
            obj = np.empty((len(good_matches[0]),2), dtype=np.float32)
            scene = np.empty((len(good_matches[0]),2), dtype=np.float32)
            sceneSize = np.empty((len(good_matches[0]),1), dtype=np.float32)
            sceneOri = np.empty((len(good_matches[0]),1), dtype=np.float32)
            for i in range(len(good_matches[0])):
                #-- Get the keypoints from the good matches
                obj[i,0] = objpoints[int(good_matches[1,i])].pt[0]
                obj[i,1] = objpoints[int(good_matches[1,i])].pt[1]
                scene[i,0] = imagepoints[int(good_matches[0,i])].pt[0]
                scene[i,1] = imagepoints[int(good_matches[0,i])].pt[1]
                sceneSize[i] = imagepoints[int(good_matches[0,i])].size
                sceneOri[i] = imagepoints[int(good_matches[0,i])].angle
            H, _ =  cv2.findHomography(obj, scene, self.method, self.reprThreshold, maxIters = self.maxIter, confidence = self.confidence)
            if not H is None:
                deviceCorners = self.rectPerspectiveTransform(0, 0, deviceObject.image.shape[1], deviceObject.image.shape[0], H)
                if (self.checkTransformedRect(img, deviceCorners) and cv2.isContourConvex(deviceCorners)):
                    self.drawTransformedRect(img, deviceCorners, (0, 0, 255), 2)
                    for led in deviceObject.leds:
                        corners = self.rectPerspectiveTransform(led['x'], led['y'], led['width'], led['height'], H)
                        self.drawTransformedRect(img, corners, (255, 0, 0), 1)

        if showKeypoints == 1 :
            out = cv2.drawKeypoints(img, imagepoints, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        elif showKeypoints == 2 and len(good_matches) >= 4 :
            keypoints = list()
            for i in range(len(scene)):
                keypoints.append(cv2.KeyPoint(scene[i][0], scene[i][1], sceneSize[i], sceneOri[i]))
            out = cv2.drawKeypoints(img, keypoints, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            out = img.copy()
        return out