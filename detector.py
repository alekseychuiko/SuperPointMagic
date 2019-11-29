import cv2
import numpy as np

class Detector:
    def __init__(self, matchMultiplier, normType, method, reprThreshold, maxIter, confidence):
        self.matchMultiplier = matchMultiplier
        self.method = method
        self.reprThreshold = reprThreshold
        self.maxIter = maxIter
        self.confidence = confidence
        self.matcher = cv2.BFMatcher(normType)
        
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
        
        matches	= self.matcher.knnMatch(imgdesc, objdesc, 2)
        
        good_matches = []
        for m,n in matches:
            if m.distance < self.matchMultiplier * n.distance:
                good_matches.append(m)
                
        if (len(good_matches) >= 4 ):
            obj = np.empty((len(good_matches),2), dtype=np.float32)
            scene = np.empty((len(good_matches),2), dtype=np.float32)
            sceneSize = np.empty((len(good_matches),1), dtype=np.float32)
            sceneOri = np.empty((len(good_matches),1), dtype=np.float32)
            for i in range(len(good_matches)):
                #-- Get the keypoints from the good matches
                obj[i,0] = objpoints[good_matches[i].trainIdx].pt[0]
                obj[i,1] = objpoints[good_matches[i].trainIdx].pt[1]
                scene[i,0] = imagepoints[good_matches[i].queryIdx].pt[0]
                scene[i,1] = imagepoints[good_matches[i].queryIdx].pt[1]
                sceneSize[i] = imagepoints[good_matches[i].queryIdx].size
                sceneOri[i] = imagepoints[good_matches[i].queryIdx].angle
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