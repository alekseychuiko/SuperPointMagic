# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:30:46 2019

@author: Alex
"""
import model
import argparse
import cv2
import time
import numpy as np
import videostreamer
import superpointfrontend
import detector

def convertToKeyPonts(pts):
  keypoints = list()
  for pt in pts.T:
    keypoints.append(cv2.KeyPoint(pt[0], pt[1], pt[2]*255.0))
  return keypoints

if __name__ == '__main__':
  # Parse command line arguments.
  parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
  parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
      help='Path to pretrained weights file (default: superpoint_v1.pth).')
  parser.add_argument('--object_path', type=str,
      help='Path to object to detect')
  parser.add_argument('--H', type=int, default=720,
      help='Input image height (default: 120).')
  parser.add_argument('--W', type=int, default=1280,
      help='Input image width (default:160).')
  parser.add_argument('--nms_dist', type=int, default=4,
      help='Non Maximum Suppression (NMS) distance (default: 4).')
  parser.add_argument('--conf_thresh', type=float, default=0.015,
      help='Detector confidence threshold (default: 0.015).')
  parser.add_argument('--nn_thresh', type=float, default=0.7,
      help='Descriptor matching threshold (default: 0.7).')
  parser.add_argument('--camid', type=int, default=0,
      help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
  parser.add_argument('--waitkey', type=int, default=1,
      help='OpenCV waitkey time in ms (default: 1).')
  parser.add_argument('--cuda', action='store_true',
      help='Use cuda GPU to speed up network processing speed (default: False)')
  parser.add_argument('--show_keypoints', type=int, default=0,
      help='0 - dont show keypoints, 1 - show matched keypoints, 2 - show all keypoints (default: not show)')
  parser.add_argument('--matcher_multiplier', type=float, default=2.0,
      help='Filter matches using the Lowes ratio test (default: 0.4).')
  parser.add_argument('--method', type=int, default=0,
      help='0 - RANSAK, 1 - LMEDS, 2 - RHO (default: 0)')
  parser.add_argument('--repr_threshold', type=int, default=3,
      help='Maximum allowed reprojection error to treat a point pair as an inlier (used in the RANSAC and RHO methods only) (default: 3)')
  parser.add_argument('--max_iter', type=int, default=2000,
      help='Maximum number of RANSAC iterations (default: 2000)')
  parser.add_argument('--confidence', type=float, default=0.995,
      help='homography confidence level (default: 0.995).')
  
  opt = parser.parse_args()
  print(opt)
  
  method = cv2.RANSAC
  if opt.method == 0 : method = cv2.RANSAC
  elif opt.method == 1 : method = cv2.LMEDS
  else : method = cv2.RHO
  # This class helps load input images from different sources.
  vs = videostreamer.VideoStreamer("camera", opt.camid, opt.H, opt.W, 1, '')

  print('==> Loading pre-trained network.')
  # This class runs the SuperPoint network and processes its outputs.
  fe = superpointfrontend.SuperPointFrontend(weights_path=opt.weights_path,
                          nms_dist=opt.nms_dist,
                          conf_thresh=opt.conf_thresh,
                          nn_thresh=opt.nn_thresh,
                          cuda=opt.cuda)
  print('==> Successfully loaded pre-trained network.')

  objDetector = detector.Detector(opt.matcher_multiplier, opt.nn_thresh, method, opt.repr_threshold, opt.max_iter, opt.confidence)

  win = 'SuperPoint Tracker'
  objwin = 'Object'
  cv2.namedWindow(win)
  cv2.namedWindow(objwin)
  
  print('==> Running Demo.')
  
  obj = model.ModelFile(opt.object_path)
  greyObj = cv2.cvtColor(obj.image, cv2.COLOR_BGR2GRAY)
  
  pts, objDesc, heatmap = fe.run(greyObj.astype('float32') / 255.)
  objKeyPoints = convertToKeyPonts(pts)
  if opt.show_keypoints != 0:
    objImg = cv2.drawKeypoints(greyObj, objKeyPoints, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow(objwin, objImg)
  else:
    cv2.imshow(objwin, greyObj)  
  
  
  while True:

    start = time.time()

    # Get a new image.
    img, status = vs.next_frame()
    
    if status is False:
      break

    # Get points and descriptors.
    start1 = time.time()
    pts, imgDesc, heatmap = fe.run(img)
    
    imgKeyPoints = convertToKeyPonts(pts)

    out = objDetector.detect((np.dstack((img, img, img)) * 255.).astype('uint8'), 
                   objKeyPoints, imgKeyPoints, objDesc, imgDesc, obj, opt.show_keypoints)
    
    end1 = time.time()
    cv2.imshow(win, out)
    
    key = cv2.waitKey(opt.waitkey) & 0xFF
    if key == ord('q'):
      print('Quitting, \'q\' pressed.')
      break

    end = time.time()
    net_t = (1./ float(end1 - start))
    total_t = (1./ float(end - start))
    print('Processed image %d (net+post_process: %.2f FPS, total: %.2f FPS).' % (vs.i, net_t, total_t))

  # Close any remaining windows.
  cv2.destroyAllWindows()

  print('==> Finshed Demo.')
