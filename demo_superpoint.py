#!/usr/bin/env python
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%


import argparse
import cv2
import time
import numpy as np
import os
import videostreamer
import superpointfrontend
import matcher
# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

class PointTracker(object):
  """ Class to manage a fixed memory of points and descriptors that enables
  sparse optical flow point tracking.

  Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
  tracks with maximum length L, where each row corresponds to:
  row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
  """

  def __init__(self, max_length, nn_thresh):
    if max_length < 2:
      raise ValueError('max_length must be greater than or equal to 2.')
    self.maxl = max_length
    self.nn_thresh = nn_thresh
    self.all_pts = []
    for n in range(self.maxl):
      self.all_pts.append(np.zeros((2, 0)))
    self.last_desc = None
    self.tracks = np.zeros((0, self.maxl+2))
    self.track_count = 0
    self.max_score = 9999

  def get_offsets(self):
    """ Iterate through list of points and accumulate an offset value. Used to
    index the global point IDs into the list of points.

    Returns
      offsets - N length array with integer offset locations.
    """
    # Compute id offsets.
    offsets = []
    offsets.append(0)
    for i in range(len(self.all_pts)-1): # Skip last camera size, not needed.
      offsets.append(self.all_pts[i].shape[1])
    offsets = np.array(offsets)
    offsets = np.cumsum(offsets)
    return offsets

  def update(self, pts, desc):
    """ Add a new set of point and descriptor observations to the tracker.

    Inputs
      pts - 3xN numpy array of 2D point observations.
      desc - DxN numpy array of corresponding D dimensional descriptors.
    """
    if pts is None or desc is None:
      print('PointTracker: Warning, no points were added to tracker.')
      return
    assert pts.shape[1] == desc.shape[1]
    # Initialize last_desc.
    if self.last_desc is None:
      self.last_desc = np.zeros((desc.shape[0], 0))
    # Remove oldest points, store its size to update ids later.
    remove_size = self.all_pts[0].shape[1]
    self.all_pts.pop(0)
    self.all_pts.append(pts)
    # Remove oldest point in track.
    self.tracks = np.delete(self.tracks, 2, axis=1)
    # Update track offsets.
    for i in range(2, self.tracks.shape[1]):
      self.tracks[:, i] -= remove_size
    self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
    offsets = self.get_offsets()
    # Add a new -1 column.
    self.tracks = np.hstack((self.tracks, -1*np.ones((self.tracks.shape[0], 1))))
    # Try to append to existing tracks.
    matched = np.zeros((pts.shape[1])).astype(bool)
    matches = matcher.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
    for match in matches.T:
      # Add a new point to it's matched track.
      id1 = int(match[0]) + offsets[-2]
      id2 = int(match[1]) + offsets[-1]
      found = np.argwhere(self.tracks[:, -2] == id1)
      if found.shape[0] > 0:
        matched[int(match[1])] = True
        row = int(found)
        self.tracks[row, -1] = id2
        if self.tracks[row, 1] == self.max_score:
          # Initialize track score.
          self.tracks[row, 1] = match[2]
        else:
          # Update track score with running average.
          # NOTE(dd): this running average can contain scores from old matches
          #           not contained in last max_length track points.
          track_len = (self.tracks[row, 2:] != -1).sum() - 1.
          frac = 1. / float(track_len)
          self.tracks[row, 1] = (1.-frac)*self.tracks[row, 1] + frac*match[2]
    # Add unmatched tracks.
    new_ids = np.arange(pts.shape[1]) + offsets[-1]
    new_ids = new_ids[~matched]
    new_tracks = -1*np.ones((new_ids.shape[0], self.maxl + 2))
    new_tracks[:, -1] = new_ids
    new_num = new_ids.shape[0]
    new_trackids = self.track_count + np.arange(new_num)
    new_tracks[:, 0] = new_trackids
    new_tracks[:, 1] = self.max_score*np.ones(new_ids.shape[0])
    self.tracks = np.vstack((self.tracks, new_tracks))
    self.track_count += new_num # Update the track count.
    # Remove empty tracks.
    keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
    self.tracks = self.tracks[keep_rows, :]
    # Store the last descriptors.
    self.last_desc = desc.copy()
    return

  def get_tracks(self, min_length):
    """ Retrieve point tracks of a given minimum length.
    Input
      min_length - integer >= 1 with minimum track length
    Output
      returned_tracks - M x (2+L) sized matrix storing track indices, where
        M is the number of tracks and L is the maximum track length.
    """
    if min_length < 1:
      raise ValueError('\'min_length\' too small.')
    valid = np.ones((self.tracks.shape[0])).astype(bool)
    good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
    # Remove tracks which do not have an observation in most recent frame.
    not_headless = (self.tracks[:, -1] != -1)
    keepers = np.logical_and.reduce((valid, good_len, not_headless))
    returned_tracks = self.tracks[keepers, :].copy()
    return returned_tracks

  def draw_tracks(self, out, tracks):
    """ Visualize tracks all overlayed on a single image.
    Inputs
      out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
      tracks - M x (2+L) sized matrix storing track info.
    """
    # Store the number of points per camera.
    pts_mem = self.all_pts
    N = len(pts_mem) # Number of cameras/images.
    # Get offset ids needed to reference into pts_mem.
    offsets = self.get_offsets()
    # Width of track and point circles to be drawn.
    stroke = 1
    # Iterate through each track and draw it.
    for track in tracks:
      clr = myjet[int(np.clip(np.floor(track[1]*10), 0, 9)), :]*255
      for i in range(N-1):
        if track[i+2] == -1 or track[i+3] == -1:
          continue
        offset1 = offsets[i]
        offset2 = offsets[i+1]
        idx1 = int(track[i+2]-offset1)
        idx2 = int(track[i+3]-offset2)
        pt1 = pts_mem[i][:2, idx1]
        pt2 = pts_mem[i+1][:2, idx2]
        p1 = (int(round(pt1[0])), int(round(pt1[1])))
        p2 = (int(round(pt2[0])), int(round(pt2[1])))
        cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
        # Draw end points of each track.
        if i == N-2:
          clr2 = (255, 0, 0)
          cv2.circle(out, p2, stroke, clr2, -1, lineType=16)

if __name__ == '__main__':

  # Parse command line arguments.
  parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
  parser.add_argument('input', type=str, default='',
      help='Image directory or movie file or "camera" (for webcam).')
  parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
      help='Path to pretrained weights file (default: superpoint_v1.pth).')
  parser.add_argument('--img_glob', type=str, default='*.png',
      help='Glob match if directory of images is specified (default: \'*.png\').')
  parser.add_argument('--skip', type=int, default=1,
      help='Images to skip if input is movie or directory (default: 1).')
  parser.add_argument('--show_extra', action='store_true',
      help='Show extra debug outputs (default: False).')
  parser.add_argument('--H', type=int, default=480,
      help='Input image height (default: 120).')
  parser.add_argument('--W', type=int, default=640,
      help='Input image width (default:160).')
  parser.add_argument('--display_scale', type=int, default=1,
      help='Factor to scale output visualization (default: 2).')
  parser.add_argument('--min_length', type=int, default=2,
      help='Minimum length of point tracks (default: 2).')
  parser.add_argument('--max_length', type=int, default=5,
      help='Maximum length of point tracks (default: 5).')
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
  parser.add_argument('--no_display', action='store_true',
      help='Do not display images to screen. Useful if running remotely (default: False).')
  parser.add_argument('--write', action='store_true',
      help='Save output frames to a directory (default: False)')
  parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
      help='Directory where to write output frames (default: tracker_outputs/).')
  opt = parser.parse_args()
  print(opt)

  # This class helps load input images from different sources.
  vs = videostreamer.VideoStreamer(opt.input, opt.camid, opt.H, opt.W, opt.skip, opt.img_glob)

  print('==> Loading pre-trained network.')
  # This class runs the SuperPoint network and processes its outputs.
  fe = superpointfrontend.SuperPointFrontend(weights_path=opt.weights_path,
                          nms_dist=opt.nms_dist,
                          conf_thresh=opt.conf_thresh,
                          nn_thresh=opt.nn_thresh,
                          cuda=opt.cuda)
  print('==> Successfully loaded pre-trained network.')

  # This class helps merge consecutive point matches into tracks.
  tracker = PointTracker(opt.max_length, nn_thresh=fe.nn_thresh)

  # Create a window to display the demo.
  if not opt.no_display:
    win = 'SuperPoint Tracker'
    cv2.namedWindow(win)
  else:
    print('Skipping visualization, will not show a GUI.')

  # Font parameters for visualizaton.
  font = cv2.FONT_HERSHEY_DUPLEX
  font_clr = (255, 255, 255)
  font_pt = (4, 12)
  font_sc = 0.4

  # Create output directory if desired.
  if opt.write:
    print('==> Will write outputs to %s' % opt.write_dir)
    if not os.path.exists(opt.write_dir):
      os.makedirs(opt.write_dir)

  print('==> Running Demo.')
  while True:

    start = time.time()

    # Get a new image.
    img, status = vs.next_frame()
    if status is False:
      break

    # Get points and descriptors.
    start1 = time.time()
    pts, desc, heatmap = fe.run(img)
    end1 = time.time()

    # Add points and descriptors to the tracker.
    tracker.update(pts, desc)

    # Get tracks for points which were match successfully across all frames.
    tracks = tracker.get_tracks(opt.min_length)

    # Primary output - Show point tracks overlayed on top of input image.
    out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
    tracks[:, 1] /= float(fe.nn_thresh) # Normalize track scores to [0,1].
    tracker.draw_tracks(out1, tracks)
    if opt.show_extra:
      cv2.putText(out1, 'Point Tracks', font_pt, font, font_sc, font_clr, lineType=16)

    # Extra output -- Show current point detections.
    out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
    for pt in pts.T:
      pt1 = (int(round(pt[0])), int(round(pt[1])))
      cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
    cv2.putText(out2, 'Raw Point Detections', font_pt, font, font_sc, font_clr, lineType=16)

    # Extra output -- Show the point confidence heatmap.
    if heatmap is not None:
      min_conf = 0.001
      heatmap[heatmap < min_conf] = min_conf
      heatmap = -np.log(heatmap)
      heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
      out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
      out3 = (out3*255).astype('uint8')
    else:
      out3 = np.zeros_like(out2)
    cv2.putText(out3, 'Raw Point Confidences', font_pt, font, font_sc, font_clr, lineType=16)

    # Resize final output.
    if opt.show_extra:
      out = np.hstack((out1, out2, out3))
      out = cv2.resize(out, (3*opt.display_scale*opt.W, opt.display_scale*opt.H))
    else:
      out = cv2.resize(out1, (opt.display_scale*opt.W, opt.display_scale*opt.H))

    # Display visualization image to screen.
    if not opt.no_display:
      cv2.imshow(win, out)
      key = cv2.waitKey(opt.waitkey) & 0xFF
      if key == ord('q'):
        print('Quitting, \'q\' pressed.')
        break

    # Optionally write images to disk.
    if opt.write:
      out_file = os.path.join(opt.write_dir, 'frame_%05d.png' % vs.i)
      print('Writing image to %s' % out_file)
      cv2.imwrite(out_file, out)

    end = time.time()
    net_t = (1./ float(end1 - start))
    total_t = (1./ float(end - start))
    print('Processed image %d (net+post_process: %.2f FPS, total: %.2f FPS).' % (vs.i, net_t, total_t))

  # Close any remaining windows.
  cv2.destroyAllWindows()

  print('==> Finshed Demo.')
