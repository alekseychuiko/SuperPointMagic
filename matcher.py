# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:52:37 2019

@author: achuiko
"""

import numpy as np

def nn_match_two_way(desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
        desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
        desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
        nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
        matches - 3xL numpy array, of L matches, where L <= N and each column i is
                  a match of two descriptors, d_i in image 1 and d_j' in image 2:
                  [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches