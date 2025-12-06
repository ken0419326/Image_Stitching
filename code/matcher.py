import numpy as np
from scipy.spatial import cKDTree
import cv2

class Match:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  
        self.trainIdx = trainIdx  
        self.distance = distance  

    def __repr__(self):
        return f"Match(queryIdx={self.queryIdx}, trainIdx={self.trainIdx}, distance={self.distance})"

def knn_matcher(descriptor1, descriptor2, ratio_threshold=0.8, k=2):
    tree = cKDTree(descriptor2)

    matches = []
    for query_idx, des in enumerate(descriptor1):
        dist, train_idx = tree.query(des, k=k)

        if len(dist) >= 2:
            d1, d2 = dist
            if d1 < ratio_threshold * d2:
                match = Match(query_idx, train_idx[0], d1)
                matches.append(match)

    return matches