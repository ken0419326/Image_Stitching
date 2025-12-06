import os
import numpy as np
import cv2
from tqdm import tqdm
from warping import solve_homography, warping
from feature_extractor import msop_DetectandDiscrptor
from matcher import knn_matcher
from drift import correct_vertical_drift
from cropping import crop_largest_rectangle


np.random.seed(999)

def panorama(imgs):
    h_max = max([2 * x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[(imgs[0].shape[0] // 2):(imgs[0].shape[0] // 2 + imgs[0].shape[0]), :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    last_best_H[1, 2] = imgs[0].shape[0] // 2
    out = None

    keypoints1, descriptors1 = None, None
    keypoints2, descriptors2 = None, None
    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # feature detection & matching
        if idx > 0:
            keypoints1, descriptors1 = keypoints2, descriptors2
        else:
            keypoints1, descriptors1 = msop_DetectandDiscrptor(im1)
        
        keypoints2, descriptors2 = msop_DetectandDiscrptor(im2)

        matches = knn_matcher(descriptors1, descriptors2, ratio_threshold=0.75)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        match_points1 = np.float32([keypoints1[m.queryIdx] for m in matches])
        match_points2 = np.float32([keypoints2[m.trainIdx] for m in matches])

        ones = np.ones(match_points1.shape[0])
        pts1 = np.stack([match_points1[:, 0], match_points1[:, 1]], axis=-1)
        homogeneous_pts2 = np.stack([match_points2[:, 0], match_points2[:, 1], ones], axis=-1)

        # RANSAC
        k = 2000
        n = 4
        threshold = 3.0
        best_inliers_cnt = 0
        best_H = np.eye(3)
        for _ in range(k):
            indices = np.random.choice(range(match_points1.shape[0]), n, replace=False)
            H = solve_homography(match_points2[indices], match_points1[indices])

            uvw = H @ homogeneous_pts2.T
            uv = uvw[:2] / uvw[2]

            error = np.linalg.norm(uv.T - pts1, axis=1)
            inliers = np.where(error < threshold)
            inliers_cnt = np.sum(inliers)

            if inliers_cnt > best_inliers_cnt:
                best_H = H
                best_inliers_cnt = inliers_cnt

        # chain the homographies
        last_best_H = last_best_H @ best_H

        # warping
        dst = warping(im2, dst, last_best_H, 0, h_max, 0, w_max)

    drifted = correct_vertical_drift(dst)
    cropped = crop_largest_rectangle(drifted)
    out = cropped

    return out



if __name__ == "__main__":
    FRAME_NUM = 4
    focal_lengths = []
    imgs = [cv2.imread("../data/frame{:d}.jpg".format(x)) for x in range(1, FRAME_NUM + 1)]
    output = panorama(imgs)
    cv2.imwrite('../result.png', output)