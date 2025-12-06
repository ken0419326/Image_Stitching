import cv2
import numpy as np
import pywt
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates

def harris_corner_detector(img, octave, sigma_d = 1, sigma_i = 1.5, sigma_o = 4.5, c_robust = 0.75, keypoint_num = 500):
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_d, sigmaY=sigma_d)

    Ix = cv2.Sobel(blurred, cv2.CV_64F, dx=1, dy=0, ksize=3)
    Iy = cv2.Sobel(blurred, cv2.CV_64F, dx=0, dy=1, ksize=3)

    Ixx = cv2.GaussianBlur(Ix * Ix, (0, 0), sigma_i)
    Iyy = cv2.GaussianBlur(Iy * Iy, (0, 0), sigma_i)
    Ixy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma_i)

    det = Ixx * Iyy - Ixy**2
    tr = Ixx + Iyy
    f_hm = det / (tr + 1e-10)

    threshold = 100

    ones = np.ones((3, 3), dtype=np.float64)
    is_max = (f_hm == cv2.dilate(f_hm, kernel=ones))
    mask = is_max & (f_hm > threshold)
    y, x = np.where(mask)
    keypoints = np.stack((y, x), axis=-1)
    scores = f_hm[y, x]

    # adaptive non-maximal suppression
    sorted_idx = np.argsort(-scores)
    sorted_scores = scores[sorted_idx]
    sorted_kps = keypoints[sorted_idx]

    radius = np.full(len(sorted_kps), 1e10)

    for i in range(1, len(sorted_kps)):
        strong_kps = sorted_kps[:i]
        strong_scores = sorted_scores[:i]
        current_score = sorted_scores[i]

        mask = c_robust * strong_scores > current_score
        tree = cKDTree(strong_kps[mask])

        if np.any(mask):
            dists, _ = tree.query([sorted_kps[i]], k=1)
            radius[i] = dists[0]

    top_idx = np.argsort(-radius)[:keypoint_num]
    selected_kps = sorted_kps[top_idx]

    # orientation
    oriented_blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_o, sigmaY=sigma_o)
    delta_Px = cv2.Sobel(oriented_blurred, cv2.CV_64F, dx=1, dy=0, ksize=3)
    delta_Py = cv2.Sobel(oriented_blurred, cv2.CV_64F, dx=0, dy=1, ksize=3)

    u_length = cv2.magnitude(delta_Px, delta_Py)

    selected_y, selected_x = selected_kps[:, 0], selected_kps[:, 1]
    cos_t = (delta_Px / (u_length + 1e-10))[selected_y, selected_x]
    sin_t = (delta_Py / (u_length + 1e-10))[selected_y, selected_x]
    octaves = np.full(selected_x.shape, octave)

    detected_kps = np.stack((selected_y, selected_x, cos_t, sin_t, octaves), axis=-1)

    return detected_kps

def extract_msop_descriptor(pyramid, keypoints, spacing=5):
    descriptors = []
    coords = []

    # Grid for 8x8 sampling (centered at 0,0), scaled by spacing
    half_size = spacing * 4  
    grid = np.linspace(-half_size + spacing / 2, half_size - spacing / 2, 8)
    grid_x, grid_y = np.meshgrid(grid, grid)

    for kp in keypoints:
        y, x, cos_t, sin_t, level = kp
        level = int(level)
        if level < 0 or level >= len(pyramid):
            continue

        img = pyramid[level].astype(np.float64)

        rot_x = grid_x * cos_t - grid_y * sin_t
        rot_y = grid_x * sin_t + grid_y * cos_t

        sample_x = x + rot_x
        sample_y = y + rot_y

        coords_2d = np.vstack([sample_y.ravel(), sample_x.ravel()])
        try:
            patch = map_coordinates(img, coords_2d, order=1, mode='reflect').reshape(8, 8)
        except Exception:
            continue

        std = np.std(patch)
        if std < 1e-6:
            continue
        patch = (patch - np.mean(patch)) / std

        A, (H, V, D) = pywt.dwt2(patch, 'haar')
        descriptor = np.concatenate([A.flatten(), H.flatten(), V.flatten(), D.flatten()])
        descriptors.append(descriptor)
        coords.append([x*(2**level), y*(2**level)])

    return np.array(coords).astype(np.float32), np.array(descriptors).astype(np.float32)


def msop_DetectandDiscrptor(img):
    keypoints = []
    pyramid = []

    pyr_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for octave in  range(1):
        octave_keypoints = harris_corner_detector(pyr_img, octave)
        keypoints.append(octave_keypoints)
        pyramid.append(pyr_img)
        pyr_img = cv2.pyrDown(pyr_img)

    keypoints = np.concatenate(keypoints, axis=0)

    kps, descriptors = extract_msop_descriptor(pyramid, keypoints)

    return kps, descriptors.astype(np.float32)


