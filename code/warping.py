import numpy as np
import cv2
from blending import blend_two_images


def solve_homography(u, v):
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    A = np.zeros((2*N, 9))

    for i in range(N):
        ux, uy = u[i]
        vx, vy = v[i]

        A[2 * i] = [ux, uy, 1, 0, 0, 0, -ux * vx, -uy * vx, -vx]
        A[2 * i + 1] = [0, 0, 0, ux, uy, 1, -ux * vy, -uy * vy, -vy]

    U, S, V = np.linalg.svd(A)
    H = V[-1]
    H = H.reshape((3, 3))

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax):
    h_src, w_src, ch = src.shape
    im2 = np.zeros_like(dst, dtype=np.uint8)
    H_inv = np.linalg.inv(H)

    X, Y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
        
    ones = np.ones_like(X)
    coords = np.stack([X, Y, ones], axis=-1).reshape(-1, 3)

    uvw = H_inv @ coords.T
    uv = uvw / (uvw[2] + 1e-10)

    u = uv[0].reshape(X.shape)
    v = uv[1].reshape(X.shape)

    mask = (u >= 0) & (u < w_src) & (v >= 0) & (v < h_src)

    mask_X = X[mask]
    mask_Y = Y[mask]
    mask_u = u[mask].astype('int')
    mask_v = v[mask].astype('int')

    im2[mask_Y, mask_X] = src[mask_v, mask_u]

    blended = blend_two_images(dst, im2)

    return blended


def cylindrical_warp(img, f):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    y_i, x_i = np.indices((h, w))
    x_ = x_i - cx
    y_ = y_i - cy

    theta = np.arctan(x_ / f)
    h_ = y_ / np.sqrt(x_**2 + f**2)

    x_c = f * theta + cx
    y_c = f * h_ + cy

    map_x = x_c.astype(np.float32)
    map_y = y_c.astype(np.float32)

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped



