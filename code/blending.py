import numpy as np

def blend_two_images(im1, im2):
    mask1 = (np.sum(im1, axis=2) > 0).astype(np.uint8)
    mask2 = (np.sum(im2, axis=2) > 0).astype(np.uint8)
    overlap = (mask1 & mask2).astype(np.uint8)

    alpha = np.zeros_like(mask1, dtype=np.float32)
    for y in range(alpha.shape[0]):
        overlap_indices = np.where(overlap[y] == 1)[0]
        if len(overlap_indices) > 1:
            min_x, max_x = overlap_indices[0], overlap_indices[-1]
            for x in range(min_x, max_x + 1):
                alpha[y, x] = 1.0 - (x - min_x) / (max_x - min_x + 1e-10)

    result = np.zeros_like(im1)
    for c in range(3):
        result[:, :, c] = (
            alpha * im1[:, :, c] +
            (1 - alpha) * im2[:, :, c]
        ).astype(np.uint8)

    result[(mask1 == 1) & (mask2 == 0)] = im1[(mask1 == 1) & (mask2 == 0)]
    result[(mask1 == 0) & (mask2 == 1)] = im2[(mask1 == 0) & (mask2 == 1)]

    return result

