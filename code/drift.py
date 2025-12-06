import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

def correct_vertical_drift(panorama):
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    boundary_y = np.zeros(width)

    # For each x-column, find the first non-black pixel from top
    for x in range(width):
        column = gray[:, x]
        non_black = np.where(column > 10)[0]  # threshold to ignore noise
        if len(non_black) > 0:
            boundary_y[x] = (non_black[0] + non_black[-1]) / 2
        else:
            boundary_y[x] = height  # if column is completely black

    # Smooth boundary_y using a spline
    x_full = np.arange(width)
    spline = UnivariateSpline(x_full, boundary_y, s=width*5)
    smoothed_boundary_y = spline(x_full)

    # Choose a reference line (e.g., median height)
    reference_y = np.median(smoothed_boundary_y)

    # Calculate vertical drift (difference from reference)
    vertical_drift = smoothed_boundary_y - reference_y

    # Fit a smooth curve (spline) through vertical drift
    x_samples = np.linspace(0, width-1, num=len(vertical_drift))
    drift_spline = UnivariateSpline(x_samples, vertical_drift, s=5)

    # Create a meshgrid for remapping
    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = map_x.astype(np.float32)  # x stays the same
    map_y = map_y.astype(np.float32)  # y is adjusted below

    # Apply vertical correction
    correction = drift_spline(np.arange(width))  # get correction value at each x
    map_y += correction[np.newaxis, :]            # shift y for each column

    # Remap
    corrected_panorama = cv2.remap(
        panorama, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return corrected_panorama