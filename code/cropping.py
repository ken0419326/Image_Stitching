import cv2
import numpy as np

def find_max_rectangle(mask):
    h, w = mask.shape
    heights = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            if mask[y, x] != 0:
                heights[y, x] = heights[y-1, x] + 1 if y > 0 else 1

    best_area = 0
    best_rect = (0, 0, 0, 0)  # (x1, y1, x2, y2)

    for y in range(h):
        stack = []
        x = 0
        while x <= w:
            cur_height = heights[y, x] if x < w else 0
            if not stack or cur_height >= heights[y, stack[-1]]:
                stack.append(x)
                x += 1
            else:
                top = stack.pop()
                width = x if not stack else x - stack[-1] - 1
                area = heights[y, top] * width
                if area > best_area:
                    best_area = area
                    x1 = (stack[-1] + 1) if stack else 0
                    y1 = y - heights[y, top] + 1
                    x2 = x - 1
                    y2 = y
                    best_rect = (x1, y1, x2, y2)

    return best_rect

def crop_largest_rectangle(input_picture, visualize=False):
    gray = cv2.cvtColor(input_picture, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    x1, y1, x2, y2 = find_max_rectangle(mask)

    cropped = input_picture[y1:y2+1, x1:x2+1]

    return cropped
