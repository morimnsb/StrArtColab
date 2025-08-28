# importance/build_map.py
import cv2
import numpy as np

def build_importance_map(gray_img):
    """
    Build an importance map for string art.

    Parameters:
        gray_img (np.ndarray): Grayscale input image (0–255).

    Returns:
        M (np.ndarray, float32): Importance map (0–1 range).
        mask (np.ndarray, uint8): Binary mask of letter/shape (255=letter, 0=background).
    """
    # Ensure grayscale
    if len(gray_img.shape) == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    # Normalize
    norm = gray_img.astype(np.float32) / 255.0

    # Threshold to create letter mask (Otsu for auto threshold)
    _, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Importance: here simply invert so darker = more important
    M = 1.0 - norm

    # Optionally multiply with mask so background importance is 0
    M *= (mask > 0).astype(np.float32)

    return M.astype(np.float32), mask.astype(np.uint8)
