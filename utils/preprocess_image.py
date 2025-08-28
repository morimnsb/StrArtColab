import cv2
import numpy as np

def preprocess_image(path, size=(400, 400), invert=True):
    """
    Load and preprocess an image for string art.
    
    Args:
        path (str): File path to image.
        size (tuple): Desired output size (height, width).
        invert (bool): If True, inverts brightness (dark becomes important).

    Returns:
        np.ndarray: Grayscale, resized, optionally inverted image (uint8)
    """
    # Load in grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    
    # Resize to standard canvas
    img = cv2.resize(img, (size[1], size[0]))

    # Invert so darker areas are brighter for simulation
    if invert:
        img = 255 - img

    return img
