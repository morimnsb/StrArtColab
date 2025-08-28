import numpy as np

def generate_nail_positions(image_shape,count, shape='circle'):
    """
    Generate nail positions around the border of the image.

    Args:
        image_shape (tuple): (height, width) of the image canvas.
        shape (str): 'circle' or 'rectangle'.

    Returns:
        List[Tuple[int, int]]: List of (x, y) nail coordinates.
    """
    h, w = image_shape
    nails = []

    if shape == 'circle':
        count = 360
        center = (w // 2, h // 2)
        radius = int(min(w, h) * 0.45)

        for i in range(count):
            angle = 2 * np.pi * i / count
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            nails.append((x, y))

    elif shape == 'rectangle':
        per_side = 200  # 200 per side → 800 total
        # Top edge (left to right)
        for i in range(per_side):
            x = int(i * (w - 1) / (per_side - 1))
            nails.append((x, 0))
        # Right edge (top to bottom)
        for i in range(per_side):
            y = int(i * (h - 1) / (per_side - 1))
            nails.append((w - 1, y))
        # Bottom edge (right to left)
        for i in range(per_side):
            x = int(w - 1 - i * (w - 1) / (per_side - 1))
            nails.append((x, h - 1))
        # Left edge (bottom to top)
        for i in range(per_side):
            y = int(h - 1 - i * (h - 1) / (per_side - 1))
            nails.append((0, y))

    else:
        raise ValueError("shape must be 'circle' or 'rectangle'")

    return nails
