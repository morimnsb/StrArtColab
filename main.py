import cv2
import numpy as np
import random

from preprocess.preprocess_image import preprocess_image
from nails.generate_nails import generate_nail_positions
from simulation.initialize_mask import initialize_mask
from simulation.draw_line import draw_line_on_mask
from simulation.score_line import score_line

# === CONFIG ===
IMAGE_PATH = "data/A.png"
NAIL_COUNT = 200
LINE_COUNT = 10
LINE_WIDTH = 1
DARKNESS_PER_LINE = 25

# === STEP 1: Load and preprocess image ===
image = preprocess_image(IMAGE_PATH, size=(400, 400), invert=True)
h, w = image.shape

# === STEP 2: Ask for nail layout ===
shape = input("Enter nail layout (circle / rectangle): ").strip().lower()
nails = generate_nail_positions((h, w), shape=shape)

# === STEP 3: Initialize darkness mask ===
mask = initialize_mask((h, w))
start_point = random.choice(nails)
used_lines = set()

# === STEP 4: Drawing loop ===
for line_num in range(LINE_COUNT):
    print(f"\n🧵 Drawing line {line_num + 1}/{LINE_COUNT}")

    best_score = -1
    best_end = None

    for candidate in nails:
        if candidate == start_point:
            continue

        line_key = tuple(sorted([start_point, candidate]))
        if line_key in used_lines:
            continue

        score = score_line(start_point, candidate, image, mask, width=LINE_WIDTH, increment=DARKNESS_PER_LINE)

        if score > best_score:
            best_score = score
            best_end = candidate

    if best_end is None or best_score <= 0:
        print("⚠️ No useful new line found. Skipping.")
        continue

    print(f"Start: {start_point} → End: {best_end} | Score: {best_score:.2f}")
    draw_line_on_mask(mask, start_point, best_end, width=LINE_WIDTH, increment=DARKNESS_PER_LINE)

    used_lines.add(tuple(sorted([start_point, best_end])))
    start_point = best_end

# === Final Output ===
final_image = 255 - mask
cv2.imwrite("output/simulated_result.png", final_image)
print("\n✅ Saved output to output/simulated_result.png")
cv2.imshow("Final Result", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
