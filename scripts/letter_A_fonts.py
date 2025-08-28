import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

# Output folder
output_dir = "letter_A_fonts"
os.makedirs(output_dir, exist_ok=True)

# Canvas + text
TEXT = "A"
W, H = 400, 400
TARGET_FRAC = 0.4  # scale letter to ~80% of min(W,H)

def measure(draw, text, font):
    """Return (tw, th, bbox) using the best available Pillow API."""
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font, anchor=None)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        return tw, th, bbox
    # Fallback for very old Pillow
    tw, th = draw.textsize(text, font=font)
    return tw, th, (0, 0, tw, th)

def fit_font_size(font_path, target_px):
    """Pick a font size so the glyph height is ~target_px."""
    # quick proportional sizing (good enough, avoids heavy loops)
    # start big and adjust once
    trial_size = int(target_px)
    trial = ImageFont.truetype(font_path, trial_size)
    img = Image.new("RGB", (W, H), "white")
    drw = ImageDraw.Draw(img)
    tw, th, _ = measure(drw, TEXT, trial)
    # If too big/small, rescale once proportional to height
    if th > 0:
        scaled = max(10, int(trial_size * (target_px / th)))
        return ImageFont.truetype(font_path, scaled)
    return trial

def render_letter(font_path):
    # choose size so the glyph height ~ 80% of canvas
    target = int(min(W, H) * TARGET_FRAC)
    font = fit_font_size(font_path, target)

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    tw, th, bbox = measure(draw, TEXT, font)
    # Account for negative bearings by using bbox offsets
    x = (W - tw) // 2 - bbox[0]
    y = (H - th) // 2 - bbox[1]

    draw.text((x, y), TEXT, fill="black", font=font)
    return img

# Collect system TTF fonts
font_paths = fm.findSystemFonts(fontext="ttf")
print(f"Found {len(font_paths)} fonts")

saved = 0
for fp in font_paths:
    try:
        img = render_letter(fp)
        fname = os.path.splitext(os.path.basename(fp))[0]
        safe = "".join(c if c.isalnum() else "_" for c in fname)
        out = os.path.join(output_dir, f"A_{safe}.png")
        img.save(out)
        saved += 1
    except Exception as e:
        print(f"Skipping {fp}: {e}")

print(f"✅ Saved {saved} images to: {output_dir}")
