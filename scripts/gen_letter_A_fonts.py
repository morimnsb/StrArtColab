import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

OUT_DIR   = os.environ.get("OUT_DIR", "letter_A_fonts")
TEXT      = "A"
W = H     = int(os.environ.get("CANVAS", 400))
TARGET_FRAC = 0.4  # ~40% of min(W,H)

os.makedirs(OUT_DIR, exist_ok=True)

def measure(draw, text, font):
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font, anchor=None)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        return tw, th, bbox
    tw, th = draw.textsize(text, font=font)
    return tw, th, (0,0,tw,th)

def fit_font_size(font_path, target_px):
    trial_size = int(target_px)
    trial = ImageFont.truetype(font_path, trial_size)
    img = Image.new("RGB", (W, H), "white")
    drw = ImageDraw.Draw(img)
    _, th, _ = measure(drw, TEXT, trial)
    if th > 0:
        scaled = max(10, int(trial_size * (target_px / th)))
        return ImageFont.truetype(font_path, scaled)
    return trial

def render_letter(font_path):
    target = int(min(W, H) * TARGET_FRAC)
    font   = fit_font_size(font_path, target)
    img    = Image.new("RGB", (W, H), "white")
    draw   = ImageDraw.Draw(img)
    tw, th, bbox = measure(draw, TEXT, font)
    x = (W - tw)//2 - bbox[0]
    y = (H - th)//2 - bbox[1]
    draw.text((x, y), TEXT, fill="black", font=font)
    return img

font_paths = fm.findSystemFonts(fontext="ttf")
print(f"Found {len(font_paths)} fonts")
saved = 0
for fp in font_paths:
    try:
        img = render_letter(fp)
        fname = os.path.splitext(os.path.basename(fp))[0]
        safe = "".join(c if c.isalnum() else "_" for c in fname)
        out  = os.path.join(OUT_DIR, f"A_{safe}.png")
        img.save(out)
        saved += 1
    except Exception as e:
        print(f"Skipping {fp}: {e}")
print(f"✅ Saved {saved} images to: {OUT_DIR}")
