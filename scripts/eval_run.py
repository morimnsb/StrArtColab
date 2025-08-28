#!/usr/bin/env python3
import argparse, json, os, cv2, numpy as np

def circle_mask(h, w, margin=0.0):
    yy, xx = np.mgrid[0:h,0:w].astype(np.float32)
    cy, cx = (h-1)/2.0, (w-1)/2.0
    r = min(cx, cy) - margin
    return (((yy-cy)**2 + (xx-cx)**2) <= r*r).astype(np.float32)

def preprocess_u8(img_u8, size, gamma, clahe_clip, clahe_grid, invert, circle):
    img = cv2.resize(img_u8, tuple(size[::-1]), interpolation=cv2.INTER_AREA)  # size=[H,W]
    clahe = cv2.createCLAHE(clipLimit=max(0.01, clahe_clip), tileGridSize=(clahe_grid, clahe_grid))
    img = clahe.apply(img)
    f = (img.astype(np.float32)/255.0)
    f = np.power(f, 1.0/max(1e-6, gamma))
    if invert: f = 1.0 - f
    if circle:
        mask = circle_mask(f.shape[0], f.shape[1], margin=1.0)
        f = f*mask + 1.0*(1.0-mask)
    return np.clip(f,0,1).astype(np.float32)

def ssim_simple(x, y, C1=0.01**2, C2=0.03**2):
    x = x.astype(np.float32); y = y.astype(np.float32)
    mu_x = cv2.GaussianBlur(x, (11,11), 1.5); mu_y = cv2.GaussianBlur(y, (11,11), 1.5)
    sigma_x = cv2.GaussianBlur(x*x,(11,11),1.5) - mu_x*mu_x
    sigma_y = cv2.GaussianBlur(y*y,(11,11),1.5) - mu_y*mu_y
    sigma_xy= cv2.GaussianBlur(x*y,(11,11),1.5) - mu_x*mu_y
    num = (2*mu_x*mu_y + C1)*(2*sigma_xy + C2)
    den = (mu_x*mu_x + mu_y*mu_y + C1)*(sigma_x + sigma_y + C2)
    return float((num/(den+1e-12)).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--target', required=True)
    args = ap.parse_args()

    # read recipe to mirror preprocessing + size
    recipe = json.load(open(os.path.join(args.out_dir,'recipe.json'),'r'))
    H,W = recipe['size']
    pp = recipe['preprocess']
    circle = bool(pp.get('circle_mask', False)) and (recipe['layout']['shape'] == 'circle')

    raw = cv2.imread(args.target, cv2.IMREAD_GRAYSCALE)
    tgt = preprocess_u8(raw, size=[H,W],
                        gamma=float(pp['gamma']),
                        clahe_clip=float(pp['clahe_clip']),
                        clahe_grid=int(pp['clahe_grid']),
                        invert=bool(pp['invert']),
                        circle=circle)
    desired_dark = 1.0 - tgt  # same as generator

    # read accumulated darkness
    dark_png = os.path.join(args.out_dir,'accumulated_darkness.png')
    sim_dark = cv2.imread(dark_png, cv2.IMREAD_GRAYSCALE)
    if sim_dark is None:
        sim = cv2.imread(os.path.join(args.out_dir,'simulated_from_darkness.png'), cv2.IMREAD_GRAYSCALE)
        sim_dark = 1.0 - (sim.astype(np.float32)/255.0)
    else:
        sim_dark = (sim_dark.astype(np.float32)/255.0)

    print(f"SSIM (preprocessed desired vs accumulated): {ssim_simple(desired_dark, sim_dark):.4f}")
    print("Lines:", recipe['stats']['lines_drawn'], "Seconds:", recipe['stats']['seconds'])

if __name__ == '__main__':
    main()
