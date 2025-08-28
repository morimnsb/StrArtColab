import os, glob, imageio.v2 as imageio

SRC = "outputs_progressive/progress_frames"
OUT = "outputs_progressive/progress.gif"
FPS = 20  # tweak speed

frames = sorted(glob.glob(os.path.join(SRC, "progress_*.png")))
imgs = [imageio.imread(f) for f in frames]
imageio.mimsave(OUT, imgs, fps=FPS)
print(f"✅ saved {OUT} ({len(frames)} frames)")

