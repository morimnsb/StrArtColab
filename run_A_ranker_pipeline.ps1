# run_A_ranker_pipeline.ps1
# End-to-end pipeline:
#   1) Generate a dataset of "A" letter images from system TTF fonts
#   2) Log shortlist features/gains into an NPZ (via select_lines_progressive)
#   3) Train an edge ranker (regression by default, optional listwise)
#   4) Benchmark: baseline vs trained ranker on A1/A2

param(
  [string]$FontsDir      = "letter_A_fonts",
  [int]   $Canvas        = 400,            # image size used by generator
  [int]   $NumLines      = 500,            # lines per logging/benchmark run
  [int]   $MaxFonts      = 200,            # cap how many fonts to log (keeps it quick)
  [string]$LogsDir       = "logs\train_A_fonts",
  [string]$NPZName       = "ranker_logs.npz",
  [string]$CkptPath      = "ckpts\edge_ranker_A_fonts_reg.pt",
  [switch]$DoListwise,                    # also train a listwise model
  [string]$CkptListwise  = "ckpts\edge_ranker_A_fonts_listwise.pt",
  [int]   $Epochs        = 12,
  [int]   $BatchSize     = 4096,
  [int]   $Hidden        = 128,
  [double]$Dropout       = 0.1,
  [double]$NNWeight      = 0.8,            # for benchmark
  [double]$NeedWeight    = 0.6,            # for benchmark
  [int]   $HybridTopK    = 256,
  [switch]$Fresh                            # if set, delete existing NPZ before logging
)

$ErrorActionPreference = "Stop"

function Run-Py($argv) {
  Write-Host ">>> python $($argv -join ' ')" -ForegroundColor Cyan
  & python @argv
}

# --- 0) Make sure folders exist
New-Item -ItemType Directory -Force -Path $FontsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
New-Item -ItemType Directory -Force -Path "ckpts" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\dataset_previews\A_fonts" | Out-Null
New-Item -ItemType Directory -Force -Path "bench_A_fonts" | Out-Null

$NPZ = Join-Path $LogsDir $NPZName
if ($Fresh -and (Test-Path $NPZ)) {
  Write-Host "Removing existing NPZ: $NPZ" -ForegroundColor Yellow
  Remove-Item $NPZ -Force
}

# --- 1) Create the font generator script (once) and render if directory is empty
$genScriptPath = "scripts\gen_letter_A_fonts.py"
if (!(Test-Path $genScriptPath)) {
@'
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
'@ | Out-File -Encoding UTF8 $genScriptPath
}

$havePNGs = Get-ChildItem $FontsDir -Filter *.png -ErrorAction SilentlyContinue
if (-not $havePNGs) {
  Write-Host "Rendering A-font dataset → $FontsDir ..." -ForegroundColor Green
  $env:OUT_DIR = $FontsDir
  $env:CANVAS  = $Canvas
  Run-Py @("-m","scripts.gen_letter_A_fonts")
  Remove-Item Env:OUT_DIR, Env:CANVAS -ErrorAction SilentlyContinue
}

# --- 2) Log features/gains from a subset of the generated images
$imgs = Get-ChildItem $FontsDir -Filter *.png | Sort-Object Name | Select-Object -First $MaxFonts
Write-Host ("Logging from {0} font images → {1}" -f $imgs.Count, $NPZ) -ForegroundColor Green

$i = 0
foreach ($img in $imgs) {
  $i++
  Write-Host ("[{0}/{1}] {2}" -f $i, $imgs.Count, $img.Name)
  $outSingle = "outputs\dataset_previews\A_fonts\$($img.BaseName)"
  New-Item -ItemType Directory -Force -Path $outSingle | Out-Null

  Run-Py @(
    "-m","scripts.select_lines_progressive",
    "--image",$img.FullName,
    "--num_lines",$NumLines,
    "--scorer","hybrid",
    "--hybrid_topk",$HybridTopK,
    "--log_ranker_npz",$NPZ,
    "--out_dir",$outSingle
  )
}

# --- 3) Train ranker (regression)
Write-Host "Training regression ranker → $CkptPath" -ForegroundColor Green
Run-Py @(
  "-m","scripts.train_edge_ranker",
  "--npz",$NPZ,
  "--out",$CkptPath,
  "--loss","reg",
  "--epochs",$Epochs,
  "--bs",$BatchSize,
  "--hidden",$Hidden,
  "--dropout",$Dropout
)

# Optional: also train listwise
if ($DoListwise) {
  Write-Host "Training listwise ranker → $CkptListwise" -ForegroundColor Green
  Run-Py @(
    "-m","scripts.train_edge_ranker",
    "--npz",$NPZ,
    "--out",$CkptListwise,
    "--loss","listwise",
    "--epochs",$Epochs,
    "--bs","8",            # groups per batch (typical)
    "--hidden",$Hidden,
    "--dropout",$Dropout,
    "--list_temp","1.0"
  )
}

# --- 4) Benchmark on A1/A2
$benchRoot = "bench_A_fonts"
Write-Host "Benchmarking → $benchRoot" -ForegroundColor Green
Run-Py @(
  "-m","scripts.benchmark_rankers",
  "--images","data\A1.png","data\A2.png",
  "--rankers","none",$CkptPath,
  "--out_root",$benchRoot,
  "--num_lines",$NumLines,
  "--scorer","hybrid",
  "--hybrid_topk",$HybridTopK,
  "--nn_weight",$NNWeight,
  "--need_weight",$NeedWeight
)

Write-Host "`n✅ Pipeline finished.
- NPZ: $NPZ
- CKPT (reg): $CkptPath
- Bench output: $benchRoot" -ForegroundColor Green
