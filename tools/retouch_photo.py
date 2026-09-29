"""Turn a casual photo into a studio-style headshot.

Cuts the subject out, removes small stains on the shirt, crops to head and
shoulders, grades the light and composites onto a soft studio backdrop.

    python tools/retouch_photo.py source.jpg
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets"

# Output headshot size (4:5 portrait).
OUT_W, OUT_H = 1000, 1250


def clean_shirt(rgb: np.ndarray, alpha: np.ndarray, face_bottom: int) -> np.ndarray:
    """Inpaint dark blotches on the shirt below the chin."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    # The shirt is a saturated rust/orange; skin is less saturated.
    shirt = ((h < 16) | (h > 170)) & (s > 110) & (alpha > 200)
    shirt[:face_bottom] = False
    # Only fill small holes in the fabric; a wide close would swallow the neck.
    shirt = cv2.morphologyEx(shirt.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    skin = (s < 105) | ((h > 5) & (h < 25) & (v > 150))
    skin = cv2.dilate(skin.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    shirt &= ~skin

    local = cv2.medianBlur(v, 41).astype(np.int16)
    spots = shirt & (v.astype(np.int16) < local - 12)
    # Ignore large dark regions: those are folds, arms or the print, not stains.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(spots.astype(np.uint8))
    mask = np.zeros_like(v)
    for i in range(1, n):
        if 12 < stats[i, cv2.CC_STAT_AREA] < 9000:
            mask[labels == i] = 255
    mask = cv2.dilate(mask, np.ones((13, 13), np.uint8))
    out = cv2.inpaint(rgb, mask, 15, cv2.INPAINT_TELEA)
    # Blend the patches in with the fabric's own texture-free tone.
    soft = cv2.GaussianBlur(mask, (0, 0), 6)[..., None].astype(np.float32) / 255.0
    fabric = cv2.medianBlur(out, 31)
    return (out * (1 - soft * 0.6) + fabric * (soft * 0.6)).astype(np.uint8)


def grade(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(l)
    lab = cv2.merge([l, a, cv2.add(b, 3)])  # a touch warmer
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # Gentle skin smoothing that keeps edges, then a light sharpen.
    smooth = cv2.bilateralFilter(out, 9, 28, 9)
    out = cv2.addWeighted(out, 0.6, smooth, 0.4, 0)
    blur = cv2.GaussianBlur(out, (0, 0), 1.6)
    return cv2.addWeighted(out, 1.35, blur, -0.35, 0)


def backdrop(w: int, h: int) -> np.ndarray:
    """Soft studio gradient: lit centre behind the head, falling off to the edges."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w * 0.5) / (w * 0.75)) ** 2 + ((yy - h * 0.38) / (h * 0.8)) ** 2)
    t = np.clip(d, 0, 1)[..., None]
    centre = np.array([74, 70, 92], np.float32)
    edge = np.array([22, 20, 32], np.float32)
    bg = centre * (1 - t) + edge * t
    noise = np.random.default_rng(7).normal(0, 1.4, (h, w, 1)).astype(np.float32)
    return np.clip(bg + noise, 0, 255)


def main(src: str) -> None:
    photo = Image.open(src).convert("RGB")
    cut = np.array(remove(photo))
    rgb, alpha = cut[:, :, :3], cut[:, :, 3]

    ys, xs = np.where(alpha > 32)
    top, bottom = ys.min(), ys.max()
    height = bottom - top
    face_bottom = top + int(height * 0.2)
    rgb = clean_shirt(rgb, alpha, face_bottom)
    rgb = grade(rgb)

    # Head-and-shoulders crop centred on the head.
    head_rows = alpha[top : top + int(height * 0.12)] > 128
    head_cols = np.where(head_rows.any(axis=0))[0]
    cx = int((head_cols.min() + head_cols.max()) / 2)
    crop_h = int(height * 0.34)
    crop_w = int(crop_h * OUT_W / OUT_H)
    y0 = max(top - int(crop_h * 0.12), 0)
    x0 = min(max(cx - crop_w // 2, 0), photo.width - crop_w)
    box = (slice(y0, y0 + crop_h), slice(x0, x0 + crop_w))

    fg = cv2.resize(rgb[box], (OUT_W, OUT_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Pull the matte in a little so the old background doesn't halo the edge.
    matte = cv2.erode(alpha[box], np.ones((5, 5), np.uint8))
    a = cv2.resize(matte, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    a = cv2.GaussianBlur(a, (0, 0), 1.2)[..., None]
    bg = backdrop(OUT_W, OUT_H)
    out = fg * a + bg * (1 - a)

    OUT_DIR.mkdir(exist_ok=True)
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    img.save(OUT_DIR / "headshot.jpg", quality=90, optimize=True, progressive=True)
    img.resize((500, 625), Image.LANCZOS).save(OUT_DIR / "headshot-500.jpg", quality=88, optimize=True)
    rgba = np.dstack([fg, a[..., 0] * 255]).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").resize((600, 750), Image.LANCZOS).save(OUT_DIR / "cutout.png", optimize=True)
    print("wrote assets/headshot.jpg, headshot-500.jpg, cutout.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: retouch_photo.py <photo>")
    main(sys.argv[1])
