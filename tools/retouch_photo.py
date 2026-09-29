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
    shirt = ((h < 16) | (h > 170)) & (s > 118) & (alpha > 200)
    shirt[:face_bottom] = False
    # Only fill small holes in the fabric; a wide close would swallow the neck.
    shirt = cv2.morphologyEx(shirt.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    skin = (s < 125) | ((h > 5) & (h < 25) & (v > 140))
    skin = cv2.dilate(skin.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
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
    return cv2.inpaint(rgb, mask, 15, cv2.INPAINT_TELEA)


def skin_mask(rgb: np.ndarray, alpha: np.ndarray, head_bottom: int) -> np.ndarray:
    """Soft mask of face skin (not hair, beard or shirt), 0..1."""
    _, cr, cb = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb))
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    skin = (cr > 135) & (cr < 175) & (cb > 85) & (cb < 130) & (sat < 120) & (alpha > 200)
    skin[head_bottom:] = False
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Keep the face: the largest skin region in the head area.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(skin)
    if n > 1:
        face = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        skin = (labels == face).astype(np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    return cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 6)


def grade(rgb: np.ndarray, face: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Brighten the subject naturally and lift shadows on the face, keeping texture."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0]
    subject = (alpha.astype(np.float32) / 255.0)

    # Overall lift for the person only: a gentle gamma that opens the midtones.
    lifted = 255.0 * (l / 255.0) ** 0.86
    l = l * (1 - subject) + lifted * subject

    # Face: split into low-frequency tone and high-frequency detail. Raise the tone
    # of shadowed patches (under the eyes, cheeks) toward the lit skin level, then
    # add the untouched detail back so skin texture stays sharp.
    weights = face + 1e-4
    low = cv2.GaussianBlur(l * face, (0, 0), 16) / cv2.GaussianBlur(weights, (0, 0), 16)
    target = np.percentile(l[face > 0.6], 72)
    lift = np.clip(target - low, 0, 45) * 0.5
    l = l + lift * face
    # A small overall brightening of the face so it sits with the neck.
    l = l + 6 * face

    # Put back some contrast on the person with a soft S-curve around mid-grey.
    x = np.clip(l, 0, 255) / 255.0
    curved = 255.0 * (x + 0.18 * (x - 0.5) * (1 - np.abs(2 * x - 1)))
    l = l * (1 - subject) + curved * subject

    lab[:, :, 0] = np.clip(l, 0, 255)
    lab[:, :, 1] += 1.5 * face  # a little warmth and life in the skin
    lab[:, :, 2] += 4.0 * face
    out = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Clarity: unsharp mask rather than smoothing.
    blur = cv2.GaussianBlur(out, (0, 0), 1.1)
    return cv2.addWeighted(out, 1.4, blur, -0.4, 0)


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
    face = skin_mask(rgb, alpha, face_bottom)
    rgb = clean_shirt(rgb, alpha, face_bottom)
    rgb = grade(rgb, face, alpha)

    # Head-and-shoulders crop centred on the head.
    head_rows = alpha[top : top + int(height * 0.12)] > 128
    head_cols = np.where(head_rows.any(axis=0))[0]
    cx = int((head_cols.min() + head_cols.max()) / 2)
    crop_h = int(height * 0.34)
    crop_w = int(crop_h * OUT_W / OUT_H)
    y0 = max(top - int(crop_h * 0.12), 0)
    x0 = min(max(cx - crop_w // 2, 0), photo.width - crop_w)
    box = (slice(y0, y0 + crop_h), slice(x0, x0 + crop_w))

    OUT_DIR.mkdir(exist_ok=True)
    img = composite(rgb, alpha, box, OUT_W, OUT_H)
    img.save(OUT_DIR / "headshot.jpg", quality=92, optimize=True, progressive=True)
    img.resize((500, 625), Image.LANCZOS).save(OUT_DIR / "headshot-500.jpg", quality=90, optimize=True)

    # Square profile photo (LinkedIn crops to a circle, so keep the head centred).
    side = int(crop_w * 1.4)
    sy0 = max(top - int(side * 0.13), 0)
    sx0 = min(max(cx - side // 2, 0), photo.width - side)
    square = composite(rgb, alpha, (slice(sy0, sy0 + side), slice(sx0, sx0 + side)), 1080, 1080)
    square.save(OUT_DIR / "profile-square.jpg", quality=95, optimize=True)
    print(f"wrote headshot.jpg, headshot-500.jpg, profile-square.jpg (source crop {crop_w}x{crop_h})")


def composite(rgb: np.ndarray, alpha: np.ndarray, box: tuple, w: int, h: int) -> Image.Image:
    """Place the graded subject from `box` on the studio backdrop at w x h."""
    fg = cv2.resize(rgb[box], (w, h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    # Pull the matte in a little so the old background doesn't halo the edge.
    matte = cv2.erode(alpha[box], np.ones((5, 5), np.uint8))
    a = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    a = cv2.GaussianBlur(a, (0, 0), 1.2 * w / 1000)[..., None]
    # Fine grain after upscaling keeps skin from looking airbrushed.
    grain = np.random.default_rng(3).normal(0, 2.2, (h, w, 1)).astype(np.float32)
    out = (fg + grain) * a + backdrop(w, h) * (1 - a)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: retouch_photo.py <photo>")
    main(sys.argv[1])
