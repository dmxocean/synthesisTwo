# -*- coding: utf-8 -*-
"""
Temporary script: generate synthetic noise assets for the historical document pipeline

Produces RGBA PNG crops at 300 DPI scale (calibrated for 2480x3508 A4 pages) for all
five noise categories: circles, crosses, lines, marks, stamps

Output: data/assets/manual/noise/{category}/{category}_{index:03d}.png
"""

import os
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DIR_NOISE = os.path.join(PATH_ROOT, "data", "assets", "manual", "noise")
PATH_FILE_FONT = os.path.join(PATH_ROOT, "data", "assets", "fonts", "SpecialElite-Regular.ttf")

# Config
OUTPUT_ROOT = PATH_DIR_NOISE
SEED            = 42
N_STAMPS        = 40
N_CIRCLES       = 35
N_CROSSES       = 35
N_LINES         = 40
N_MARKS         = 35

# Historical ink palette: black, dark blue, dark red, purple, sepia
INK_PALETTE = [
    (22,  22,  22),    # Black
    (18,  38, 115),    # Dark blue
    (145, 22,  22),    # Dark red
    (88,  22, 108),    # Dark purple
    (82,  52,  18),    # Sepia/brown
    (28,  68,  48),    # Dark green
]


def _ink_color():
    return random.choice(INK_PALETTE)


def _alpha_jitter(base=210, spread=40):
    return min(255, max(120, base + random.randint(-spread, spread)))


def _add_ink_noise(img, intensity=18):
    """Roughens the alpha channel to simulate real ink irregularity"""
    arr  = np.array(img).astype(np.int16)
    mask = arr[:, :, 3] > 0
    noise = np.random.randint(-intensity, intensity + 1, arr.shape[:2], dtype=np.int16)
    arr[:, :, 3] = np.clip(arr[:, :, 3] + noise * mask, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def _rotate_crop(img, angle):
    """Rotates RGBA image and crops to bounding box of non-transparent content"""
    rotated = img.rotate(angle, expand=True, resample=Image.BICUBIC)
    bbox    = rotated.getbbox()
    return rotated.crop(bbox) if bbox else rotated


def _make_canvas(w, h):
    return Image.new("RGBA", (w, h), (0, 0, 0, 0))


def _make_stamp(rng):
    """Stamps - rectangular/oval ink stamps with border and inner lines"""
    # Size: 350–850px wide, 200–500px tall (realistic for 300 DPI stamps)
    W = rng.randint(350, 850)
    H = rng.randint(200, 500)
    img  = _make_canvas(W, H)
    draw = ImageDraw.Draw(img)
    color = _ink_color()
    alpha = _alpha_jitter(200, 35)
    ink   = (*color, alpha)

    border   = rng.randint(4, 14)
    pad      = rng.randint(12, 28)
    style    = rng.choice(["rect", "rect_double", "oval", "rect_rounded"])

    # Outer border
    if style == "oval":
        draw.ellipse([0, 0, W - 1, H - 1], outline=ink, width=border)
        inner_pad = pad + border
        draw.ellipse([inner_pad, inner_pad, W - 1 - inner_pad, H - 1 - inner_pad],
                     outline=(*color, int(alpha * 0.6)), width=max(2, border - 3))
    elif style == "rect_double":
        draw.rectangle([0, 0, W - 1, H - 1], outline=ink, width=border)
        d = border + 5
        draw.rectangle([d, d, W - 1 - d, H - 1 - d],
                       outline=(*color, int(alpha * 0.7)), width=max(2, border - 2))
    else:
        radius = rng.randint(0, 20) if style == "rect_rounded" else 0
        draw.rounded_rectangle([0, 0, W - 1, H - 1], radius=radius, outline=ink, width=border)

    # Interior horizontal lines simulating stamp text content
    n_lines = rng.randint(2, 6)
    inner_y0 = pad + border + 8
    inner_y1 = H - pad - border - 8
    inner_x0 = pad + border + 10
    inner_x1 = W - pad - border - 10
    if inner_y1 > inner_y0 + 10 and inner_x1 > inner_x0 + 10:
        step = max(8, (inner_y1 - inner_y0) // max(1, n_lines))
        for i in range(n_lines):
            y     = inner_y0 + i * step + rng.randint(0, max(1, step // 3))
            x_off = rng.randint(0, 20)
            line_alpha = int(alpha * rng.uniform(0.55, 0.85))
            draw.line([(inner_x0 + x_off, y), (inner_x1 - x_off, y)],
                      fill=(*color, line_alpha), width=rng.randint(3, 8))

    img = _add_ink_noise(img)
    angle = rng.uniform(-8, 8)
    return _rotate_crop(img, angle)


def _make_circle(rng):
    """Circles - round seals / circular stamps"""
    D     = rng.randint(280, 680)
    img   = _make_canvas(D, D)
    draw  = ImageDraw.Draw(img)
    color = _ink_color()
    alpha = _alpha_jitter(195, 40)
    ink   = (*color, alpha)

    style      = rng.choice(["ring", "double_ring", "ring_cross", "solid_dot"])
    ring_w     = rng.randint(5, 18)

    if style == "solid_dot":
        r = D // 2
        draw.ellipse([r // 2, r // 2, r + r // 2, r + r // 2], fill=ink)
    elif style == "double_ring":
        draw.ellipse([0, 0, D - 1, D - 1], outline=ink, width=ring_w)
        gap = ring_w + rng.randint(6, 18)
        draw.ellipse([gap, gap, D - 1 - gap, D - 1 - gap],
                     outline=(*color, int(alpha * 0.75)), width=max(3, ring_w - 4))
    elif style == "ring_cross":
        draw.ellipse([0, 0, D - 1, D - 1], outline=ink, width=ring_w)
        cx, cy = D // 2, D // 2
        r      = D // 2 - ring_w - 6
        cross_alpha = int(alpha * 0.65)
        draw.line([(cx - r, cy), (cx + r, cy)], fill=(*color, cross_alpha),
                  width=rng.randint(3, 8))
        draw.line([(cx, cy - r), (cx, cy + r)], fill=(*color, cross_alpha),
                  width=rng.randint(3, 8))
    else:
        draw.ellipse([0, 0, D - 1, D - 1], outline=ink, width=ring_w)

    # Optional inner text-line arc simulation (horizontal lines near centre)
    if rng.random() < 0.5 and style != "solid_dot":
        cx, cy = D // 2, D // 2
        inner_r = D // 2 - ring_w - 12
        if inner_r > 20:
            n = rng.randint(1, 3)
            for i in range(n):
                y_off = rng.randint(-inner_r // 2, inner_r // 2)
                half_w = int(math.sqrt(max(0, inner_r ** 2 - y_off ** 2)))
                if half_w > 10:
                    la = int(alpha * rng.uniform(0.4, 0.7))
                    draw.line([(cx - half_w, cy + y_off), (cx + half_w, cy + y_off)],
                              fill=(*color, la), width=rng.randint(3, 7))

    img = _add_ink_noise(img)
    angle = rng.uniform(-5, 5)
    return _rotate_crop(img, angle)


def _make_cross(rng):
    """Crosses - X marks, correction crosses, plus marks"""
    S     = rng.randint(80, 340)
    img   = _make_canvas(S, S)
    draw  = ImageDraw.Draw(img)
    color = _ink_color()
    alpha = _alpha_jitter(220, 30)
    ink   = (*color, alpha)
    w     = rng.randint(4, max(5, S // 14))

    style = rng.choice(["x_mark", "x_mark", "plus", "check", "x_bold"])
    pad   = w

    if style in ("x_mark", "x_bold"):
        bold  = 2 if style == "x_bold" else 0
        draw.line([(pad, pad), (S - pad, S - pad)], fill=ink, width=w + bold)
        draw.line([(S - pad, pad), (pad, S - pad)], fill=ink, width=w + bold)
    elif style == "plus":
        cx, cy = S // 2, S // 2
        draw.line([(pad, cy), (S - pad, cy)], fill=ink, width=w)
        draw.line([(cx, pad), (cx, S - pad)], fill=ink, width=w)
    elif style == "check":
        # Simple check mark via polyline
        pts = [(pad, S // 2),
               (S // 3, S - pad),
               (S - pad, pad)]
        draw.line(pts, fill=ink, width=w)

    img = _add_ink_noise(img)
    angle = rng.uniform(-25, 25)
    return _rotate_crop(img, angle)


def _make_line(rng):
    """Lines - underlines, strikethroughs, rule lines"""
    # Width 300–1400px, height 10–55px (thick ink stroke)
    W      = rng.randint(300, 1400)
    stroke = rng.randint(5, 30)
    H      = stroke + rng.randint(10, 40)  # Canvas taller than stroke for wobble room
    img    = _make_canvas(W, H)
    draw   = ImageDraw.Draw(img)
    color  = _ink_color()
    alpha  = _alpha_jitter(210, 30)
    ink    = (*color, alpha)

    style  = rng.choice(["straight", "straight", "wavy", "double", "tapered"])
    cy     = H // 2

    if style == "wavy":
        pts   = []
        n_seg = rng.randint(4, 10)
        for i in range(n_seg + 1):
            x = int(i * W / n_seg)
            y = cy + rng.randint(-stroke, stroke)
            pts.append((x, y))
        draw.line(pts, fill=ink, width=stroke)
    elif style == "double":
        gap = rng.randint(4, 12)
        draw.line([(0, cy - gap), (W, cy - gap)], fill=ink, width=max(2, stroke // 2))
        draw.line([(0, cy + gap), (W, cy + gap)], fill=ink, width=max(2, stroke // 2))
    elif style == "tapered":
        # Thicker in the middle, tapered at ends
        n_seg = 20
        for i in range(n_seg):
            x0  = int(i * W / n_seg)
            x1  = int((i + 1) * W / n_seg)
            t   = abs(i / n_seg - 0.5) * 2  # 0 at centre, 1 at ends
            w_i = max(2, int(stroke * (1 - 0.5 * t)))
            draw.line([(x0, cy), (x1, cy)], fill=ink, width=w_i)
    else:
        # Straight with slight end squiggles
        draw.line([(0, cy), (W, cy)], fill=ink, width=stroke)
        # Ink bleed blobs at ends
        for ex in (0, W - 1):
            blob = rng.randint(1, stroke)
            draw.ellipse([ex - blob, cy - blob, ex + blob, cy + blob], fill=ink)

    img = _add_ink_noise(img, intensity=12)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    angle = rng.uniform(-3, 3)
    return _rotate_crop(img, angle)


def _make_mark(rng):
    """Marks - marginal marks, check ticks, asterisks, dots, bracket marks"""
    S     = rng.randint(60, 260)
    img   = _make_canvas(S, S)
    draw  = ImageDraw.Draw(img)
    color = _ink_color()
    alpha = _alpha_jitter(215, 35)
    ink   = (*color, alpha)
    w     = rng.randint(4, max(5, S // 12))

    style = rng.choice(["dot", "tick", "asterisk", "bracket_left", "bracket_right",
                         "dash", "arrow", "circle_dot", "squiggle"])
    cx, cy = S // 2, S // 2
    pad    = max(w, 8)

    if style == "dot":
        r = rng.randint(8, S // 3)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ink)

    elif style == "tick":
        pts = [(pad, cy), (S // 3, S - pad), (S - pad, pad)]
        draw.line(pts, fill=ink, width=w)

    elif style == "asterisk":
        r = S // 2 - pad
        for angle_deg in range(0, 180, 45):
            a = math.radians(angle_deg)
            draw.line([(cx - r * math.cos(a), cy - r * math.sin(a)),
                       (cx + r * math.cos(a), cy + r * math.sin(a))],
                      fill=ink, width=w)

    elif style == "bracket_left":
        draw.line([(cx, pad), (cx, S - pad)], fill=ink, width=w)
        draw.line([(cx, pad), (cx + S // 4, pad)], fill=ink, width=w)
        draw.line([(cx, S - pad), (cx + S // 4, S - pad)], fill=ink, width=w)

    elif style == "bracket_right":
        draw.line([(cx, pad), (cx, S - pad)], fill=ink, width=w)
        draw.line([(cx - S // 4, pad), (cx, pad)], fill=ink, width=w)
        draw.line([(cx - S // 4, S - pad), (cx, S - pad)], fill=ink, width=w)

    elif style == "dash":
        draw.line([(pad, cy), (S - pad, cy)], fill=ink, width=w)

    elif style == "arrow":
        draw.line([(pad, cy), (S - pad, cy)], fill=ink, width=w)
        tip   = S - pad
        aw    = S // 5
        draw.line([(tip, cy), (tip - aw, cy - aw)], fill=ink, width=w)
        draw.line([(tip, cy), (tip - aw, cy + aw)], fill=ink, width=w)

    elif style == "circle_dot":
        r1 = S // 2 - pad
        r2 = max(3, r1 // 3)
        draw.ellipse([cx - r1, cy - r1, cx + r1, cy + r1], outline=ink, width=w)
        draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=ink)

    elif style == "squiggle":
        n   = rng.randint(3, 7)
        pts = []
        for i in range(n + 1):
            x = pad + int(i * (S - 2 * pad) / n)
            y = cy + (1 if i % 2 == 0 else -1) * rng.randint(S // 8, S // 4)
            pts.append((x, y))
        draw.line(pts, fill=ink, width=w)

    img = _add_ink_noise(img)
    angle = rng.uniform(-30, 30)
    return _rotate_crop(img, angle)


_GENERATORS = {
    "stamps":  (_make_stamp,  N_STAMPS),
    "circles": (_make_circle, N_CIRCLES),
    "crosses": (_make_cross,  N_CROSSES),
    "lines":   (_make_line,   N_LINES),
    "marks":   (_make_mark,   N_MARKS),
}

PAGE_W, PAGE_H = 2480, 3508
PAPER_COLOR    = (245, 242, 235)

# How many assets from each category to scatter on the preview page
_PREVIEW_COUNTS = {
    "stamps":  6,
    "circles": 6,
    "crosses": 8,
    "lines":   7,
    "marks":   10,
}

# Label colours per category (dark, legible on paper)
_LABEL_COLORS = {
    "stamps":  (160,  20,  20),
    "circles": ( 20,  20, 160),
    "crosses": ( 20, 120,  20),
    "lines":   (120,  20, 120),
    "marks":   (140,  80,   0),
}


def _label_asset(asset, text, color):
    """Burns a small category label into the bottom-left of an RGBA asset copy"""
    from PIL import ImageFont
    labeled = asset.copy()
    draw    = ImageDraw.Draw(labeled)
    font_size = max(18, min(40, asset.width // 8))
    try:
        font = ImageFont.truetype(PATH_FILE_FONT, font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, max(0, asset.height - font_size - 4)), text, fill=(*color, 220), font=font)
    return labeled


def _make_preview(rng):
    """
    Assembles one A4 preview page with assets from every category scattered
    across it to allow visual size verification
    """
    page  = Image.new("RGB", (PAGE_W, PAGE_H), PAPER_COLOR)
    # Faint grid lines to give a sense of scale (every 300px = 1 inch at 300 DPI)
    grid  = ImageDraw.Draw(page)
    grid_color = (220, 216, 208)
    for x in range(0, PAGE_W, 300):
        grid.line([(x, 0), (x, PAGE_H)], fill=grid_color, width=1)
    for y in range(0, PAGE_H, 300):
        grid.line([(0, y), (PAGE_W, y)], fill=grid_color, width=1)

    # Section dividers and category banners
    section_h = PAGE_H // len(_PREVIEW_COUNTS)
    banner    = ImageDraw.Draw(page)
    try:
        from PIL import ImageFont
        font_banner = ImageFont.truetype(PATH_FILE_FONT, 60)
    except Exception:
        font_banner = ImageFont.load_default()

    for sec_idx, category in enumerate(_PREVIEW_COUNTS):
        y_top = sec_idx * section_h
        # Faint horizontal band separator
        banner.rectangle([0, y_top, PAGE_W, y_top + 3], fill=(200, 195, 185))
        # Category label in left margin
        label_color = _LABEL_COLORS[category]
        banner.text((30, y_top + 14), category.upper(), fill=(*label_color, 200),
                    font=font_banner)

    # Place assets
    for sec_idx, (category, n) in enumerate(_PREVIEW_COUNTS.items()):
        y_top    = sec_idx * section_h + 90   # leave room for banner
        y_bottom = (sec_idx + 1) * section_h - 20
        gen_fn   = _GENERATORS[category][0]
        label_c  = _LABEL_COLORS[category]

        placed   = 0
        attempts = 0
        while placed < n and attempts < n * 8:
            attempts += 1
            asset = gen_fn(rng)
            asset = _label_asset(asset, category[:3].upper(), label_c)

            # Random position within the section band, away from margins
            margin = 60
            max_x  = max(margin, PAGE_W  - asset.width  - margin)
            max_y  = max(y_top,  y_bottom - asset.height)
            if max_x <= margin or max_y <= y_top:
                continue
            x = rng.randint(margin, max_x)
            y = rng.randint(y_top,  max_y)

            page.paste(asset, (x, y), asset)
            placed += 1

    return page


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    total = 0
    for category, (fn, count) in _GENERATORS.items():
        out_dir = os.path.join(OUTPUT_ROOT, category)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[*] Generating {count} {category}...")
        for i in range(count):
            img  = fn(rng)
            path = os.path.join(out_dir, f"{category}_{i:03d}.png")
            img.save(path)
            total += 1
        print(f"    -> {count} saved to {os.path.relpath(out_dir)}")

    print(f"\n[*] Done - {total} assets written to {os.path.relpath(OUTPUT_ROOT)}")

    # Preview page
    print("\n[*] Generating preview page...")
    preview_dir  = os.path.join(PATH_ROOT, "data", "synthetic", "verify")
    os.makedirs(preview_dir, exist_ok=True)
    preview_path = os.path.join(preview_dir, "noise_preview.png")
    preview      = _make_preview(rng)
    preview.save(preview_path)
    print(f"[*] Preview saved to {os.path.relpath(preview_path)}")


if __name__ == "__main__":
    main()
