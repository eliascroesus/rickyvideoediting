"""Iman-Gadzhi-style edit of 'This is how much youtube paid me for 144m views'  (v2).

Design system (from the Iman Gadzhi breakdown):
  colour  60/30/10 : near-black navy base, slate/grey secondary, warm orange accent
  fonts            : Inter Display (sans, regular words) + Instrument Serif Italic (special words, orange + soft glow)
  shapes           : rounded rectangles (30-70 px corners), gradient fill, light-sweep rim, faded icons
  animation        : per-character text write-on (rise + blur->sharp + yellow->final colour, posterised to 15 fps),
                     text placed behind the speaker (person matte), blurred/darkened background "adjustment layer",
                     smooth camera that never stops, fast-in/slow-out UI pop-ups, trim-path underlines,
                     zoom-through transitions, spotlight (dim + blur everything but the focus area)

usage:
  python tools/edit_youtube_paid_me.py RAW WORK video  T0 T1 OUT.mp4   (render output-time range)
  python tools/edit_youtube_paid_me.py RAW WORK audio  OUT.wav
WORK must hold masks_*.npz / stab_*.npy from tools/precompute_paid_me.py and the serif fonts in WORK/fonts.
"""
import functools
import math
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(2)
RAW, WORK, MODE = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
REPO = Path(__file__).resolve().parent.parent
SW, SH = 2560, 1440
OW, OH = 2560, 1440
K = OW / 1920  # design units are 1080p
FPS = 30
SR = 48000
END = 609.0

# ------------------------------------------------------------------ palette (BGR)
BASE = (18, 13, 11)          # 60 %  near-black navy
SECOND = (168, 156, 150)     # 30 %  slate grey text
ACCENT = (48, 172, 250)      # 10 %  warm gold accent
WHITE = (242, 242, 242)
INTRO = (110, 220, 255)       # characters start yellow and settle to their colour

F_SANS = "/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf"
F_SANS_M = "/usr/share/fonts/opentype/inter/InterDisplay-Medium.otf"
F_SANS_SB = "/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf"
F_SERIF = str(WORK / "fonts/EBGaramond-BoldItalic.ttf")


@functools.lru_cache(None)
def font(path, px):
    return ImageFont.truetype(path, int(px))


# ------------------------------------------------------------------ cuts / time maps
CUTS = [
    (39.16, 39.63), (54.77, 55.20), (66.36, 66.65), (109.88, 111.42),
    (140.20, 140.47), (187.71, 188.07), (205.32, 206.97), (259.33, 259.73),
    (358.34, 358.83), (367.64, 367.93), (397.93, 398.88), (496.56, 497.72),
]


def to_out(t):
    return t - sum(min(b, max(a, t)) - a for a, b in CUTS)


def to_src(tau):
    t = tau
    for a, b in CUTS:
        if t >= a:
            t += b - a
    return t


TOTAL = to_out(END)


# ------------------------------------------------------------------ easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def e_out(x):  # cubic out
    x = clamp(x)
    return 1 - (1 - x) ** 3


def e_expo(x):  # fast entry, long settle
    x = clamp(x)
    return 1 if x >= 1 else 1 - 2 ** (-10 * x)


def e_io(x):  # quintic in-out: velocity peaks mid-way
    x = clamp(x)
    return 16 * x ** 5 if x < 0.5 else 1 - (-2 * x + 2) ** 5 / 2


def e_sm(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ramp(t, a, b, fin=0.35, fout=0.35):
    if t < a or t > b:
        return 0.0
    return min(e_sm((t - a) / fin) if fin else 1, e_sm((b - t) / fout) if fout else 1)


# ------------------------------------------------------------------ image helpers
def soft_blur(img, sigma):
    h, w = img.shape[:2]
    s = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), max(0.5, sigma / 4))
    return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)


def blend_rgba(dst, spr, x, y, alpha=1.0):
    """alpha-composite uint8 BGRA sprite onto uint8 BGR dst at (x, y)."""
    if alpha <= 0.003:
        return
    h, w = spr.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[:, :, 3:4].astype(np.float32) * (alpha / 255.0)
    r = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (r + (s[:, :, :3].astype(np.float32) - r) * a).astype(np.uint8)


def add_rgba(dst, spr, x, y, alpha=1.0):
    """additive (screen-ish) composite for glows."""
    if alpha <= 0.003:
        return
    h, w = spr.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    add = s[:, :, :3] * (s[:, :, 3:4] / 255.0) * alpha
    dst[y0:y1, x0:x1] = np.clip(dst[y0:y1, x0:x1].astype(np.float32) + add, 0, 255).astype(np.uint8)


def pil_to_bgra(im):
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2BGRA)


def rgb(c):
    return (c[2], c[1], c[0])


# ------------------------------------------------------------------ grade + vignette (footage)
_x = np.arange(256) / 255.0
_s = np.clip(0.5 + (_x - 0.5) * 1.10, 0, 1)
_s = _s * 0.94 + 0.012
LUT = np.stack([np.clip(_s * 0.985, 0, 1), np.clip(_s, 0, 1), np.clip(_s * 1.03 + 0.005, 0, 1)], 1)
LUT = (LUT * 255).astype(np.uint8).reshape(256, 1, 3)
_yy, _xx = np.mgrid[0:OH, 0:OW]
_r = np.sqrt(((_xx - OW / 2) / (OW / 2)) ** 2 + ((_yy - OH / 2) / (OH / 2)) ** 2) / math.sqrt(2)
VIG = np.repeat((np.clip(1 - 0.42 * _r ** 2.3, 0, 1) * 255).astype(np.uint8)[:, :, None], 3, 2)
del _yy, _xx, _r


def grade(fr):
    fr = cv2.LUT(fr, LUT)
    return cv2.multiply(fr, VIG, scale=1 / 255)


# ------------------------------------------------------------------ grid background (UI scenes)
def _make_page():
    """black page with a warm orange gradient glow and a grid whose opacity fades out towards the edges."""
    m = 240
    H, W = OH + m, OW + m
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.empty((H, W, 3), np.float32)
    img[:] = (7, 6, 6)
    r1 = np.sqrt(((xx - W * 0.5) / (W * 0.62)) ** 2 + ((yy - H * 1.02) / (H * 0.80)) ** 2)
    g1 = np.clip(1 - r1, 0, 1) ** 1.7
    r2 = np.sqrt(((xx - W * 0.10) / (W * 0.42)) ** 2 + ((yy + H * 0.05) / (H * 0.55)) ** 2)
    g2 = np.clip(1 - r2, 0, 1) ** 2.0
    orange = np.array((18, 88, 220), np.float32)
    img += orange[None, None] * (g1 * 0.62 + g2 * 0.20)[:, :, None]
    step = int(90 * K)
    lines = np.zeros((H, W), np.float32)
    lines[:, ::step] = 1
    lines[::step, :] = 1
    lines = cv2.GaussianBlur(lines, (0, 0), 0.7)
    rc = np.sqrt(((xx - W / 2) / (W * 0.52)) ** 2 + ((yy - H * 0.52) / (H * 0.58)) ** 2)
    fade = np.clip(1 - rc, 0, 1) ** 1.2
    img += lines[:, :, None] * (55 * fade + 45 * g1)[:, :, None]
    return np.clip(img, 0, 255).astype(np.uint8), step


GRID, GSTEP = _make_page()


def grid_bg(t):
    ox = int((t * 7) % GSTEP)
    return GRID[120:120 + OH, 120 - ox:120 - ox + OW].copy()


# ------------------------------------------------------------------ rounded shapes / cards
def rr_mask(w, h, r, ss=1):
    im = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, fill=255)
    if ss > 1:
        im = im.resize((w, h), Image.LANCZOS)
    return np.array(im)


def persp(w, h, s=1.0, rx=0.0, ry=0.0, rz=0.0, f=None):
    """homography for a w x h layer scaled by s and rotated in 3D about its centre.
    returns (H, out_w, out_h, cx_off, cy_off): warp into an out_w x out_h canvas whose centre is the layer centre."""
    f = f or 2.4 * max(w, h)
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], np.float64)
    ax, ay, az = math.radians(rx), math.radians(ry), math.radians(rz)
    out = []
    for x, y in pts:
        x, y, z = x * s, y * s, 0.0
        y, z = y * math.cos(ax) - z * math.sin(ax), y * math.sin(ax) + z * math.cos(ax)
        x, z = x * math.cos(ay) + z * math.sin(ay), -x * math.sin(ay) + z * math.cos(ay)
        k = f / (f + z)
        x, y = x * k, y * k
        x, y = x * math.cos(az) - y * math.sin(az), x * math.sin(az) + y * math.cos(az)
        out.append((x, y))
    out = np.array(out)
    ow, oh = int(np.abs(out[:, 0]).max() * 2) + 4, int(np.abs(out[:, 1]).max() * 2) + 4
    dst = (out + [ow / 2, oh / 2]).astype(np.float32)
    src = (pts + [w / 2, h / 2]).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst), ow, oh


def draw_text_pil(im, xy, text, path, px, color, anchor="la", glow=False, tracking=0):
    d = ImageDraw.Draw(im)
    f = font(path, px)
    if glow:  # deep glow: tight + wide halo
        gl = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(gl).text(xy, text, font=f, fill=color[:3] + (255,), anchor=anchor)
        arr = np.array(gl).astype(np.float32)
        a = arr[:, :, 3]
        halo = cv2.GaussianBlur(a, (0, 0), 5 * K) * 0.55 + cv2.GaussianBlur(a, (0, 0), 18 * K) * 0.45
        arr[:, :, 3] = np.clip(halo, 0, 255)
        im.alpha_composite(Image.fromarray(arr.astype(np.uint8)))
    d.text(xy, text, font=f, fill=color, anchor=anchor)


@functools.lru_cache(None)
def icon_alpha(kind, size):
    """clean white line icons (4x supersampled)."""
    S4 = size * 4
    im = Image.new("L", (S4, S4), 0)
    d = ImageDraw.Draw(im)
    lw = int(S4 * 0.05)
    c = S4 / 2
    if kind == "play":
        d.rounded_rectangle((S4 * .08, S4 * .22, S4 * .92, S4 * .78), int(S4 * .14), outline=255, width=lw)
        d.polygon([(S4 * .42, S4 * .37), (S4 * .42, S4 * .63), (S4 * .64, S4 * .5)], fill=255)
    elif kind == "dollar":
        d.ellipse((S4 * .08, S4 * .08, S4 * .92, S4 * .92), outline=255, width=lw)
        d.text((c, c), "$", font=font(F_SERIF, S4 * .56), fill=255, anchor="mm")
    elif kind == "tv":
        d.rounded_rectangle((S4 * .08, S4 * .18, S4 * .92, S4 * .70), int(S4 * .07), outline=255, width=lw)
        d.line([(S4 * .34, S4 * .86), (S4 * .66, S4 * .86)], fill=255, width=lw)
        d.line([(c, S4 * .70), (c, S4 * .86)], fill=255, width=lw)
    elif kind == "ad":
        d.rounded_rectangle((S4 * .08, S4 * .24, S4 * .92, S4 * .76), int(S4 * .12), outline=255, width=lw)
        d.text((c, c), "AD", font=font(F_SANS, S4 * .28), fill=255, anchor="mm")
    elif kind == "eye":
        d.chord((S4 * .06, S4 * .2, S4 * .94, S4 * .95), 200, 340, outline=255, width=lw)
        d.chord((S4 * .06, S4 * .05, S4 * .94, S4 * .8), 20, 160, outline=255, width=lw)
        d.ellipse((S4 * .36, S4 * .36, S4 * .64, S4 * .64), outline=255, width=lw)
    elif kind == "film":
        d.rounded_rectangle((S4 * .2, S4 * .06, S4 * .8, S4 * .94), int(S4 * .1), outline=255, width=lw)
        d.polygon([(S4 * .44, S4 * .38), (S4 * .44, S4 * .62), (S4 * .62, S4 * .5)], fill=255)
    elif kind == "chart":
        for i, hh in enumerate((.35, .55, .8)):
            x0 = S4 * (.14 + i * .26)
            d.rounded_rectangle((x0, S4 * (.9 - hh), x0 + S4 * .18, S4 * .9), int(S4 * .04), outline=255, width=lw)
    return cv2.resize(np.array(im), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


@functools.lru_cache(None)
def ui_card(w, h, main="", sub="", style="sans", icon=None, accent=False, num="", check=False, main_px=0,
            align="c"):
    """Iman-style card: rounded rect, orange -> black gradient, faded grid inside, glowing orange edge (light sweep)
    with a deep outer glow, faded track-matted icon. returns (BGRA uint8, glow margin)."""
    w, h = int(w * K), int(h * K)
    r = int(min(56 * K, h * 0.28))
    g = int(90 * K)
    W, H = w + 2 * g, h + 2 * g
    m = rr_mask(w, h, r, 2).astype(np.float32) / 255
    full = np.zeros((H, W), np.float32)
    full[g:g + h, g:g + w] = m
    YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = (XX - g) / w, (YY - g) / h
    d = np.clip(u * 0.55 + v * 0.45, 0, 1)
    warm = np.clip(1 - d * 1.25, 0, 1) ** 1.5                     # 1 at the top-left corner, 0 past the middle
    orange = np.array((232, 104, 24), np.float32)
    dark = np.array((9, 8, 8), np.float32)
    body = dark + (orange - dark) * (warm * (0.62 if accent else 0.42))[:, :, None]
    # faded grid inside the card
    step = int(30 * K)
    gl = np.zeros((H, W), np.float32)
    gl[:, g % step::step] = 1
    gl[g % step::step, :] = 1
    gl = cv2.GaussianBlur(gl, (0, 0), 0.6)
    body += gl[:, :, None] * (8 + 46 * warm)[:, :, None]
    # inner top highlight
    hl = np.exp(-((YY - g - 3 * K) / (3 * K)) ** 2) * np.clip(1 - u * 1.2, 0, 1)
    body += hl[:, :, None] * 40
    # rim with light sweep + deep glow
    rim = np.clip(full - cv2.erode(full, np.ones((3, 3), np.uint8), iterations=max(1, int(1.4 * K))), 0, 1)
    sweep = np.clip(1 - d * 1.1, 0, 1) ** 1.3
    rim_i = rim * (0.16 + 0.84 * sweep)
    rimcol = np.array((255, 156, 64), np.float32)
    glow_t = np.clip(cv2.GaussianBlur(rim_i, (0, 0), 7 * K) * 2.4, 0, 1)
    glow_w = np.clip(cv2.GaussianBlur(rim_i, (0, 0), 30 * K) * 3.0, 0, 1)
    outside = 1 - full
    shadow = np.roll(cv2.GaussianBlur(full, (0, 0), 28 * K), int(20 * K), 0) * 0.75 * outside
    rgbf = np.zeros((H, W, 3), np.float32)
    a = np.zeros((H, W), np.float32)

    def over(col, al):
        nonlocal rgbf, a
        al3 = al[:, :, None]
        rgbf = rgbf * (1 - al3) + col * al3
        a = a + al * (1 - a)

    over(np.zeros(3, np.float32), shadow)
    over(rimcol, glow_w * outside * (0.42 if accent else 0.30))
    over(rimcol, glow_t * outside * (0.75 if accent else 0.55))
    over(body, full * 0.96)
    if icon:
        size = int(h * 1.05)
        ia = icon_alpha(icon, size)
        lay = np.zeros((H, W), np.float32)
        x0, y0 = int(g + w - size * 0.72), int(g + (h - size) / 2)
        xs0, xs1 = max(0, x0), min(W, x0 + size)
        lay[y0:y0 + size, xs0:xs1] = ia[:, xs0 - x0:xs1 - x0]
        wipe = np.clip((XX - x0) / size * 1.4 - 0.1, 0, 1) * 0.16
        over(np.array((255, 214, 170), np.float32), lay * wipe * full)
    over(np.array((255, 196, 140), np.float32), rim_i)
    im = Image.fromarray(np.dstack([rgbf, a * 255]).clip(0, 255).astype(np.uint8), "RGBA")
    gold = (250, 172, 48, 255)
    x = g + 54 * K
    if check:
        dd = ImageDraw.Draw(im)
        rr = h * 0.17
        cy = g + h / 2
        dd.ellipse((x, cy - rr, x + 2 * rr, cy + rr), fill=gold)
        dd.line([(x + rr * .55, cy + rr * .02), (x + rr * .9, cy + rr * .38), (x + rr * 1.5, cy - rr * .4)],
                fill=(14, 14, 16, 255), width=int(5 * K), joint="curve")
        x += 2 * rr + 30 * K
    if num:
        draw_text_pil(im, (x, g + h / 2), num, F_SERIF, h * 0.42, gold, "lm", glow=True)
        x += font(F_SERIF, h * 0.42).getlength(num) + 30 * K
    if main:
        serif = style == "serif"
        px = main_px * K if main_px else (h * (0.40 if serif else 0.22))
        path = F_SERIF if serif else F_SANS
        col = gold if (serif and accent) else (246, 244, 242, 255)
        cy = g + h * (0.42 if sub else 0.5)
        if align == "c":
            draw_text_pil(im, (g + w / 2, cy), main, path, px, col, "mm", glow=serif and accent)
        else:
            draw_text_pil(im, (x, cy), main, path, px, col, "lm", glow=serif and accent)
        if sub:
            draw_text_pil(im, (g + w / 2 if align == "c" else x, g + h * 0.75), sub, F_SANS_M, h * 0.095,
                          (214, 206, 198, 255), "mm" if align == "c" else "lm")
    return pil_to_bgra(im), g


def pop_in(dst, spr_g, cx, cy, t, t0, t_out=None, dur=0.8, rise=90):
    """UI pop-up: scale + rise + 3D tilt settling flat; fast entry, long settle; tilt away on exit."""
    spr, g = spr_g
    if t < t0:
        return
    p = e_expo((t - t0) / dur)
    a = clamp((t - t0) / 0.22)
    q = 0.0
    if t_out is not None:
        q = e_sm(clamp((t - (t_out - 0.35)) / 0.35))
        a *= 1 - q
    if a <= 0.003:
        return
    s = 0.74 + 0.26 * p - 0.06 * q
    rx = 32 * (1 - p) - 14 * q
    h, w = spr.shape[:2]
    if abs(rx) < 0.05 and abs(s - 1) < 0.002:
        im = spr
    else:
        Hm, ow, oh = persp(w, h, s, rx=rx)
        im = cv2.warpPerspective(spr, Hm, (ow, oh), flags=cv2.INTER_LINEAR)
    h2, w2 = im.shape[:2]
    blend_rgba(dst, im, cx - w2 / 2, cy - h2 / 2 + ((1 - p) * rise - q * 30) * K, a)


# ------------------------------------------------------------------ per-character text engine
BLUR_LEVELS = [0, 1.5, 3, 5, 7.5, 10]


class TextSeq:
    """lines: [[(word, style, t_start), ...], ...]   style: 'sans' | 'serif' | 'grey'
    sizes (sans_px, serif_px) in 1080 units.  align: 'c' centre at x | 'r' right edge at x | 'l' left edge at x.
    ry / rz: 3D tilt (degrees) of the whole text plane, like a 3D text layer under a camera."""

    def __init__(self, lines, t_out, centre=(OW / 2, OH / 2), sizes=(84, 150), behind=False,
                 gap=1.0, align="c", stagger=0.035, dur=0.5, ry=0.0, rz=0.0, shadow=0.55):
        self.t_out, self.behind, self.stagger, self.dur = t_out, behind, stagger, dur
        self.ry, self.rz, self.shadow = ry, rz, shadow
        self.chars = []
        sans_px, serif_px = sizes[0] * K, sizes[1] * K
        heights, layouts = [], []
        for line in lines:
            x, items, lh = 0.0, [], 0
            for wi, (word, style, t0) in enumerate(line):
                path, px = ((F_SERIF, serif_px) if style == "serif" else
                            (F_SANS_M, sans_px) if style == "grey" else (F_SANS, sans_px))
                f = font(path, px)
                items.append((word, style, t0, f, px, x))
                x += f.getlength(word) + (f.getlength(" ") if wi < len(line) - 1 else 0)
                lh = max(lh, px)
            layouts.append((items, x))
            heights.append(lh)
        y = centre[1] - sum(heights) * gap / 2
        self.box = [1e9, 1e9, -1e9, -1e9]
        for (items, width), lh in zip(layouts, heights):
            base_y = y + lh * 0.80
            x0 = {"c": centre[0] - width / 2, "r": centre[0] - width, "l": centre[0]}[align]
            for word, style, t0, f, px, wx in items:
                for i, ch in enumerate(word):
                    if ch == " ":
                        continue
                    self._add_char(ch, f, px, x0 + wx + f.getlength(word[:i]), base_y, style, t0 + i * stagger)
            y += lh * gap
        mg = 130 * K
        self.box = [int(max(0, self.box[0] - mg)), int(max(0, self.box[1] - mg)),
                    int(min(OW, self.box[2] + mg)), int(min(OH, self.box[3] + mg))]
        self.H = None
        if self.ry or self.rz:
            x0, y0, x1, y1 = self.box
            w, h = x1 - x0, y1 - y0
            Hm, ow, oh = persp(w, h, 1.0, ry=self.ry, rz=self.rz, f=3.0 * w)
            T = np.array([[1, 0, -(ow - w) / 2], [0, 1, -(oh - h) / 2], [0, 0, 1]])
            self.H = (T @ Hm).astype(np.float64)

    def _add_char(self, ch, f, px, x, base_y, style, t0):
        pad = int(max(px * 0.5, 60 * K))
        adv = f.getlength(ch)
        w, h = int(adv + 2 * pad), int(px * 1.25 + 2 * pad)
        im = Image.new("L", (w, h), 0)
        ImageDraw.Draw(im).text((pad, pad + px * 0.95), ch, font=f, fill=255, anchor="ls")
        a = np.array(im).astype(np.float32) / 255
        levels = [a if s == 0 else cv2.GaussianBlur(a, (0, 0), s * K) for s in BLUR_LEVELS]
        glow = None
        if style == "serif":  # deep glow
            glow = cv2.GaussianBlur(a, (0, 0), 4 * K) * 0.7 + cv2.GaussianBlur(a, (0, 0), 16 * K) * 0.65
        colour = {"serif": ACCENT, "sans": WHITE, "grey": SECOND}[style]
        ox, oy = x - pad, base_y - pad - px * 0.95
        self.chars.append(dict(levels=levels, glow=glow, col=np.array(colour, np.float32), x=ox, y=oy,
                               t0=t0, px=px))
        self.box[0] = min(self.box[0], ox + pad * 0.6)
        self.box[1] = min(self.box[1], oy + pad * 0.6)
        self.box[2] = max(self.box[2], ox + w - pad * 0.6)
        self.box[3] = max(self.box[3], oy + h - pad * 0.6)

    def active(self, t):
        return self.chars and self.chars[0]["t0"] - 0.05 <= t <= self.t_out + 0.05

    def draw(self, dst, t, person=None):
        if not self.active(t):
            return
        tq = math.floor(t * 15) / 15  # posterize time: 15 fps write-on
        x0, y0, x1, y1 = self.box
        if x1 <= x0 or y1 <= y0:
            return
        bw, bh = x1 - x0, y1 - y0
        C = np.zeros((bh, bw, 3), np.float32)
        A = np.zeros((bh, bw), np.float32)
        G = np.zeros((bh, bw, 3), np.float32)
        q_out = clamp((t - (self.t_out - 0.35)) / 0.35)
        any_ = False
        for c in self.chars:
            p = (tq - c["t0"]) / self.dur
            if p <= 0:
                continue
            e = e_out(p)
            alpha = clamp(p * 1.8) * (1 - q_out)
            if alpha <= 0.002:
                continue
            any_ = True
            lvl = int(round((1 - e) * 5 + q_out * 4))
            m = c["levels"][min(5, lvl)]
            dy = (1 - e) * 0.45 * c["px"] - e_sm(q_out) * 0.18 * c["px"]
            col = c["col"] + (np.array(INTRO, np.float32) - c["col"]) * (1 - e_sm(clamp(p * 1.3)))
            cx, cy = int(round(c["x"])) - x0, int(round(c["y"] + dy)) - y0
            h, w = m.shape
            a0, b0 = max(cx, 0), max(cy, 0)
            a1, b1 = min(cx + w, bw), min(cy + h, bh)
            if a0 >= a1 or b0 >= b1:
                continue
            mm = m[b0 - cy:b1 - cy, a0 - cx:a1 - cx] * alpha
            C[b0:b1, a0:a1] = C[b0:b1, a0:a1] * (1 - mm[:, :, None]) + col * mm[:, :, None]
            A[b0:b1, a0:a1] = A[b0:b1, a0:a1] + mm * (1 - A[b0:b1, a0:a1])
            if c["glow"] is not None:
                gg = c["glow"][b0 - cy:b1 - cy, a0 - cx:a1 - cx][:, :, None] * (alpha * e)
                G[b0:b1, a0:a1] += gg * col * 0.42
        if not any_:
            return
        if self.H is not None:
            C = cv2.warpPerspective(C, self.H, (bw, bh), flags=cv2.INTER_LINEAR)
            A = cv2.warpPerspective(A, self.H, (bw, bh), flags=cv2.INTER_LINEAR)
            G = cv2.warpPerspective(G, self.H, (bw, bh), flags=cv2.INTER_LINEAR)
        orig = dst[y0:y1, x0:x1].copy() if (self.behind and person is not None) else None
        roi = dst[y0:y1, x0:x1].astype(np.float32)
        if self.shadow:
            sh = cv2.GaussianBlur(A, (0, 0), 7 * K) * self.shadow
            roi *= (1 - sh)[:, :, None]
        roi = roi * (1 - A[:, :, None]) + C + G
        dst[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
        if orig is not None:
            pm = person[y0:y1, x0:x1][:, :, None]
            dst[y0:y1, x0:x1] = (dst[y0:y1, x0:x1] * (1 - pm) + orig * pm).astype(np.uint8)


class Stroke:
    """trim-path line that draws on (pts in output px, or in source px when cam=True)."""

    def __init__(self, pts, t0, t_out, dur=0.55, colour=ACCENT, width=7, cam=False):
        self.pts = np.array(pts, np.float32)
        self.t0, self.t_out, self.dur, self.col, self.w, self.cam = t0, t_out, dur, colour, width, cam
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.cum = np.concatenate([[0], np.cumsum(seg)])

    def draw(self, dst, t, M=None):
        if t < self.t0 or t > self.t_out:
            return
        p = e_io((t - self.t0) / self.dur) if True else 1
        a = 1 - clamp((t - (self.t_out - 0.3)) / 0.3)
        L = self.cum[-1] * p
        pts = self.pts
        if self.cam and M is not None:
            pts = pts @ M[:, :2].T + M[:, 2]
        keep = [pts[0]]
        for i in range(1, len(pts)):
            if self.cum[i] <= L:
                keep.append(pts[i])
            else:
                f = (L - self.cum[i - 1]) / max(1e-6, self.cum[i] - self.cum[i - 1])
                keep.append(pts[i - 1] + (pts[i] - pts[i - 1]) * f)
                break
        poly = np.array(keep, np.float32)
        if len(poly) < 2:
            return
        x0, y0 = np.floor(poly.min(0) - 60 * K).astype(int)
        x1, y1 = np.ceil(poly.max(0) + 60 * K).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(OW, x1), min(OH, y1)
        if x1 <= x0 or y1 <= y0:
            return
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.polylines(m, [((poly - (x0, y0)) * 4).astype(np.int32)], False, 255, int(self.w * K), cv2.LINE_AA, 2)
        mf = m.astype(np.float32) / 255
        gl = cv2.GaussianBlur(mf, (0, 0), 9 * K)
        roi = dst[y0:y1, x0:x1].astype(np.float32)
        col = np.array(self.col, np.float32)
        roi += gl[:, :, None] * col * 0.6 * a
        roi += (col - roi) * (mf[:, :, None] * a)
        dst[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


def curve(x0, y0, x1, y1, bend=0.0, n=40):
    """slightly curved hand-drawn style line (output px)."""
    t = np.linspace(0, 1, n)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + bend
    xs = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * mx + t ** 2 * x1
    ys = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * my + t ** 2 * y1
    return list(zip(xs, ys))


# ------------------------------------------------------------------ source frames
class Source:
    def __init__(self, t_start):
        self.i0 = max(0, int(t_start * FPS))
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{self.i0 / FPS:.4f}", "-i", RAW, "-an",
                                   "-vf", f"fps={FPS}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                                  stdout=subprocess.PIPE, bufsize=SW * SH * 3 * 4)
        self.next = self.i0
        self.cache = {}
        self.last = None

    def get(self, t):
        i = max(self.i0, int(round(t * FPS)))
        while self.next <= i:
            buf = self.p.stdout.read(SW * SH * 3)
            if len(buf) < SW * SH * 3:
                break
            self.cache[self.next] = np.frombuffer(buf, np.uint8).reshape(SH, SW, 3)
            self.next += 1
            for k in [k for k in self.cache if k < self.next - 45]:
                del self.cache[k]
        if i in self.cache:
            self.last = self.cache[i]
        return self.last.copy()


# ------------------------------------------------------------------ precomputed data
STAB = {a: np.load(WORK / f"stab_{a:.2f}.npy") for a in (39.27, 167.80)}
MASKS = {}


def stab_at(t):
    for a, arr in STAB.items():
        i = int(round((t - a) * FPS))
        if 0 <= i < len(arr):
            return arr[i]
    return (0.0, 0.0, 0.0)


def mask_at(t):
    for a in (111.13, 592.03):
        if a not in MASKS and (a <= t < a + 60):
            MASKS[a] = np.load(WORK / f"masks_{a:.2f}.npz")["m"]
        if a in MASKS:
            arr = MASKS[a]
            i = int(round((t - a) * FPS))
            if 0 <= i < len(arr):
                return arr[i]
    return None


# ------------------------------------------------------------------ camera
def cam_matrix(z, cx, cy, stab=None, out_w=OW, out_h=OH):
    s = z * out_w / SW
    R = np.eye(2)
    d = np.zeros(2)
    if stab is not None:
        dx, dy, da = stab
        c, s_ = math.cos(da), math.sin(da)
        R = np.array([[c, -s_], [s_, c]])
        C0 = np.array([SW / 2, SH / 2])
        d = C0 - R @ C0 + np.array([dx, dy])
    A = s * R
    b = s * (d - np.array([cx, cy])) + np.array([out_w / 2, out_h / 2])
    return np.hstack([A, b[:, None]]).astype(np.float32)


def clamp_cam(z, cx, cy):
    hw, hh = SW / (2 * z), SH / (2 * z)
    return z, min(max(cx, hw), SW - hw), min(max(cy, hh), SH - hh)


class Track:
    """keyframes (t, z, cx, cy); quintic in-out between keys -> movement never fully stops."""

    def __init__(self, keys):
        self.k = sorted(keys)

    def __call__(self, t):
        k = self.k
        if t <= k[0][0]:
            return clamp_cam(*k[0][1:])
        for (t0, *a), (t1, *b) in zip(k, k[1:]):
            if t <= t1:
                e = e_io((t - t0) / (t1 - t0))
                z = a[0] * (b[0] / a[0]) ** e
                return clamp_cam(z, a[1] + (b[1] - a[1]) * e, a[2] + (b[2] - a[2]) * e)
        return clamp_cam(*k[-1][1:])


# ------------------------------------------------------------------ spotlight
def spotlight(fr, rect_src, M, amount, dim=0.34, blur=9):
    if amount <= 0.01:
        return fr
    x0, y0, x1, y1 = rect_src
    pts = np.array([[x0, y0], [x1, y1]], np.float32) @ M[:, :2].T + M[:, 2]
    (a, b), (c, d) = pts
    q = 4
    m = np.zeros((OH // q, OW // q), np.uint8)
    rr = int(26 * K / q)
    cv2.rectangle(m, (int(a / q) + rr, int(b / q)), (int(c / q) - rr, int(d / q)), 255, -1)
    cv2.rectangle(m, (int(a / q), int(b / q) + rr), (int(c / q), int(d / q) - rr), 255, -1)
    for px_, py_ in ((a / q + rr, b / q + rr), (c / q - rr, b / q + rr), (a / q + rr, d / q - rr), (c / q - rr, d / q - rr)):
        cv2.circle(m, (int(px_), int(py_)), rr, 255, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 7)
    m = cv2.resize(m, (OW, OH), interpolation=cv2.INTER_LINEAR)[:, :, None]
    bg = (soft_blur(fr, blur * K) * dim).astype(np.float32)
    k = 1 - amount * (1 - m)
    return (fr * k + bg * (1 - k)).astype(np.uint8)


def dim_all(fr, amount, dim=0.42, blur=10):
    if amount <= 0.01:
        return fr
    bg = soft_blur(fr, blur * K).astype(np.float32) * dim
    return (fr * (1 - amount) + bg * amount).astype(np.uint8)


# ------------------------------------------------------------------ revenue blur (source space)
REVEAL = 26.0
BLUR_BOXES = [(1395, 555, 1745, 735), (1590, 780, 1700, 1000)]


def revenue_blur(fr, t):
    if not (t < REVEAL + 0.2):
        return
    k = 1.0 if t < REVEAL else 1 - (t - REVEAL) / 0.2
    for x0, y0, x1, y1 in BLUR_BOXES:
        roi = fr[y0:y1, x0:x1]
        small = cv2.resize(roi, ((x1 - x0) // 10, (y1 - y0) // 10), interpolation=cv2.INTER_AREA)
        bl = cv2.GaussianBlur(cv2.resize(small, (x1 - x0, y1 - y0)), (0, 0), 7)
        fr[y0:y1, x0:x1] = cv2.addWeighted(bl, k, roi, 1 - k, 0)


# ------------------------------------------------------------------ window framing (screen recordings)
@functools.lru_cache(16)
def window_parts(w, h):
    r = int(30 * K)
    m = rr_mask(w, h, r, 2).astype(np.float32) / 255
    g = int(70 * K)
    W, H = w + 2 * g, h + 2 * g
    full = np.zeros((H, W), np.float32)
    full[g:g + h, g:g + w] = m
    shadow = np.clip(cv2.GaussianBlur(full, (0, 0), 26 * K) * 0.85, 0, 1)
    sh = np.zeros((H, W, 4), np.uint8)
    sh[:, :, 3] = (shadow * (1 - full) * 255).astype(np.uint8)
    er = cv2.erode(full, np.ones((3, 3), np.uint8), iterations=int(1.4 * K))
    rim = np.clip(full - er, 0, 1)
    sweep = np.exp(-((np.arange(W) - W * 0.3) / (W * 0.35)) ** 2)[None, :] * np.clip(1.2 - np.arange(H) / H, 0.2, 1)[:, None]
    rim_i = rim * (0.18 + 0.6 * sweep)
    glow = cv2.GaussianBlur(rim_i, (0, 0), 12 * K) * 0.8 * (1 - full)
    rimspr = np.zeros((H, W, 4), np.uint8)
    rimspr[:, :, :3] = 255
    rimspr[:, :, 3] = (np.clip(rim_i + glow, 0, 1) * 255).astype(np.uint8)
    corner = (m < 0.999)
    return sh, rimspr, g, m, corner


def window(bg, content, cx, cy):
    h, w = content.shape[:2]
    sh, rim, g, m, corner = window_parts(w, h)
    x, y = int(cx - w / 2), int(cy - h / 2)
    blend_rgba(bg, sh, x - g, y - g)
    roi = bg[y:y + h, x:x + w]
    # paste with rounded corners (blend only the corner squares)
    r = int(34 * K)
    out = content.copy()
    for ys in (slice(0, r), slice(h - r, h)):
        for xs in (slice(0, r), slice(w - r, w)):
            mm = m[ys, xs][:, :, None]
            out[ys, xs] = (content[ys, xs] * mm + roi[ys, xs] * (1 - mm)).astype(np.uint8)
    bg[y:y + h, x:x + w] = out
    blend_rgba(bg, rim, x - g, y - g)
    return bg


# ------------------------------------------------------------------ SCENES
# Each scene: (src_start, src_end, kind, params). kinds: window, paper, head, ui, phone
VIEWS = (400, 555, 750, 740)
REV = (1395, 555, 1745, 740)
FACE = (1220, 600)
PC = (1370, 700)  # paper centre

SCENES = []


def scene(a, b, kind, **kw):
    SCENES.append(dict(a=a, b=b, kind=kind, **kw))


# 1 hook: analytics, one push into "122.8M views"
scene(0.0, 7.3, "screen",
      cam=Track([(0.0, 1.0, 1280, 720), (0.7, 1.0, 1280, 720), (2.35, 1.95, 600, 640), (7.4, 2.02, 600, 646)]),
      spots=[(2.2, 7.4, VIEWS)],
      strokes=[Stroke(curve(482, 672, 628, 670, 3), 2.75, 7.25, cam=True, width=5)])
# 2 page: "YouTube AdSense", struck through
scene(7.3, 12.25, "ui", ui="adsense")
# 3 analytics: hold wide, one move onto the blurred revenue card, reveal, back out
scene(12.25, 39.16, "screen",
      cam=Track([(12.25, 1.0, 1280, 720), (21.7, 1.0, 1280, 720), (25.95, 2.05, 1570, 655),
                 (31.6, 2.1, 1570, 660), (33.4, 1.0, 1280, 720), (39.2, 1.0, 1280, 720)]),
      spots=[(22.3, 32.0, REV)],
      strokes=[Stroke(curve(1452, 672, 1652, 669, 4), 26.35, 31.6, cam=True, width=4)])
# 4 paper (stabilised), gentle camera; money cards pop over the darkened paper
scene(39.63, 111.0, "paper",
      cam=Track([(39.63, 1.14, 1370, 620), (80.0, 1.14, 1380, 860), (110.5, 1.08, 1370, 760)]),
      carddim=[(56.8, 74.7)])
# 5 talking head: $40,000 beside his head
scene(111.42, 167.8, "head",
      cam=Track([(111.42, 1.0, 1280, 720), (167.8, 1.0, 1280, 720)]),
      bgfx=[(116.9, 121.0), (124.9, 131.3), (136.5, 140.5), (144.3, 149.7)])
scene(167.8, 172.9, "ui", ui="ch1")
scene(172.9, 205.32, "paper",
      cam=Track([(172.9, 1.14, 1350, 640), (205.4, 1.10, 1370, 720)]))
# screen recordings stay raw: full frame, no captions, only spotlights on the key parts
scene(206.97, 308.3, "screen", cam=Track([(206.97, 1.0, 1280, 720), (308.3, 1.0, 1280, 720)]),
      carddim=[(277.9, 286.4)])
scene(308.3, 312.8, "ui", ui="ch2")
scene(312.8, 359.2, "screen", cam=Track([(312.8, 1.0, 1280, 720), (359.2, 1.0, 1280, 720)]),
)
scene(359.2, 363.9, "ui", ui="ch3")
scene(363.9, 505.0, "screen", cam=Track([(363.9, 1.0, 1280, 720), (505.0, 1.0, 1280, 720)]))
scene(505.0, 509.5, "ui", ui="ch4")
scene(509.5, 592.03, "screen", cam=Track([(509.5, 1.0, 1280, 720), (592.03, 1.0, 1280, 720)]))
scene(592.03, 609.45, "head",
      cam=Track([(592.03, 1.0, 1240, 690), (609.0, 1.06, 1235, 670)]),
      bgfx=[(599.0, 604.6)])

# transitions: zoom-through between consecutive scenes
NO_TRANS = {111.42}  # paper -> camera is a straight cut

# ------------------------------------------------------------------ text / UI content
W_ = OW / 2
T = []  # (scene index resolver by time) global overlays in output px, drawn on top of scene render


def L(*words):
    return list(words)


TEXTS = {
    "adsense": [
        TextSeq([L(("all", "sans", 7.45), ("my", "sans", 7.68), ("money", "sans", 7.84), ("comes", "sans", 8.24),
                   ("from", "sans", 8.48))], 12.2, (W_, OH / 2 - 250), sizes=(96, 230)),
        TextSeq([L(("YouTube", "serif", 8.64), ("AdSense", "serif", 9.12))], 12.2, (W_, OH / 2 + 40), sizes=(84, 230)),
    ],
    "40k": [
        TextSeq([L(("last", "sans", 117.1), ("month", "sans", 117.25)), L(("I", "sans", 117.4), ("made", "sans", 117.5))],
                120.9, (880, 600), sizes=(104, 200), align="r", behind=True, ry=-14, rz=-2),
        TextSeq([L(("$40,000", "serif", 117.84))], 120.9, (1690, 640), sizes=(100, 160), align="l", behind=True,
                ry=16, rz=4, stagger=0.05),
    ],
    "eyes": [
        TextSeq([L(("millions", "sans", 136.8), ("of", "sans", 137.6))], 140.3, (880, 610), sizes=(104, 200),
                align="r", behind=True, ry=-14),
        TextSeq([L(("eyes", "serif", 137.9))], 140.3, (1700, 620), sizes=(100, 230), align="l", behind=True, ry=16, rz=4),
    ],
    "paid": [
        TextSeq([L(("you", "sans", 144.6), ("get", "sans", 144.75))], 149.6, (880, 590), sizes=(110, 200),
                align="r", behind=True, ry=-14),
        TextSeq([L(("paid", "serif", 144.9))], 149.6, (1700, 600), sizes=(100, 240), align="l", behind=True, ry=16, rz=4),
    ],
    "discord": [
        TextSeq([L(("join", "sans", 599.3), ("the", "sans", 599.5))], 604.6, (880, 610), sizes=(110, 200), align="r",
                behind=True, ry=-14),
        TextSeq([L(("Discord", "serif", 599.7))], 604.6, (1690, 630), sizes=(100, 180), align="l", behind=True,
                ry=16, rz=4),
    ],
}


def chapter_texts(n, pre, special, t0, t1):
    return [
        TextSeq([L((f"STEP 0{n}", "grey", t0 + 0.05))], t1, (W_, 470), sizes=(40, 80), stagger=0.03),
        TextSeq([L(*[(w, "sans", t0 + 0.3 + 0.08 * i) for i, w in enumerate(pre.split())],
                   (special, "serif", t0 + 0.3 + 0.08 * len(pre.split())))], t1, (W_, 700), sizes=(110, 190)),
    ]


CH = {
    "ch1": (1, "Find a", "viral product", 167.85, 172.95),
    "ch2": (2, "Test it with", "ChatGPT", 308.35, 312.85),
    "ch3": (3, "Make the ads", "with AI", 359.25, 363.95),
    "ch4": (4, "Edit it in", "CapCut", 505.05, 509.55),
}
for key, (n, pre, sp, a, b) in CH.items():
    TEXTS[key] = chapter_texts(n, pre, sp, a, b)

STROKES = {
    "adsense": [Stroke(curve(W_ - 780, 800, W_ + 780, 768, -8 * K), 10.3, 12.2, dur=0.35, colour=WHITE, width=7)],
}
for key, (n, pre, sp, a, b) in CH.items():
    seq = TEXTS[key][1]
    serif_chars = [c for c in seq.chars if c["glow"] is not None]
    x0 = min(c["x"] for c in serif_chars) + 40 * K
    x1 = max(c["x"] for c in serif_chars) + serif_chars[-1]["px"] * 0.55
    y = 815
    STROKES[key] = [Stroke(curve(x0, y, x1, y - 8 * K, 14 * K), a + 1.0, b, dur=0.6, width=6)]


# ------------------------------------------------------------------ UI scene renderers
def ui_frame(name, t):
    fr = grid_bg(t)
    if name == "adsense":
        for s in TEXTS["adsense"] + STROKES["adsense"]:
            s.draw(fr, t)
    elif name in CH:
        n, _, _, a, b = CH[name]
        chapter_numeral(fr, n, t, a - 0.05, b + 0.2)
        for s in TEXTS[name] + STROKES[name]:
            s.draw(fr, t)
    return fr


@functools.lru_cache(None)
def numeral_outline(n):
    im = Image.new("L", (int(900 * K), int(700 * K)), 0)
    ImageDraw.Draw(im).text((im.size[0] / 2, im.size[1] / 2), f"0{n}", font=font(F_SERIF, 640 * K), fill=255,
                            anchor="mm", stroke_width=int(2 * K), stroke_fill=255)
    a_ = np.array(im).astype(np.float32) / 255
    inner = np.array(Image.new("L", im.size, 0))
    d = Image.new("L", im.size, 0)
    ImageDraw.Draw(d).text((im.size[0] / 2, im.size[1] / 2), f"0{n}", font=font(F_SERIF, 640 * K), fill=255, anchor="mm")
    inner = np.array(d).astype(np.float32) / 255
    return np.clip(a_ - cv2.erode(inner, np.ones((3, 3), np.uint8), iterations=int(2 * K)) * 0.9, 0, 1)


def chapter_numeral(fr, n, t, a, b):
    """huge faint outlined step number in the background."""
    k = ramp(t, a, b, 0.4, 0.3)
    if k <= 0:
        return
    outline = numeral_outline(n)
    s = 1.0 + 0.05 * (t - a) / max(0.1, b - a)
    outline = cv2.resize(outline, None, fx=s, fy=s)
    h, w = outline.shape
    x, y = int(OW / 2 - w / 2), int(OH / 2 - h / 2 - 20 * K)
    roi = fr[y:y + h, x:x + w].astype(np.float32)
    roi += outline[:, :, None] * 40 * k
    fr[y:y + h, x:x + w] = np.clip(roi, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ scene rendering
def render_window_content(sc, src, t, ws=0.9):
    fr = src.get(t)
    if t < REVEAL + 0.25:
        fr = fr.copy()
        revenue_blur(fr, t)
    z, cx, cy = sc["cam"](t) if "cam" in sc else (1.0, SW / 2, SH / 2)
    wcw, wch = int(OW * ws) // 2 * 2, int(OH * ws) // 2 * 2
    M = cam_matrix(z, cx, cy, out_w=wcw, out_h=wch)
    content = cv2.warpAffine(fr, M, (wcw, wch), flags=cv2.INTER_CUBIC if z * ws > 1.02 else cv2.INTER_AREA,
                             borderMode=cv2.BORDER_REPLICATE)
    return content, M


def window_scene(sc, src, t):
    bg = grid_bg(t)
    ws = 0.9
    wcx, wcy = OW / 2, OH / 2
    if sc.get("checklist"):
        k = e_io((t - 513.6) / 0.9) * (1 - e_io((t - 555.0) / 0.9))
        ws = 0.9 - 0.35 * k
        wcx = OW / 2 - 390 * K * k
    content, M = render_window_content(sc, src, t, ws)
    # spotlights / dimming happen inside the window (window-local coordinates)
    for a, b, rect in sc.get("spots", []):
        amt = ramp(t, a, b, 0.45, 0.45)
        if amt > 0:
            content = spotlight_local(content, rect, M, amt)
    for key in ("dimall", "dimall2"):
        if key in sc:
            a, b = sc[key]
            amt = ramp(t, a, b, 0.4, 0.4)
            content = dim_all(content, amt, 0.36, 11)
    for s in sc.get("strokes", []):
        s.draw_local = True
        s_draw_local(s, content, t, M)
    bg = window(bg, content, wcx, wcy)
    return bg


def spotlight_local(img, rect_src, M, amount, dim=0.32, blur=9):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = rect_src
    pts = np.array([[x0, y0], [x1, y1]], np.float32) @ M[:, :2].T + M[:, 2]
    (a, b), (c, d) = pts
    q = 4
    m = np.zeros((h // q, w // q), np.uint8)
    rr = max(2, int(24 * K / q))
    a_, b_, c_, d_ = int(a / q), int(b / q), int(c / q), int(d / q)
    cv2.rectangle(m, (a_ + rr, b_), (c_ - rr, d_), 255, -1)
    cv2.rectangle(m, (a_, b_ + rr), (c_, d_ - rr), 255, -1)
    for px_, py_ in ((a_ + rr, b_ + rr), (c_ - rr, b_ + rr), (a_ + rr, d_ - rr), (c_ - rr, d_ - rr)):
        cv2.circle(m, (px_, py_), rr, 255, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 6)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
    bg = soft_blur(img, blur * K).astype(np.float32) * dim
    k = 1 - amount * (1 - m)
    return (img * k + bg * (1 - k)).astype(np.uint8)


def s_draw_local(stroke, img, t, M):
    if t < stroke.t0 or t > stroke.t_out:
        return
    # temporarily treat img as the canvas (window-local)
    global OW, OH
    ow, oh = OW, OH
    OW, OH = img.shape[1], img.shape[0]
    try:
        stroke.draw(img, t, M)
    finally:
        OW, OH = ow, oh


def paper_scene(sc, src, t):
    fr = src.get(t)
    z, cx, cy = sc["cam"](t)
    z = max(z, 1.04)
    M = cam_matrix(z, cx, cy, stab=stab_at(t))
    fr = cv2.warpAffine(fr, M, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    fr = grade(fr)
    for key in ("dimall", "dimall2"):
        if key in sc:
            a, b = sc[key]
            fr = dim_all(fr, ramp(t, a, b, 0.4, 0.4), 0.38, 12)
    return fr


def head_scene(sc, src, t):
    fr = src.get(t)
    z, cx, cy = sc["cam"](t)
    M = cam_matrix(z, cx, cy)
    fr = cv2.warpAffine(fr, M, (OW, OH), flags=cv2.INTER_CUBIC if z > 1.01 else cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    fr = grade(fr)
    person = None
    amt = max([ramp(t, a, b, 0.45, 0.4) for a, b in sc.get("bgfx", [])] + [0])
    if amt > 0:
        # adjustment layer (blur + darker) masked by an inverted, heavily feathered ellipse around the speaker
        q = 4
        e = np.zeros((OH // q, OW // q), np.float32)
        c = np.array([1240, 820], np.float32) @ M[:, :2].T + M[:, 2]
        ax = (int(560 * z / q), int(820 * z / q))
        cv2.ellipse(e, (int(c[0] / q), int(c[1] / q)), ax, 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        e = cv2.GaussianBlur(e, (0, 0), 38)
        e = cv2.resize(e, (OW, OH), interpolation=cv2.INTER_LINEAR)[:, :, None]
        bg = soft_blur(fr, 16 * K).astype(np.float32) * 0.42
        k = 1 - amt * (1 - e)
        fr = (fr * k + bg * (1 - k)).astype(np.uint8)
        m = mask_at(t)
        if m is not None:  # soft matte, only used to tuck text slightly behind him
            Mq = M.copy()
            Mq[:, 2] /= 4
            mq = cv2.warpAffine(m, Mq, (OW // 4, OH // 4), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mq = cv2.GaussianBlur(mq.astype(np.float32) / 255, (0, 0), 2.0)
            mq = np.clip((mq - 0.45) / 0.4, 0, 1)
            person = cv2.resize(mq, (OW, OH), interpolation=cv2.INTER_LINEAR)
    return fr, person


PHONE = (1105, 110, 1495, 785)


def phone_scene(sc, src, t):
    fr = src.get(t)
    x0, y0, x1, y1 = PHONE
    crop = fr[y0:y1, x0:x1]
    a = sc["a"]
    p = e_out((t - a) / 1.2)
    s = 1.0 + 0.04 * (t - a) / (sc["b"] - a)
    # background: same clip, huge, blurred and dark
    bgc = cv2.resize(crop, (OW, int(OW * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_LINEAR)
    oy = (bgc.shape[0] - OH) // 2
    bg = soft_blur(bgc[oy:oy + OH], 40 * K)
    bg = (bg.astype(np.float32) * 0.28).astype(np.uint8)
    ph = int(1220 * s)
    pw = int(ph * crop.shape[1] / crop.shape[0]) // 2 * 2
    ph = ph // 2 * 2
    content = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_CUBIC)
    bg = window(bg, content, OW / 2, OH / 2 + 10 * K)
    return bg


@functools.lru_cache(None)
def TextSeq_cache_phone():
    return TextSeq([L(("the", "grey", 562.6), ("finished", "grey", 562.75), ("ad", "grey", 562.95))], 566.5,
                   (W_, 70 * K), sizes=(34, 60), stagger=0.02)


def card_dim(fr, amt):
    """'adjustment layer' behind cards: blur + darken + heavy vignette, so the cards pop."""
    if amt <= 0.01:
        return fr
    bg = soft_blur(fr, 12 * K)
    bg = cv2.multiply(cv2.multiply(bg, VIG, scale=1 / 255), VIG, scale=1 / 255)
    bg = bg.astype(np.float32) * 0.45
    return (fr * (1 - amt) + bg * amt).astype(np.uint8)


def money_cards(fr, t):
    if not (56.9 <= t <= 74.8):
        return
    y = OH / 2 + 30
    pop_in(fr, ui_card(540, 380, "1M views", "on YouTube Shorts", icon="play"), 450, y, t, 57.0, 74.6)
    Stroke(curve(828, y, 902, y, 0), 60.6, 74.4, dur=0.35, colour=(150, 190, 240), width=4).draw(fr, t)
    val = 0 if t < 61.73 else int(round(100 * e_out((t - 61.73) / 0.7)))
    pop_in(fr, ui_card(540, 380, f"${val}", "what YouTube pays", style="serif", icon="dollar", main_px=118),
           1280, y, t, 60.9, 74.6)
    if t >= 64.0:
        Stroke(curve(1280 - 190, y - 30, 1280 + 190, y - 62, 0), 64.0, 74.4, dur=0.3,
               colour=(70, 70, 235), width=6).draw(fr, t)
    Stroke(curve(1658, y, 1732, y, 0), 70.3, 74.4, dur=0.35, colour=ACCENT, width=4).draw(fr, t)
    pop_in(fr, ui_card(540, 380, "$500–$2,000", "what I want", style="serif", accent=True, icon="dollar",
                       main_px=80), 2110, y, t, 70.85, 74.6)


def overlays(fr, sc, t, person):
    """scene-specific text/cards drawn on the final composite."""
    a = sc["a"]
    if a == 39.63:
        money_cards(fr, t)
    elif a == 111.42:
        for s in TEXTS["40k"] + TEXTS["eyes"] + TEXTS["paid"]:
            s.draw(fr, t, person)
        if 125.2 <= t <= 131.3:  # the old way, one card at a time on his left while he keeps talking
            for (lab, ic, t0), yy in zip((("Pay advertisers", "dollar", 125.4), ("TV commercials", "tv", 128.7),
                                          ("Paid ads", "ad", 129.9)), (470, 720, 970)):
                pop_in(fr, ui_card(600, 170, lab, icon=ic, main_px=50), 520, yy, t, t0, 131.2)
    elif a == 206.97:
        if 278.0 <= t <= 286.6:
            for (big, sub, t0), xx in zip((("64M", "views · 1 week", 278.4), ("3M", "views · 1 week", 281.0),
                                           ("2M", "views · 1 week", 282.2), ("1M", "views · 4 days", 284.1)),
                                          (390, 1003, 1617, 2230)):
                pop_in(fr, ui_card(420, 330, big, sub, style="serif", accent=big == "64M"), xx, OH / 2, t, t0, 286.2)
    elif a == 592.03:
        for s in TEXTS["discord"]:
            s.draw(fr, t, person)


def ring_card(fr, t):
    w, h = int(760 * K), int(560 * K)
    base, g = ui_card(760, 560, accent=True)
    p = e_out((t - 325.6) / 1.1)
    val = int(round(82 * p))
    im = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGRA2RGBA))
    d = ImageDraw.Draw(im)
    cx, cy, R = g + w * 0.30, g + h * 0.5, h * 0.30
    d.ellipse((cx - R, cy - R, cx + R, cy + R), outline=(60, 60, 66, 255), width=int(16 * K))
    if val > 0:
        d.arc((cx - R, cy - R, cx + R, cy + R), -90, -90 + 360 * val / 100, fill=rgb(ACCENT) + (255,), width=int(16 * K))
    draw_text_pil(im, (cx, cy + 6 * K), str(val), F_SERIF, 150 * K, rgb(ACCENT) + (255,), "mm", glow=True)
    tx = g + w * 0.58
    draw_text_pil(im, (tx, g + h * 0.30), "PRODUCT SCORE", F_SANS_SB, 24 * K, rgb(SECOND) + (255,), "lm", tracking=4 * K)
    draw_text_pil(im, (tx, g + h * 0.45), "82 / 100", F_SANS_SB, 64 * K, rgb(WHITE) + (255,), "lm")
    if t >= 327.6:
        draw_text_pil(im, (tx, g + h * 0.66), "Verdict:", F_SANS, 46 * K, rgb(SECOND) + (255,), "lm")
        draw_text_pil(im, (tx + 190 * K, g + h * 0.65), "Good", F_SERIF, 84 * K, rgb(ACCENT) + (255,), "lm", glow=True)
    pop_in(fr, (pil_to_bgra(im), g), OW / 2, OH / 2, t, 323.0, 329.8, dur=0.6)


def screen_scene(sc, src, t):
    """screen recordings: full frame and raw, camera only for the highlights."""
    content, M = render_window_content(sc, src, t, 1.0)
    for a, b, rect in sc.get("spots", []):
        amt = ramp(t, a, b, 0.45, 0.45)
        if amt > 0:  # hook: strong focus; screen-share tutorial: light highlight, everything stays readable
            content = (spotlight_local(content, rect, M, amt) if sc["a"] < 40 else
                       spotlight_local(content, rect, M, amt, dim=0.55, blur=2))
    for s in sc.get("strokes", []):
        s_draw_local(s, content, t, M)
    return content


def render_scene(sc, src, t):
    kind = sc["kind"]
    person = None
    if kind == "screen":
        fr = screen_scene(sc, src, t)
    elif kind == "paper":
        fr = paper_scene(sc, src, t)
    elif kind == "head":
        fr, person = head_scene(sc, src, t)
    elif kind == "ui":
        fr = ui_frame(sc["ui"], t)
    else:
        fr = phone_scene(sc, src, t)
    amt = max([ramp(t, a, b, 0.35, 0.35) for a, b in sc.get("carddim", [])] + [0])
    if amt > 0:
        fr = card_dim(fr, amt)
    overlays(fr, sc, t, person)
    return fr


def zoom_frame(fr, s, blur=0.0):
    if abs(s - 1) > 1e-3:
        M = np.float32([[s, 0, OW / 2 * (1 - s)], [0, s, OH / 2 * (1 - s)]])
        fr = cv2.warpAffine(fr, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if blur > 0.5:
        fr = soft_blur(fr, blur)
    return fr


TD = 0.40  # transition length


def scene_index(t):
    for i, sc in enumerate(SCENES):
        if sc["a"] <= t < sc["b"]:
            return i
    return len(SCENES) - 1


def frame_at(tau, src):
    t = to_src(tau)
    i = scene_index(t)
    sc = SCENES[i]
    # transition into this scene?
    ta = to_out(sc["a"])
    tb = to_out(sc["b"])
    if i > 0 and sc["a"] not in NO_TRANS and tau - ta < TD / 2:
        prev = SCENES[i - 1]
        u = 0.5 + (tau - ta) / TD
        A = render_scene(prev, src, prev["b"] + (tau - ta))
        B = render_scene(sc, src, t)
        return trans(A, B, u)
    if i < len(SCENES) - 1 and SCENES[i + 1]["a"] not in NO_TRANS and tb - tau < TD / 2:
        nxt = SCENES[i + 1]
        u = 0.5 - (tb - tau) / TD
        A = render_scene(sc, src, t)
        B = render_scene(nxt, src, nxt["a"] - (tb - tau))
        return trans(A, B, u)
    return render_scene(sc, src, t)


def trans(A, B, u):
    w = e_sm(u)
    a = zoom_frame(A, 1 + 0.08 * e_io(u), 18 * K * math.sin(math.pi * clamp(u)) ** 1.5)
    b = zoom_frame(B, 0.95 + 0.05 * e_out(u), 18 * K * math.sin(math.pi * clamp(u)) ** 1.5)
    return cv2.addWeighted(a, 1 - w, b, w, 0)


def render_video(t0, t1, out):
    src = Source(to_src(t0) - 1.0)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-tune", "film",
         "-pix_fmt", "yuv420p", "-x264-params", "keyint=60", out], stdin=subprocess.PIPE)
    n0 = int(round(t0 * FPS))
    n1 = int(round(min(t1, TOTAL) * FPS))
    for n in range(n0, n1):
        tau = n / FPS
        fr = frame_at(tau, src)
        if tau > TOTAL - 1.2:  # fade to black
            fr = (fr * clamp((TOTAL - tau) / 1.2)).astype(np.uint8)
        enc.stdin.write(np.ascontiguousarray(fr).tobytes())
        if n % 150 == 0:
            print(f"{out}: {tau:7.1f}/{t1:.1f}", flush=True)
    enc.stdin.close()
    enc.wait()
    src.p.kill()


# ------------------------------------------------------------------ AUDIO
def load(path, start=0.0, dur=None):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start)]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-i", str(path), "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    a = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)
    return a.reshape(-1, 2).copy()


def db(x):
    return 10 ** (x / 20)


def rms(a):
    return float(np.sqrt(np.mean(a ** 2)) + 1e-9)


RNG = np.random.default_rng(7)


def onepole(x, fc):
    """time-varying one-pole low-pass, fc array (Hz)."""
    y = np.zeros_like(x)
    a = np.exp(-2 * np.pi * fc / SR)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a[i]) * x[i] + a[i] * acc
        y[i] = acc
    return y


def sfx_whoosh(d=0.7, lo=250, hi=5500, peak=0.55):
    n = int(d * SR)
    t = np.linspace(0, 1, n)
    env = np.where(t < peak, (t / peak) ** 2.2, ((1 - t) / (1 - peak)) ** 1.6)
    fc = lo + (hi - lo) * env
    noise = RNG.standard_normal(n)
    lp = onepole(noise, fc)
    hp = lp - onepole(lp, np.full(n, 160.0))
    s = hp * env
    s /= np.abs(s).max() + 1e-9
    pan = np.clip(t * 1.4 - 0.2, 0, 1)
    return np.stack([s * np.cos(pan * np.pi / 2) * 1.2, s * np.sin(pan * np.pi / 2) * 1.2], 1).astype(np.float32)


def sfx_hit(d=2.4, f0=58):
    n = int(d * SR)
    t = np.arange(n) / SR
    f = f0 * (1 + 1.2 * np.exp(-t * 18))
    ph = 2 * np.pi * np.cumsum(f) / SR
    sub = np.sin(ph) * np.exp(-t * 2.2)
    trans_ = RNG.standard_normal(n) * np.exp(-t * 40)
    trans_ = onepole(trans_, np.full(n, 2500.0))
    tail = onepole(RNG.standard_normal(n), np.full(n, 900.0)) * np.exp(-t * 2.8) * 0.25
    s = sub * 1.0 + trans_ * 0.6 + tail
    s /= np.abs(s).max()
    return np.stack([s, s], 1).astype(np.float32)


def sfx_pop(f=740):
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * np.cumsum(f * (1 + 0.6 * np.exp(-t * 60))) / SR
    s = np.sin(ph) * np.exp(-t * 34) + RNG.standard_normal(n) * np.exp(-t * 300) * 0.25
    s /= np.abs(s).max()
    return np.stack([s, s], 1).astype(np.float32)


def sfx_shimmer(d=1.4):
    n = int(d * SR)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for f, g in ((2637, 1), (3520, .7), (4186, .55), (5274, .4), (6272, .3)):
        s += np.sin(2 * np.pi * f * t + RNG.uniform(0, 6)) * g * np.exp(-t * (3 + f / 2000))
    s *= np.clip(t / 0.01, 0, 1)
    s /= np.abs(s).max()
    l = s
    r = np.roll(s, int(0.012 * SR))
    return np.stack([l, r], 1).astype(np.float32)


def sfx_tick():
    n = int(0.025 * SR)
    t = np.arange(n) / SR
    s = RNG.standard_normal(n) * np.exp(-t * 400)
    s = s - onepole(s, np.full(n, 3000.0))
    s /= np.abs(s).max()
    return np.stack([s, s], 1).astype(np.float32)


def sfx_ching():
    t = np.arange(int(SR * 1.6)) / SR
    bell = sum(np.sin(2 * np.pi * f * t) * g * np.exp(-t * dd)
               for f, g, dd in ((2093, 1, 3.0), (2637, .7, 3.5), (3136, .5, 4), (4186, .35, 5), (5274, .2, 6)))
    s = bell * 0.4 + RNG.standard_normal(len(t)) * np.exp(-t * 60) * 0.5
    s /= np.abs(s).max()
    return np.stack([s, s], 1).astype(np.float32)


def render_audio(out_wav):
    voice = load(RAW)[: int(END * SR)]
    # clips played during the screen share were recorded twice (desktop audio + mic hearing the speakers,
    # ~51 ms later). Cancel that copy with an inverse comb filter on just those stretches.
    for a_, b_, dly, gain, lp in ((244.0, 259.6, 0.0514, 0.30, 1500.0), (563.0, 579.1, 0.0511, 0.30, 3000.0)):
        i0, i1 = int(a_ * SR), int(b_ * SR)
        seg = voice[i0:i1]
        N = 1 << int(np.ceil(np.log2(len(seg) + SR)))
        f = np.fft.rfftfreq(N, 1 / SR)
        G = gain * np.exp(-2j * np.pi * f * dly) / (1 + 1j * f / lp)
        clean = np.fft.irfft(np.fft.rfft(seg, N, axis=0) / (1 + G)[:, None], N, axis=0)[:len(seg)].astype(np.float32)
        k = int(0.3 * SR)
        w = np.ones(len(seg), np.float32)
        w[:k] = np.linspace(0, 1, k)
        w[-k:] = np.linspace(1, 0, k)
        voice[i0:i1] = seg * (1 - w[:, None]) + clean * w[:, None]
    segs, prev = [], 0.0
    for a, b in CUTS + [(END, END)]:
        segs.append(voice[int(prev * SR):int(a * SR)])
        prev = b
    fade = int(0.006 * SR)
    v = segs[0]
    for s in segs[1:]:
        if len(s) == 0:
            continue
        r = np.linspace(0, 1, fade)[:, None]
        v = np.concatenate([v[:-fade], v[-fade:] * (1 - r) + s[:fade] * r, s[fade:]])
    n = int(TOTAL * SR)
    v = v[:n]
    if len(v) < n:
        v = np.concatenate([v, np.zeros((n - len(v), 2), np.float32)])
    vr = rms(v[np.abs(v).max(1) > 0.02])
    mus = np.zeros((n, 2), np.float32)
    fx = np.zeros((n, 2), np.float32)

    def place(buf, clip, t_out, gain_db, fi=0.0, fo=0.0):
        i = int(t_out * SR)
        c = clip.copy()
        if fi:
            k = min(len(c), int(fi * SR))
            c[:k] *= np.linspace(0, 1, k)[:, None]
        if fo:
            k = min(len(c), int(fo * SR))
            c[-k:] *= np.linspace(1, 0, k)[:, None]
        if i < 0:
            c, i = c[-i:], 0
        j = min(n, i + len(c))
        if j > i:
            buf[i:j] += c[: j - i] * db(gain_db)

    def bed(path, a_src, b_src, level_db, start=0.0, fi=0.6, fo=0.8, loop_from=None):
        a, b = to_out(a_src), to_out(b_src)
        dur = b - a
        m = load(path, start)
        ref = rms(load(path, start, 30))
        while len(m) < dur * SR:
            tail = load(path, loop_from if loop_from is not None else start)
            k = 2 * SR
            r = np.linspace(0, 1, k)[:, None]
            m = np.concatenate([m[:-k], m[-k:] * (1 - r) + tail[:k] * r, tail[k:]])
        m = m[: int(dur * SR)]
        m *= vr * db(level_db) / ref
        place(mus, m, a, 0, fi, fo)

    devil = Path("/root/.claude/uploads/a247f031-ec7c-562e-977d-60044216bebc/0ca69663-Me_And_The_Devil_Instrumental.mp3")
    if not devil.exists():
        devil = REPO / "assets/music/user/me-and-the-devil-instrumental.mp3"
    mdir = REPO / "assets/music"
    # music: intense track for the hook and the money story, cut to silence right before each reveal
    bed(devil, 0.0, 25.85, -18, start=1.35, fi=0.02, fo=0.12)
    bed(devil, 26.0, 39.3, -20, start=46.0, fi=0.02, fo=1.2)
    # paper drawing + tutorial: mature, minimal beds kept low so his explanation sits on top
    bed(mdir / "suspense-underscore/crypto.mp3", 39.63, 111.2, -29, start=0.0, fi=1.5, fo=2.5)
    # back on camera: the intense track creeps in very slowly underneath him
    bed(devil, 111.42, 167.9, -23, start=40.0, fi=7.0, fo=1.0, loop_from=6.0)
    # step 1 paper; fades out slowly across the cut into the screen share
    bed(devil, 167.8, 210.5, -27, start=6.0, fi=1.2, fo=4.5)
    # screen shares (206.97-592.03) stay raw: no music
    bed(devil, 592.03, END, -22, start=6.0, fi=3.5, fo=2.5)

    # sound design
    wh_fast, wh_mid, wh_slow = sfx_whoosh(0.55, 300, 6500, 0.6), sfx_whoosh(0.8, 200, 5000, 0.6), sfx_whoosh(1.3, 150, 3800, 0.7)
    pop, shim, tick, hit, ching = sfx_pop(), sfx_shimmer(), sfx_tick(), sfx_hit(), sfx_ching()
    for tt in (125.4, 128.7, 129.9):
        place(fx, pop, to_out(tt), -22)
    cg = REPO / "assets/sfx/cinematic-generated"
    riser = load(cg / "riser-tonal-4s.wav")
    swell = load(cg / "reverse-swell-2s.wav")
    boom = load(cg / "boom-cinematic-3-punchy.wav")
    boom2 = load(cg / "boom-cinematic-1.wav")
    sub = load(cg / "sub-drop-short.wav")

    def S(clip, t_src, g, align_end=False):
        t = to_out(t_src)
        if align_end:
            t -= len(clip) / SR
        place(fx, clip, t, g)

    # transitions
    for i in range(1, len(SCENES)):
        a = SCENES[i]["a"]
        if a in NO_TRANS:
            continue
        S(wh_mid, a, -15, align_end=False) if False else place(fx, wh_mid, to_out(a) - 0.8 * 0.6, -14)
    # hook
    place(fx, sub, 0.0, -10)
    S(wh_slow, 1.0, -20)
    S(shim, 2.75, -24)
    S(riser, REVEAL, -12, align_end=True)
    S(swell, REVEAL, -14, align_end=True)
    S(boom, REVEAL, -7)
    S(hit, REVEAL, -9)
    S(ching, REVEAL + 0.05, -16)
    S(shim, 26.35, -22)
    # text write-ons: soft shimmer on special (serif) words, ticks on count-ups
    for key, seqs in TEXTS.items():
        for s in seqs:
            for c in s.chars:
                if c["glow"] is not None:
                    S(shim, c["t0"], -27)
                    break
    for tt in np.arange(61.73, 62.4, 0.05):
        S(tick, tt, -24)
    # UI pops
    for tt in (57.0, 60.9, 70.85, 278.4, 281.0, 282.2, 284.1):
        S(pop, tt, -19)
        S(wh_fast, tt - 0.25, -24)
    S(hit, 70.85, -14)
    S(ching, 61.75, -21)
    # $40,000: silence, then hit
    S(swell, 117.84, -13, align_end=True)
    S(boom2, 117.84, -8)
    S(hit, 117.84, -10)
    S(ching, 117.9, -17)
    # spotlights
    pass  # spotlights on screen shares stay silent
    # zooms in window scenes
    pass

    # duck music under the finished-ad playback is implicit (no bed there)
    mix = v + mus + fx * db(-3)
    raw = WORK / "mix_raw.wav"
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(raw), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af",
                    "highpass=f=60,acompressor=threshold=-20dB:ratio=2.2:attack=10:release=150:makeup=1,"
                    "loudnorm=I=-14:TP=-1.2:LRA=10", "-ar", str(SR), "-c:a", "pcm_s16le", str(out_wav)], check=True)


if __name__ == "__main__":
    if MODE == "video":
        render_video(float(sys.argv[4]), float(sys.argv[5]), sys.argv[6])
    elif MODE == "audio":
        render_audio(sys.argv[4])
    elif MODE == "frames":  # debug: write PNGs at given output times
        for tau in sys.argv[5:]:
            tau = float(tau)
            src = Source(to_src(tau) - 0.5)
            cv2.imwrite(f"{sys.argv[4]}/f_{tau:.2f}.jpg", frame_at(tau, src), [cv2.IMWRITE_JPEG_QUALITY, 88])
            src.p.kill()
