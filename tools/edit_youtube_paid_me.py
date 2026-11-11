"""Edit of 'This is how much youtube paid me for 144m views'.

Video pass: decodes the raw 2560x1440 clip at 30 fps, applies
  * the revenue blur (until he says "$17,000"), then the reveal
  * smooth punch-in zooms / slow push-ins
  * chapter labels + a few text pops
  * removes a handful of dead-air gaps
and pipes 1920x1080 frames to x264.

Audio pass (numpy): rebuilds the voice track with the same cuts, lays music beds
and sound effects, then ffmpeg loudness-normalises to -14 LUFS and muxes.

usage: python tools/edit_youtube_paid_me.py RAW.mp4 OUT.mp4 WORKDIR [t0 t1]
"""
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAW, OUT, WORK = sys.argv[1], sys.argv[2], Path(sys.argv[3])
T0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
T1 = float(sys.argv[5]) if len(sys.argv) > 5 else None
WORK.mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parent.parent
SFX = REPO / "assets/sfx"
SW, SH = 2560, 1440
OW, OH = 1920, 1080
FPS = 30
SR = 48000
END = 609.0  # drop the trailing "Mm."

# ---------------------------------------------------------------- cuts (source s)
CUTS = [
    (39.16, 39.63), (54.77, 55.20), (66.36, 66.65), (109.88, 111.42),
    (140.20, 140.47), (187.71, 188.07), (205.32, 206.97), (259.33, 259.73),
    (358.34, 358.83), (367.64, 367.93), (397.93, 398.88), (496.56, 497.72),
]


def kept(t):
    return t < END and not any(a <= t < b for a, b in CUTS)


def to_out(t):
    """source time -> output time"""
    return t - sum(min(b, max(a, t)) - a for a, b in CUTS)


# ---------------------------------------------------------------- camera
def ease(x):
    x = min(1.0, max(0.0, x))
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


CENTER = (SW / 2, SH / 2)
FACE = (1220, 620)
# (start, end, z_from, z_to, centre)
BASE = [
    (0.0, 39.27, 1.0, 1.08, CENTER),
    (39.27, 111.13, 1.0, 1.12, (1333, 720)),
    (111.13, 167.8, 1.0, 1.06, FACE),
    (167.8, 206.87, 1.0, 1.12, (1333, 720)),
    (206.87, 592.03, 1.0, 1.0, CENTER),
    (592.03, 610.0, 1.0, 1.08, FACE),
]
# (start, end, zoom factor, centre, ease-in s, ease-out s)
PUNCH = [
    (2.25, 5.40, 1.90, (573, 647), 0.20, 0.50),      # "122 million views"
    (21.8, 29.30, 1.60, (1570, 647), 4.00, 0.70),    # drift to the revenue card
    (26.0, 29.30, 1.30, (1570, 647), 0.12, 0.70),    # "$17,000" reveal punch
    (61.6, 66.20, 1.30, (1333, 700), 0.15, 0.40),    # "$100" per million
    (70.8, 74.30, 1.30, (1333, 780), 0.15, 0.40),    # "$500 to $1,000"
    (117.75, 121.9, 1.35, (1220, 640), 0.10, 0.35),  # "$40,000"
    (130.8, 142.6, 1.12, FACE, 0.12, 0.35),
    (150.2, 158.0, 1.18, FACE, 0.12, 0.35),          # "this business is insane"
    (278.2, 285.6, 1.80, (800, 690), 0.25, 0.45),    # 64M views in one week
    (323.0, 329.0, 2.00, (900, 480), 0.25, 0.45),    # 82/100
    (562.4, 579.3, 1.90, (1298, 446), 0.35, 0.45),   # the finished ad
]


def camera(t):
    z, (cx, cy) = 1.0, CENTER
    for a, b, z0, z1, c in BASE:
        if a <= t < b:
            z = z0 + (z1 - z0) * (t - a) / (b - a)
            cx, cy = c
    for a, b, pz, (px, py), ei, eo in PUNCH:
        if a <= t < b + eo:
            p = ease((t - a) / ei) if t < b else 1 - ease((t - b) / eo)
            z *= 1 + p * (pz - 1)
            cx += p * (px - cx)
            cy += p * (py - cy)
    # keep the crop inside the frame
    hw, hh = SW / (2 * z), SH / (2 * z)
    cx = min(max(cx, hw), SW - hw)
    cy = min(max(cy, hh), SH - hh)
    return z, cx, cy


# ---------------------------------------------------------------- revenue blur
BLUR_BOXES = [(1395, 555, 1745, 735), (1590, 780, 1700, 1000)]  # card, $ y-axis
REVEAL = 26.0
CARD = (1407, 567, 1733, 727)


def apply_blur(frame, t):
    if t >= REVEAL + 0.15:
        return
    k = 1.0 if t < REVEAL else 1 - (t - REVEAL) / 0.15
    for x0, y0, x1, y1 in BLUR_BOXES:
        roi = frame[y0:y1, x0:x1]
        small = cv2.resize(roi, ((x1 - x0) // 8, (y1 - y0) // 8), interpolation=cv2.INTER_AREA)
        bl = cv2.GaussianBlur(cv2.resize(small, (x1 - x0, y1 - y0)), (0, 0), 6)
        frame[y0:y1, x0:x1] = cv2.addWeighted(bl, k, roi, 1 - k, 0)


# ---------------------------------------------------------------- text overlays
FONT_X = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
FONT_B = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
GREEN = (61, 255, 139)


def rgba(img):
    a = np.array(img)
    return cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA)


def chapter(step, title):
    fs, fb = ImageFont.truetype(FONT_X, 30), ImageFont.truetype(FONT_X, 52)
    tag = f"STEP {step}"
    tw = fs.getbbox(tag)[2] + 36
    ttw = fb.getbbox(title)[2]
    w, h = 24 + tw + 24 + ttw + 36, 104
    im = Image.new("RGBA", (w + 20, h + 20), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((10, 10, 10 + w, 10 + h), 30, fill=(14, 14, 18, 225), outline=(255, 255, 255, 40), width=2)
    d.rounded_rectangle((34, 34, 34 + tw, 34 + 56), 18, fill=GREEN + (255,))
    d.text((34 + 18, 34 + 10), tag, font=fs, fill=(10, 10, 10, 255))
    d.text((34 + tw + 24, 30), title, font=fb, fill=(255, 255, 255, 255))
    return rgba(im)


def money(value, sub):
    fv, fsb = ImageFont.truetype(FONT_X, 170), ImageFont.truetype(FONT_X, 44)
    vw, sw_ = fv.getbbox(value)[2], fsb.getbbox(sub)[2]
    w, h = max(vw, sw_) + 80, 300
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text(((w - sw_) / 2, 20), sub, font=fsb, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(glow).text(((w - vw) / 2, 70), value, font=fv, fill=GREEN + (170,))
    arr = cv2.GaussianBlur(np.array(glow), (0, 0), 14)
    im = Image.alpha_composite(Image.fromarray(arr), im)
    d = ImageDraw.Draw(im)
    d.text(((w - vw) / 2, 70), value, font=fv, fill=GREEN + (255,), stroke_width=6, stroke_fill=(0, 0, 0, 255))
    return rgba(im)


def pill(text):
    f = ImageFont.truetype(FONT_X, 46)
    tw = f.getbbox(text)[2]
    w, h = tw + 80, 96
    im = Image.new("RGBA", (w + 20, h + 20), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((10, 10, 10 + w, 10 + h), 48, fill=(14, 14, 18, 230), outline=GREEN + (200,), width=3)
    d.text((50, 30), text, font=f, fill=(255, 255, 255, 255))
    return rgba(im)


# (start, end, image, anchor, (x, y))  anchor "tl" top-left / "c" centre
OVERLAYS = [
    (117.84, 121.9, money("$40,000", "LAST MONTH"), "c", (OW / 2, 905)),
    (168.0, 173.4, chapter(1, "Find a viral product"), "tl", (48, 44)),
    (308.4, 313.6, chapter(2, "Test it with ChatGPT"), "tl", (48, 44)),
    (359.3, 364.6, chapter(3, "Make the ads with AI"), "tl", (48, 44)),
    (505.0, 510.2, chapter(4, "Edit it in CapCut"), "tl", (48, 44)),
    (599.3, 605.0, pill("Discord link in the description  ↓"), "c", (OW / 2, 940)),
]


def blend(frame, img, x, y, alpha):
    h, w = img.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, OW), min(y + h, OH)
    if x0 >= x1 or y0 >= y1:
        return
    src = img[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[:, :, 3:4].astype(np.float32) / 255.0 * alpha
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi * (1 - a) + src[:, :, :3] * a).astype(np.uint8)


def draw_overlays(frame, t):
    for a, b, img, anc, (px, py) in OVERLAYS:
        if not (a <= t < b):
            continue
        r = t - a
        k = min(1.0, r / 0.22)
        s = 0.8 + 0.2 * (1 - (1 - k) ** 3) + 0.06 * math.sin(math.pi * k)  # pop with overshoot
        alpha = min(1.0, r / 0.12, (b - t) / 0.25)
        im = img if abs(s - 1) < 1e-3 else cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        h, w = im.shape[:2]
        if anc == "c":
            x, y = px - w / 2, py - h / 2
        else:
            x = px - (w - img.shape[1]) / 2
            y = py - (h - img.shape[0]) / 2
            x += (1 - k) * -40
        blend(frame, im, x, y, alpha)


def draw_card_glow(frame, t, z, cx, cy):
    a, b = REVEAL, 29.6
    if not (a <= t < b):
        return
    alpha = min(1.0, (t - a) / 0.2, (b - t) / 0.4)
    sc = z * OW / SW

    def m(x, y):
        return int((x - cx) * sc + OW / 2), int((y - cy) * sc + OH / 2)

    p0, p1 = m(CARD[0] - 6, CARD[1] - 6), m(CARD[2] + 6, CARD[3] + 6)
    rad = int(22 * sc)
    mask = np.zeros(frame.shape[:2], np.uint8)
    (x0, y0), (x1, y1) = p0, p1
    cv2.rectangle(mask, (x0 + rad, y0), (x1 - rad, y1), 255, -1)
    cv2.rectangle(mask, (x0, y0 + rad), (x1, y1 - rad), 255, -1)
    for cx_, cy_ in ((x0 + rad, y0 + rad), (x1 - rad, y0 + rad), (x0 + rad, y1 - rad), (x1 - rad, y1 - rad)):
        cv2.circle(mask, (cx_, cy_), rad, 255, -1, cv2.LINE_AA)
    ring = cv2.subtract(cv2.dilate(mask, np.ones((7, 7), np.uint8)), cv2.erode(mask, np.ones((3, 3), np.uint8)))
    halo = cv2.GaussianBlur(ring, (0, 0), 9).astype(np.float32) * 1.6
    core = ring.astype(np.float32)
    glow = np.zeros_like(frame, dtype=np.float32)
    for i, c in enumerate((139, 255, 61)):
        glow[:, :, i] = np.clip(halo * c / 255 + core * (c / 255 * 0.5 + 0.5), 0, 255)
    frame[:] = cv2.add(frame, (glow * alpha).astype(np.uint8))


# ---------------------------------------------------------------- render video
def render_video(vout):
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", str(T0), "-i", RAW, "-an", "-vf", f"fps={FPS}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=SW * SH * 3 * 2)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-pix_fmt", "yuv420p", vout], stdin=subprocess.PIPE)
    n, fsz = 0, SW * SH * 3
    while True:
        buf = dec.stdout.read(fsz)
        if len(buf) < fsz:
            break
        t = T0 + n / FPS
        n += 1
        if T1 is not None and t >= T1:
            break
        if not kept(t):
            continue
        fr = np.frombuffer(buf, np.uint8).reshape(SH, SW, 3).copy()
        apply_blur(fr, t)
        z, cx, cy = camera(t)
        if z * OW / SW < 1.0:
            small = cv2.resize(fr, (OW, OH), interpolation=cv2.INTER_AREA)
            s, ccx, ccy = z, cx * OW / SW, cy * OH / SH
            M = np.float32([[s, 0, OW / 2 - ccx * s], [0, s, OH / 2 - ccy * s]])
            out = cv2.warpAffine(small, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            s = z * OW / SW
            M = np.float32([[s, 0, OW / 2 - cx * s], [0, s, OH / 2 - cy * s]])
            out = cv2.warpAffine(fr, M, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        draw_card_glow(out, t, z, cx, cy)
        draw_overlays(out, t)
        enc.stdin.write(out.tobytes())
        if n % 900 == 0:
            print(f"video {t:7.1f}s", flush=True)
    enc.stdin.close()
    enc.wait()
    dec.kill()


# ---------------------------------------------------------------- audio
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


def cash_ching():
    t = np.arange(int(SR * 1.6)) / SR
    bell = sum(np.sin(2 * np.pi * f * t) * g * np.exp(-t * d)
               for f, g, d in ((2093, 1, 3.0), (2637, .7, 3.5), (3136, .5, 4), (4186, .35, 5), (5274, .2, 6)))
    bell2 = np.zeros_like(bell)
    off = int(SR * 0.07)
    bell2[off:] = bell[:-off] * 0.6
    rng = np.random.default_rng(1)
    click = rng.standard_normal(len(t)) * np.exp(-t * 60) * 0.6
    drawer = np.sin(2 * np.pi * 90 * t) * np.exp(-t * 18) * 0.8
    s = (bell + bell2) * 0.35 + click + drawer
    s /= np.abs(s).max()
    return np.stack([s, s], 1).astype(np.float32)


def render_audio(aout, total):
    voice = load(RAW)
    voice = voice[: int(END * SR)]
    # rebuild voice with the cuts (5 ms crossfade at each join)
    segs, prev = [], 0.0
    for a, b in CUTS + [(END, END)]:
        segs.append(voice[int(prev * SR):int(a * SR)])
        prev = b
    fade = int(0.005 * SR)
    v = segs[0]
    for s in segs[1:]:
        if len(s) == 0:
            continue
        ramp = np.linspace(0, 1, fade)[:, None]
        v = np.concatenate([v[:-fade], v[-fade:] * (1 - ramp) + s[:fade] * ramp, s[fade:]])
    v = v[: int(total * SR)]
    n = len(v)
    vr = rms(v[np.abs(v).max(1) > 0.02])
    mix = np.zeros((n, 2), np.float32)

    def place(clip, t_src, gain_db, fade_in=0.0, fade_out=0.0, src_time=True):
        t = to_out(t_src) if src_time else t_src
        i = int(t * SR)
        c = clip.copy()
        if fade_in:
            k = min(len(c), int(fade_in * SR))
            c[:k] *= np.linspace(0, 1, k)[:, None]
        if fade_out:
            k = min(len(c), int(fade_out * SR))
            c[-k:] *= np.linspace(1, 0, k)[:, None]
        j = min(n, i + len(c))
        if j > i >= 0:
            mix[i:j] += c[: j - i] * db(gain_db)

    def bed(path, a_src, b_src, level_db, start=0.0, fi=1.0, fo=1.0):
        a, b = to_out(a_src), to_out(b_src)
        dur = b - a
        m = load(path, start)
        while len(m) < dur * SR:  # loop with a 2 s crossfade
            k = 2 * SR
            r = np.linspace(0, 1, k)[:, None]
            m = np.concatenate([m[:-k], m[-k:] * (1 - r) + m[:k] * r, m[k:]])
        m = m[: int(dur * SR)]
        m *= vr * db(level_db) / rms(m)
        place(m, a, 0, fi, fo, src_time=False)

    music = REPO / "assets/music"
    ext = music / "background-chill"
    # music beds (level relative to the voice)
    bed(music / "tension-build/i-can-feel-it-coming.mp3", 0.0, 25.85, -20, fi=0.4, fo=0.5)
    bed(ext / "investigations.mp3", 26.4, 109.9, -21, fi=1.5, fo=1.0)
    bed(ext / "chill-wave.mp3", 111.42, 117.65, -21, fi=0.6, fo=0.25)
    bed(ext / "chill-wave.mp3", 119.2, 562.35, -22, start=8.0, fi=1.2, fo=0.6)
    bed(ext / "chill-wave.mp3", 579.4, END, -22, start=120.0, fi=1.5, fo=3.0)

    def sfx(name, t, g, align_end=False):
        c = load(SFX / name)
        if align_end:
            t -= len(c) / SR
        place(c, t, g)

    cg = "cinematic-generated/"
    ui = "ui-clicks-pops/kenney-interface-sounds/"
    sfx(cg + "whoosh-fast.wav", 2.12, -17)
    sfx(cg + "riser-tonal-4s.wav", REVEAL, -13, align_end=True)
    sfx(cg + "boom-cinematic-3-punchy.wav", REVEAL, -8)
    place(cash_ching(), REVEAL + 0.05, -15)
    sfx(cg + "whoosh-medium.wav", 39.3, -20)
    sfx(ui + "drop_002.wav", 61.6, -16)
    sfx(ui + "drop_002.wav", 70.8, -16)
    sfx(cg + "boom-cinematic-1.wav", 117.80, -12)
    place(cash_ching(), 117.86, -16)
    sfx(cg + "whoosh-medium.wav", 167.8, -20)
    for t in (168.0, 308.4, 359.3, 505.0):
        sfx(cg + "whoosh-fast.wav", t - 0.1, -16)
        sfx(ui + "select_002.wav", t + 0.05, -14)
    sfx(cg + "whoosh-medium.wav", 206.97, -20)
    sfx(cg + "whoosh-fast.wav", 278.0, -18)
    sfx(cg + "whoosh-fast.wav", 322.8, -18)
    sfx(ui + "confirmation_002.wav", 325.7, -16)
    sfx(cg + "whoosh-slow.wav", 562.0, -18)
    sfx(cg + "whoosh-medium.wav", 592.0, -20)
    sfx(ui + "select_002.wav", 599.35, -14)

    full = v + mix
    raw = WORK / "mix_raw.wav"
    import wave
    pcm = (np.clip(full, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(raw), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    # gentle clean-up + loudness to -14 LUFS (YouTube)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af",
                    "highpass=f=70,acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120:makeup=1,"
                    "loudnorm=I=-14:TP=-1.5:LRA=9", "-ar", str(SR), "-c:a", "pcm_s16le", str(aout)], check=True)


if __name__ == "__main__":
    vtmp = WORK / "video_only.mp4"
    render_video(str(vtmp))
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(vtmp)], capture_output=True, text=True).stdout)
    if T0 == 0 and T1 is None:
        aout = WORK / "mix.wav"
        render_audio(aout, dur)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(vtmp), "-i", str(aout), "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart",
                        OUT], check=True)
    else:
        subprocess.run(["cp", str(vtmp), OUT], check=True)
    print("done", dur)
