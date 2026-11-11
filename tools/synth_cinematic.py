"""Synthesize original cinematic tension SFX (risers, booms, sub drops, braams...).

Everything here is generated from scratch (noise + oscillators), so the output
is an original work and can be dedicated to the public domain (CC0).

usage: python synth_cinematic.py OUT_DIR
"""
import sys
import os
import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)


def rng(seed):
    return np.random.default_rng(seed)


def tvec(dur):
    return np.arange(int(round(dur * SR))) / SR


def stft_filter(x, gain_fn, nperseg=2048):
    """Time-varying filter: gain_fn(freqs[:,None], frame_times[None,:]) -> gain matrix."""
    nover = nperseg * 3 // 4
    f, t, Z = signal.stft(x, SR, nperseg=nperseg, noverlap=nover, boundary="even")
    g = gain_fn(np.maximum(f, 1.0)[:, None], t[None, :])
    _, y = signal.istft(Z * g, SR, nperseg=nperseg, noverlap=nover, boundary=True)
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def bandpass_gain(center_fn, bw_oct):
    def g(f, t):
        fc = center_fn(t)
        return np.exp(-0.5 * ((np.log2(f) - np.log2(fc)) / bw_oct) ** 2)
    return g


def lowpass_gain(cut_fn, order=4):
    def g(f, t):
        return 1.0 / np.sqrt(1.0 + (f / cut_fn(t)) ** (2 * order))
    return g


def butter(x, kind, freq, order=4):
    sos = signal.butter(order, freq, btype=kind, fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def sine_sweep(freq_curve):
    phase = 2 * np.pi * np.cumsum(freq_curve) / SR
    return np.sin(phase)


def reverb(stereo, seconds=3.0, tau=0.7, wet=0.3, seed=0, lp=5000):
    r = rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    out = np.zeros((stereo.shape[0] + n - 1, 2))
    for ch in range(2):
        ir = r.standard_normal(n) * np.exp(-tt / tau)
        ir = butter(ir, "low", lp, 2)
        ir[: int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))  # pre-delay softening
        ir /= np.sqrt(np.sum(ir ** 2))
        out[:, ch] = signal.fftconvolve(stereo[:, ch], ir)
    dry = np.zeros_like(out)
    dry[: stereo.shape[0]] = stereo
    return dry * (1 - wet) + out * wet


def fade(x, fin=0.0, fout=0.0):
    x = x.copy()
    a, b = int(fin * SR), int(fout * SR)
    if a:
        x[:a] *= np.linspace(0, 1, a)[:, None] if x.ndim == 2 else np.linspace(0, 1, a)
    if b:
        x[-b:] *= np.linspace(1, 0, b)[:, None] if x.ndim == 2 else np.linspace(1, 0, b)
    return x


def trim_tail(x, thresh_db=-60, keep=0.05):
    """Cut trailing near-silence (e.g. reverb tails) and fade the last bit."""
    lvl = np.max(np.abs(x), axis=1)
    thr = 10 ** (thresh_db / 20) * np.max(lvl)
    idx = np.nonzero(lvl > thr)[0]
    end = min(len(x), (idx[-1] if len(idx) else len(x)) + int(keep * SR))
    return fade(x[:end], fout=min(0.25, end / SR / 4))


def stereo(mono_l, mono_r=None):
    return np.stack([mono_l, mono_l if mono_r is None else mono_r], axis=1)


def save(name, x, peak_db=-1.0):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = stereo(x)
    x = x - np.mean(x, axis=0)  # remove DC
    true_peak = np.max(np.abs(signal.resample_poly(x, 4, 1, axis=0)))
    x = x / max(true_peak, np.max(np.abs(x))) * 10 ** (peak_db / 20)
    tpdf = (rng(123).random(x.shape) - rng(456).random(x.shape)) / 32768.0
    pcm = np.clip(np.round((x + tpdf) * 32767), -32768, 32767).astype(np.int16)
    wavfile.write(os.path.join(OUT, name + ".wav"), SR, pcm)
    print(f"{name:34s} {len(x) / SR:6.2f}s")


# ---------------------------------------------------------------- risers
def riser(dur, seed, f0=180, f1=9000, tonal=True):
    t = tvec(dur)
    p = t / dur
    r = rng(seed)
    centre = lambda tt: f0 * (f1 / f0) ** np.clip(tt / dur, 0, 1) ** 1.4
    chans = []
    for ch in range(2):
        n = r.standard_normal(len(t))
        bp = bandpass_gain(centre, 0.9)
        comp = lambda f, tt: bp(f, tt) * (centre(tt) / f1) ** -0.4  # keep low start audible
        chans.append(stft_filter(n, comp))
    noise = np.stack(chans, axis=1)
    env = 0.06 + 0.94 * (np.exp(2.8 * p) - 1) / (np.exp(2.8) - 1)
    out = noise / np.max(np.abs(noise)) * env[:, None]
    if tonal:
        # gliding detuned oscillators with accelerating tremolo pulses
        base = 110 * (4.0) ** (p ** 1.6)  # two octaves up
        trem_rate = 3 + 21 * p ** 2
        trem = 0.55 + 0.45 * np.sin(2 * np.pi * np.cumsum(trem_rate) / SR)
        tone = np.zeros((len(t), 2))
        for k, det in enumerate((-0.006, 0.0, 0.006)):
            for mult, amp in ((1, 1.0), (2, 0.5), (1.5, 0.35)):
                s = sine_sweep(base * mult * (1 + det))
                tone[:, k % 2] += amp * s
                tone[:, (k + 1) % 2] += 0.5 * amp * s
        tone = tone / np.max(np.abs(tone)) * (env ** 1.2 * trem)[:, None]
        out = out + 0.55 * tone
    out = np.tanh(1.6 * out)
    out[-int(0.006 * SR):] *= np.linspace(1, 0, int(0.006 * SR))[:, None]
    return out


# ---------------------------------------------------------------- impacts
def boom(seed, sub_f=(90, 38), tail=3.5, bright=1.0):
    dur = 4.0
    t = tvec(dur)
    r = rng(seed)
    f = sub_f[1] + (sub_f[0] - sub_f[1]) * np.exp(-t / 0.12)
    sub = sine_sweep(f) * np.exp(-t / 0.9) * np.minimum(1, t / 0.002)
    body = butter(r.standard_normal(len(t)), "low", 260) * np.exp(-t / 0.22)
    body /= np.max(np.abs(body))
    click = butter(r.standard_normal(len(t)), "high", 1800) * np.exp(-t / 0.008)
    click /= np.max(np.abs(click))
    mono = 1.0 * sub + 0.7 * body + 0.18 * bright * click
    mono = np.tanh(1.8 * mono)
    st = fade(stereo(mono), fout=1.2)  # let the dry layer die away before the buffer ends
    st = reverb(st, seconds=tail, tau=0.75, wet=0.32, seed=seed + 1, lp=3500)
    return trim_tail(st)


def sub_drop(dur, f0=160, f1=28, seed=0):
    t = tvec(dur)
    f = f1 + (f0 - f1) * np.exp(-t / (dur * 0.28))
    s = sine_sweep(f) * np.minimum(1, t / 0.004) * np.exp(-t / (dur * 0.45))
    s = np.tanh(2.2 * s)  # harmonics so it is audible on phone speakers
    click = butter(rng(seed).standard_normal(len(t)), "low", 900) * np.exp(-t / 0.015)
    s = s + 0.25 * click / np.max(np.abs(click))
    return fade(stereo(s), fout=0.05)


def braam(seed, root=55.0, dur=4.5):
    t = tvec(dur)
    r = rng(seed)
    notes = [(root, 1.0), (root * 1.5, 0.6), (root * 2, 0.7), (root * 3, 0.3)]
    sig = np.zeros((len(t), 2))
    for fr, amp in notes:
        for ch, cents in ((0, -9), (1, 9), (0, 3), (1, -3)):
            ff = fr * 2 ** (cents / 1200) * (1 + 0.002 * np.sin(2 * np.pi * 0.3 * t + r.random() * 6))
            ph = np.cumsum(ff) / SR
            saw = 2 * (ph - np.floor(ph + 0.5))
            sig[:, ch] += amp * saw
    cut = lambda tt: 140 + 1700 * np.exp(-np.maximum(tt - 0.18, 0) / 1.1) * np.minimum(1, tt / 0.18)
    for ch in range(2):
        sig[:, ch] = stft_filter(sig[:, ch], lowpass_gain(cut, 3))
    env = np.minimum(1, t / 0.07) * np.exp(-np.maximum(t - 0.2, 0) / 1.6)
    growl = 1 + 0.12 * np.sin(2 * np.pi * 28 * t)
    sig = sig / np.max(np.abs(sig)) * (env * growl)[:, None]
    sig = fade(np.tanh(2.0 * sig), fout=1.0)
    sig = reverb(sig, seconds=3.5, tau=0.9, wet=0.3, seed=seed + 7, lp=3000)
    return trim_tail(sig)


# ---------------------------------------------------------------- whooshes
def whoosh(dur, seed, f_lo=350, f_hi=3800, pan=(0.12, 0.88)):
    t = tvec(dur)
    p = t / dur
    r = rng(seed)
    shape = np.sin(np.pi * np.clip(p, 0, 1)) ** 1.5
    centre = lambda tt: f_lo * (f_hi / f_lo) ** (np.sin(np.pi * np.clip(tt / dur, 0, 1)) ** 1.5)
    n = stft_filter(r.standard_normal(len(t)), bandpass_gain(centre, 0.7), nperseg=1024)
    env = (p ** 1.4) * (1 - p) ** 0.9
    env /= env.max()
    m = n / np.max(np.abs(n)) * env
    th = (pan[0] + (pan[1] - pan[0]) * p) * np.pi / 2
    st = np.stack([m * np.cos(th), m * np.sin(th)], axis=1)
    st = reverb(st, seconds=1.2, tau=0.25, wet=0.18, seed=seed + 3, lp=7000)
    return trim_tail(st)


def reverse_swell(dur, seed):
    t = tvec(dur)
    r = rng(seed)
    cym = butter(r.standard_normal(len(t)), "high", 2500) * np.exp(-t / (dur * 0.3))
    low = butter(r.standard_normal(len(t)), "low", 400) * np.exp(-t / (dur * 0.2))
    st = stereo(cym + 0.5 * low, butter(r.standard_normal(len(t)), "high", 2500) * np.exp(-t / (dur * 0.3)) + 0.5 * low)
    st = reverb(st, seconds=dur * 0.8, tau=dur * 0.2, wet=0.45, seed=seed + 5, lp=9000)[: len(t)]
    st = st[::-1]
    st[-int(0.004 * SR):] *= np.linspace(1, 0, int(0.004 * SR))[:, None]
    return fade(st, fin=dur * 0.3)


# ---------------------------------------------------------------- beds
def tension_drone(dur=30.0, seed=11):
    t = tvec(dur)
    r = rng(seed)
    out = np.zeros((len(t), 2))
    partials = [(55.0, 1.0), (58.27, 0.55), (82.41, 0.5), (116.54, 0.3), (164.81, 0.18), (233.08, 0.08)]
    for fr, amp in partials:
        for ch in range(2):
            lfo = 0.6 + 0.4 * np.sin(2 * np.pi * (0.04 + 0.08 * r.random()) * t + 6 * r.random())
            ff = fr * (1 + 0.003 * np.sin(2 * np.pi * 0.11 * t + 6 * r.random()))
            out[:, ch] += amp * lfo * sine_sweep(ff)
    rumble = np.stack([butter(r.standard_normal(len(t)), "band", (60, 260), 2) for _ in range(2)], axis=1)
    out = out / np.max(np.abs(out)) + 0.35 * rumble / np.max(np.abs(rumble))
    eerie = np.sin(2 * np.pi * np.cumsum(1760 * (1 + 0.004 * np.sin(2 * np.pi * 5 * t))) / SR)
    eerie *= 0.025 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t))
    out += stereo(eerie, -eerie)
    out = np.tanh(1.3 * out)
    return fade(out, fin=3.0, fout=3.0)


def accelerating_pulse(dur=12.0, bpm0=60, bpm1=160, seed=21):
    t = tvec(dur)
    out = np.zeros(len(t))
    hit_t = tvec(0.35)
    thump = sine_sweep(45 + 60 * np.exp(-hit_t / 0.03)) * np.exp(-hit_t / 0.09)
    thump = np.tanh(2.0 * thump)
    pos = 0.0
    while pos < dur - 0.35:
        p = pos / dur
        bpm = bpm0 * (bpm1 / bpm0) ** (p ** 1.3)
        i = int(pos * SR)
        gain = 0.55 + 0.45 * p
        out[i:i + len(thump)] += gain * thump
        pos += 60.0 / bpm
    st = reverb(stereo(out), seconds=1.5, tau=0.3, wet=0.2, seed=seed)[: len(t)]
    return fade(st, fout=0.05)


def riser_into_boom(seed):
    rs = riser(5.0, seed)
    bm = boom(seed + 50)
    rs = rs / np.max(np.abs(rs)) * 0.8
    bm = bm / np.max(np.abs(bm))
    return np.concatenate([rs, bm], axis=0)


if __name__ == "__main__":
    save("riser-noise-3s", riser(3.0, 1, tonal=False))
    save("riser-noise-6s", riser(6.0, 2, tonal=False))
    save("riser-tonal-4s", riser(4.0, 3))
    save("riser-tonal-8s", riser(8.0, 4))
    save("riser-tonal-12s", riser(12.0, 5))
    save("riser-into-boom-5s", riser_into_boom(6))
    save("reverse-swell-2s", reverse_swell(2.0, 7))
    save("reverse-swell-4s", reverse_swell(4.0, 8))
    save("boom-cinematic-1", boom(9))
    save("boom-cinematic-2-deep", boom(10, sub_f=(70, 30), tail=4.5, bright=0.5))
    save("boom-cinematic-3-punchy", boom(12, sub_f=(130, 48), tail=2.5, bright=1.6))
    save("sub-drop-short", sub_drop(1.6, seed=13))
    save("sub-drop-long", sub_drop(3.5, f0=140, f1=24, seed=14))
    save("braam-low", braam(15, root=55.0))
    save("braam-dark", braam(16, root=49.0))
    save("whoosh-fast", whoosh(0.55, 17))
    save("whoosh-medium", whoosh(1.1, 18, pan=(0.85, 0.15)))
    save("whoosh-slow", whoosh(2.0, 19, f_lo=250, f_hi=2600))
    save("tension-drone-30s", tension_drone(), peak_db=-3.0)
    save("accelerating-pulse-12s", accelerating_pulse())
