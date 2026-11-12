"""Precompute helpers for edit_youtube_paid_me.py.

  seg   : person masks (u2net_human_seg) for the talking-head shots -> masks_<a>.npz
  stab  : frame-to-frame shake for the top-down paper shots -> stab_<a>.npy (dx, dy, da) correction

usage: python tools/precompute_paid_me.py seg|stab RAW WORKDIR MODEL(seg only)
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

FPS = 30
SW, SH = 2560, 1440


def frames(raw, a, b, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", raw, "-an",
                          "-vf", f"fps={FPS},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)


def seg(raw, work, model, a, b):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    out, prev = [], None
    for i, fr in enumerate(frames(raw, a, b, 640, 360)):
        x = cv2.resize(fr, (320, 320), interpolation=cv2.INTER_AREA)[:, :, ::-1] / 255.0
        x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        m = sess.run(None, {inp: x})[0][0, 0]
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)
        m = cv2.resize(m, (640, 360), interpolation=cv2.INTER_LINEAR)
        prev = m if prev is None else 0.55 * m + 0.45 * prev  # temporal smoothing against flicker
        out.append((np.clip(prev, 0, 1) * 255).astype(np.uint8))
        if i % 300 == 0:
            print("seg", a, i, flush=True)
    np.savez_compressed(work / f"masks_{a:.2f}.npz", m=np.stack(out))


def stab(raw, work, a, b):
    w, h = 960, 540
    prev, traj = None, [(0.0, 0.0, 0.0)]
    for fr in frames(raw, a, b, w, h):
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            pts = cv2.goodFeaturesToTrack(prev, 400, 0.01, 12)
            dx = dy = da = 0.0
            if pts is not None:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None)
                ok = st.ravel() == 1
                if ok.sum() > 12:
                    M, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC,
                                                       ransacReprojThreshold=1.5)
                    if M is not None:
                        dx, dy, da = M[0, 2], M[1, 2], np.arctan2(M[1, 0], M[0, 0])
            x, y, r = traj[-1]
            traj.append((x + dx, y + dy, r + da))
        prev = g
    traj = np.array(traj[: max(1, len(traj))])
    k = 90  # wide smoothing (~3 s): effectively locks the camera, removes all jitter
    pad = np.pad(traj, ((3 * k, 3 * k), (0, 0)), mode="edge")
    ker = cv2.getGaussianKernel(6 * k + 1, k).ravel()
    smooth = np.stack([np.convolve(pad[:, i], ker, mode="valid") for i in range(3)], 1)
    corr = smooth - traj
    corr[:, :2] *= SW / w  # to source pixels
    np.save(work / f"stab_{a:.2f}.npy", corr)
    print("stab", a, len(corr), np.abs(corr).max(0), flush=True)


if __name__ == "__main__":
    mode, raw, work = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    if mode == "seg":
        for a, b in ((111.13, 167.80), (592.03, 609.45)):
            seg(raw, work, sys.argv[4], a, b)
    else:
        for a, b in ((39.27, 111.13), (167.80, 206.87)):
            stab(raw, work, a, b)
