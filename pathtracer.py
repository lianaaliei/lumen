#!/usr/bin/env python3
"""
Vectorized Monte Carlo path tracer -- numpy + PIL only.

Features: global illumination, soft shadows, glass (Fresnel + refraction +
Beer-Lambert tint), fuzzy metals, emissive area lights, depth of field,
sub-pixel anti-aliasing, ACES tone mapping, bloom, vignette.

Usage:
    python pathtracer.py --width 960 --spp 128 --out render.png
"""
import argparse
import time

import numpy as np
from PIL import Image, ImageFilter

EPS = 2e-3
DIFFUSE, METAL, GLASS, EMIT = 0, 1, 2, 3
MAX_BOUNCES = 8
CLAMP = 24.0  # firefly clamp per sample


def norm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def dot(a, b):
    return np.einsum('ij,ij->i', a, b)


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------
def build_scene(seed):
    rng = np.random.default_rng(seed)
    items = []

    def add(c, r, mat, alb=(1, 1, 1), par=0.0, em=(0, 0, 0)):
        items.append((c, r, mat, alb, par, em))

    # hero objects
    add((0, 1, 0), 1.0, GLASS, alb=(0.03, 0.10, 0.05))          # glass ball
    add((-2.3, 0.8, 0.5), 0.8, METAL, alb=(1.0, 0.72, 0.35), par=0.03)  # gold
    add((2.2, 0.7, 0.3), 0.7, DIFFUSE, alb=(0.85, 0.12, 0.10))  # red
    add((0.9, 0.4, 2.1), 0.4, METAL, alb=(0.95, 0.95, 0.97))     # mirror
    add((-1.0, 0.3, 2.3), 0.3, DIFFUSE, alb=(0.15, 0.35, 0.90))  # blue
    add((1.9, 0.25, 2.4), 0.25, EMIT, em=(0.6, 4.5, 4.0))        # teal glow
    # lights (big = soft shadows)
    add((-5, 6, 4), 2.0, EMIT, em=(13, 11, 8.5))                 # warm key
    add((6, 4, -2), 1.2, EMIT, em=(2.5, 5, 10))                  # cool rim

    # scatter small spheres in the background
    placed = [(c[0], c[2], r) for c, r, *_ in items if c[1] < 3]
    tries = 0
    while len(placed) < 6 + 16 and tries < 2000:
        tries += 1
        x, z = rng.uniform(-8, 8), rng.uniform(-10, 1.0)
        r = rng.uniform(0.18, 0.5)
        if any(np.hypot(x - px, z - pz) < r + pr + 0.15 for px, pz, pr in placed):
            continue
        placed.append((x, z, r))
        u = rng.random()
        col = tuple(rng.uniform(0.15, 0.95, 3))
        if u < 0.5:
            add((x, r, z), r, DIFFUSE, alb=col)
        elif u < 0.8:
            add((x, r, z), r, METAL, alb=col, par=rng.uniform(0, 0.25))
        else:
            add((x, r, z), r, GLASS, alb=(0.02, 0.06, 0.08))

    f32 = np.float32
    return dict(
        C=np.array([i[0] for i in items], f32),
        R=np.array([i[1] for i in items], f32),
        MAT=np.array([i[2] for i in items], np.int32),
        ALB=np.array([i[3] for i in items], f32),
        PAR=np.array([i[4] for i in items], f32),
        EM=np.array([i[5] for i in items], f32),
    )


def sky(d):
    t = np.clip(0.5 * (d[:, 1] + 1), 0, 1)[:, None]
    a = np.array([1.0, 0.95, 0.9], np.float32)
    b = np.array([0.3, 0.5, 0.9], np.float32)
    return ((1 - t) * a + t * b) * 0.45


# --------------------------------------------------------------------------
# Integrator: every ray in the image advances one bounce per loop iteration
# --------------------------------------------------------------------------
def trace(o, d, sc, rng):
    C, R, MAT, ALB, PAR, EM = (sc[k] for k in ("C", "R", "MAT", "ALB", "PAR", "EM"))
    S = len(R)
    n = len(o)
    L = np.zeros((n, 3), np.float32)
    idx = np.arange(n)
    thr = np.ones((n, 3), np.float32)
    up_y = np.array([0, 1, 0], np.float32)
    lights = np.nonzero(MAT == EMIT)[0]
    blockers = np.nonzero(MAT != EMIT)[0]
    spec = np.ones(n, bool)   # True -> emission seen by BSDF sampling counts

    for bounce in range(MAX_BOUNCES):
        m = len(idx)
        if m == 0:
            break

        # ---- nearest intersection --------------------------------------
        tb = np.full(m, np.inf, np.float32)
        obj = np.full(m, -1, np.int32)
        for i in range(S):
            oc = o - C[i]
            b = dot(oc, d)
            c = dot(oc, oc) - R[i] * R[i]
            disc = b * b - c
            sq = np.sqrt(np.maximum(disc, 0))
            t1 = -b - sq
            t = np.where(t1 > EPS, t1, -b + sq)
            hit = (disc > 0) & (t > EPS) & (t < tb)
            tb = np.where(hit, t, tb)
            obj = np.where(hit, i, obj)
        dy = d[:, 1]
        tp = -o[:, 1] / np.where(np.abs(dy) > 1e-8, dy, -1e-8)
        hp = (dy < 0) & (tp > EPS) & (tp < tb)
        tb = np.where(hp, tp, tb)
        obj = np.where(hp, S, obj)

        # ---- escaped rays pick up the sky ------------------------------
        miss = obj < 0
        if miss.any():
            L[idx[miss]] += thr[miss] * sky(d[miss])
        k = ~miss
        idx, o, d, thr, tb, obj, spec = idx[k], o[k], d[k], thr[k], tb[k], obj[k], spec[k]
        m = len(idx)
        if m == 0:
            break

        # ---- surface info ----------------------------------------------
        P = o + d * tb[:, None]
        isS = obj < S
        oi = np.minimum(obj, S - 1)
        N = np.where(isS[:, None], (P - C[oi]) / R[oi][:, None], up_y)
        front = dot(d, N) < 0
        N = np.where(front[:, None], N, -N)
        mat = np.where(isS, MAT[oi], DIFFUSE)
        chk = ((np.floor(P[:, 0]) + np.floor(P[:, 2])) % 2)[:, None]
        floor_alb = chk * np.array([.72, .72, .70], np.float32) + \
            (1 - chk) * np.array([.12, .12, .14], np.float32)
        alb = np.where(isS[:, None], ALB[oi], floor_alb)

        # ---- emitters end the path -------------------------------------
        em = mat == EMIT
        ea = em & spec
        if ea.any():
            L[idx[ea]] += thr[ea] * EM[oi[ea]]
        k = ~em
        idx, o, d, thr, tb, spec = idx[k], o[k], d[k], thr[k], tb[k], spec[k]
        P, N, front, mat, alb, oi = P[k], N[k], front[k], mat[k], alb[k], oi[k]
        m = len(idx)
        if m == 0:
            break

        # ---- scatter ---------------------------------------------------
        ru = norm(rng.standard_normal((m, 3)).astype(np.float32))
        d_dif = norm(N + ru)                                   # Lambertian

        refl = d - 2 * dot(d, N)[:, None] * N
        d_met = norm(refl + PAR[oi][:, None] * ru)             # fuzzy metal
        met_bad = dot(d_met, N) <= 0

        cosi = np.clip(-dot(d, N), 0, 1)                       # glass
        eta = np.where(front, 1 / 1.5, 1.5).astype(np.float32)
        sin2t = eta * eta * (1 - cosi * cosi)
        schlick = 0.04 + 0.96 * (1 - cosi) ** 5
        do_refl = (sin2t > 1) | (rng.random(m) < schlick)
        refr = eta[:, None] * d + (eta * cosi - np.sqrt(np.maximum(1 - sin2t, 0)))[:, None] * N
        d_gl = np.where(do_refl[:, None], refl, norm(refr))

        is_m, is_g = mat == METAL, mat == GLASS
        nd = np.where(is_m[:, None], d_met, np.where(is_g[:, None], d_gl, d_dif))
        side = np.where(is_g & ~do_refl, -1.0, 1.0).astype(np.float32)
        no = P + N * (EPS * side)[:, None]

        # ---- next-event estimation: sample the sphere lights directly ---
        dm = np.nonzero(mat == DIFFUSE)[0]
        if len(dm) and len(lights):
            nd_ = len(dm)
            po = P[dm] + N[dm] * EPS
            base = thr[dm] * alb[dm]
            for j in lights:
                to = C[j] - po
                dist = np.linalg.norm(to, axis=1)
                w = to / dist[:, None]
                cosmax = np.sqrt(1 - np.minimum((R[j] / dist) ** 2, 0.999))
                cost = 1 - rng.random(nd_).astype(np.float32) * (1 - cosmax)
                sint = np.sqrt(np.maximum(1 - cost * cost, 0))
                phi = 2 * np.pi * rng.random(nd_)
                ax = np.where(np.abs(w[:, :1]) > 0.9, np.array([0, 1, 0], np.float32),
                              np.array([1, 0, 0], np.float32))
                uu = norm(np.cross(ax, w))
                vv = np.cross(w, uu)
                ld = (uu * (sint * np.cos(phi))[:, None] + vv * (sint * np.sin(phi))[:, None]
                      + w * cost[:, None])
                cosn = dot(N[dm], ld)
                ok = (cosn > 0) & (dist > R[j] * 1.001)
                tmax = dist - R[j]
                for i in blockers:                      # shadow ray
                    oc = po - C[i]
                    b = dot(oc, ld)
                    disc = b * b - (dot(oc, oc) - R[i] * R[i])
                    t = -b - np.sqrt(np.maximum(disc, 0))
                    ok &= ~((disc > 0) & (t > EPS) & (t < tmax))
                omega = 2 * np.pi * (1 - cosmax)
                contrib = base * EM[j] * (cosn * omega / np.pi)[:, None]
                L[idx[dm[ok]]] += contrib[ok]

        # Beer-Lambert absorption while travelling inside glass
        tm = np.where(is_g[:, None], np.exp(-alb * (tb * ~front)[:, None]), alb)
        thr = thr * tm
        term = is_m & met_bad

        # ---- Russian roulette ------------------------------------------
        if bounce >= 3:
            p = np.clip(thr.max(axis=1), 0.05, 1.0)
            term |= rng.random(m) >= p
            thr = thr / p[:, None]

        k = ~term
        spec = (mat != DIFFUSE)[k]
        idx = idx[k]
        o = no[k].astype(np.float32)
        d = nd[k].astype(np.float32)
        thr = thr[k].astype(np.float32)

    return L


# --------------------------------------------------------------------------
# Camera + progressive rendering
# --------------------------------------------------------------------------
def develop(acc, n, W, H, path):
    x = (acc / n).reshape(H, W, 3)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14              # ACES (Narkowicz)
    ldr = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1) ** (1 / 2.2)

    bright = Image.fromarray((np.clip((ldr - 0.75) * 4, 0, 1) * 255).astype(np.uint8))
    bloom = np.asarray(bright.filter(ImageFilter.GaussianBlur(W / 70)), np.float32) / 255
    ldr = ldr + 0.6 * bloom

    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) * 2
    ldr = ldr * (1 - 0.45 * r2)[..., None]

    Image.fromarray((np.clip(ldr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def render(W, H, spp, seed, out):
    rng = np.random.default_rng(seed)
    sc = build_scene(3)

    cam = np.array([0.6, 1.6, 6.4], np.float32)
    target = np.array([0, 0.85, 0], np.float32)
    fwd = norm(target - cam)
    right = norm(np.cross(fwd, [0, 1, 0]))
    up = np.cross(right, fwd)
    th = np.tan(np.radians(34) / 2)
    focus = np.linalg.norm(np.array([0, 1, 0]) - cam)
    aperture = 0.09

    acc = np.zeros((H * W, 3), np.float64)
    t0 = time.time()
    for s in range(spp):
        px = (np.arange(W)[None, :] + rng.random((H, W))) / W
        py = (np.arange(H)[:, None] + rng.random((H, W))) / H
        u = (2 * px - 1) * (W / H) * th
        v = (1 - 2 * py) * th
        dirs = norm((u[..., None] * right + v[..., None] * up + fwd).reshape(-1, 3))

        fp = cam + dirs * (focus / (dirs @ fwd))[:, None]      # depth of field
        rr = aperture * np.sqrt(rng.random(H * W))
        ang = 2 * np.pi * rng.random(H * W)
        org = cam + right * (rr * np.cos(ang))[:, None] + up * (rr * np.sin(ang))[:, None]
        dr = norm(fp - org)

        L = trace(org.astype(np.float32), dr.astype(np.float32), sc, rng)
        acc += np.nan_to_num(np.minimum(L, CLAMP))

        done = s + 1
        el = time.time() - t0
        print(f"pass {done}/{spp}  {el:5.0f}s  eta {el / done * (spp - done):5.0f}s", flush=True)
        if done % 8 == 0 or done == spp:
            develop(acc, done, W, H, out)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="render.png")
    a = ap.parse_args()
    render(a.width, a.width * 9 // 16, a.spp, a.seed, a.out)
