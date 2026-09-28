# Simple STL viewer for the terminal.
# 1. Put your STL file path below.
# 2. Run:  python stl_view.py
# Press Ctrl+C to quit.

STL_FILE = r"C:\Users\aarsh\Desktop\PARIKALP\NEW_QCAD\testing files\R02B_spider_paired.stl"

import math, os, shutil, struct, sys, time
import numpy as np

CHARS = np.array(list(".,-~:;=!*#$@"))  # dark -> bright


def load_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    # Binary STL: 80-byte header, triangle count, then 50 bytes per triangle
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + n * 50 == len(data):
            dt = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            return np.frombuffer(data, dt, count=n, offset=84)["v"].astype(float)
    # ASCII STL
    text = data.decode(errors="ignore").splitlines()
    verts = [line.split()[1:4] for line in text if line.strip().startswith("vertex")]
    return np.array(verts, dtype=float).reshape(-1, 3, 3)


def render(tris, angle, W, H, radius):
    # Spin around the vertical (Z) axis, then tilt 30 degrees to look from above
    ca, sa = math.cos(angle), math.sin(angle)
    ct, st = math.cos(math.radians(30)), math.sin(math.radians(30))
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    p = tris @ (rx @ rz).T  # x = right, y = depth, z = up
    a, b, c = p[:, 0], p[:, 1], p[:, 2]

    # Brightness from the angle between each face and the light
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.4, -1.0, 0.6])
    light /= np.linalg.norm(light)
    shade = np.round(np.abs(n @ light) * (len(CHARS) - 1)).astype(int)

    # Terminal characters are about twice as tall as wide
    scale = 0.95 * min(W / 4, H / 2) / radius
    sx, sy = p[:, :, 0] * scale * 2, p[:, :, 2] * scale

    # Fill each triangle with enough sample points to cover its screen area
    edge = np.max(np.abs(np.stack([sx - np.roll(sx, 1, 1), sy - np.roll(sy, 1, 1)])), axis=(0, 2))
    steps = np.clip(np.ceil(edge).astype(int) + 1, 1, 300)

    cols, rows, depths, shades = [], [], [], []
    for s in np.unique(steps):
        idx = np.where(steps == s)[0]
        i, j = np.meshgrid(np.arange(s + 1), np.arange(s + 1))
        keep = i + j <= s
        u, v = i[keep] / s, j[keep] / s
        w = 1 - u - v
        pts = (w[None, :, None] * a[idx, None] + u[None, :, None] * b[idx, None]
               + v[None, :, None] * c[idx, None])
        cols.append(np.round(pts[..., 0] * scale * 2 + W / 2).astype(int).ravel())
        rows.append(np.round(H / 2 - pts[..., 2] * scale).astype(int).ravel())
        depths.append(pts[..., 1].ravel())
        shades.append(np.repeat(shade[idx], len(u)))

    col, row = np.concatenate(cols), np.concatenate(rows)
    depth, sh = np.concatenate(depths), np.concatenate(shades)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    col, row, depth, sh = col[ok], row[ok], depth[ok], sh[ok]

    # Draw far points first so near points overwrite them
    order = np.argsort(-depth)
    img = np.full(H * W, " ", dtype="<U1")
    img[(row * W + col)[order]] = CHARS[sh[order]]
    return "\n".join("".join(r) for r in img.reshape(H, W))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else STL_FILE
    tris = load_stl(path)
    pts = tris.reshape(-1, 3)
    tris = tris - (pts.min(0) + pts.max(0)) / 2  # center the model
    radius = np.linalg.norm(tris.reshape(-1, 3), axis=1).max() or 1.0

    os.system("")  # enables colors/cursor codes on Windows
    print("\033[2J\033[?25l", end="")  # clear screen, hide cursor
    angle = 0.0
    try:
        while True:
            W, H = shutil.get_terminal_size()
            frame = render(tris, angle, W, H - 1, radius)
            sys.stdout.write("\033[H" + frame)
            sys.stdout.flush()
            angle += 0.08
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\033[?25h")  # show cursor again