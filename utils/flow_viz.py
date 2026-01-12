import numpy as np
import cv2

"""
光流颜色化（Middlebury 颜色编码）
来自常见实现（flow_to_image）
输入 flow: HxWx2
输出 RGB 图像 HxWx3 (uint8)
"""

def compute_color(u, v):
    """
    u, v: numpy arrays
    """
    UNKNOWN_FLOW_THRESH = 1e7
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    u[nan_u] = 0
    v[nan_v] = 0

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    eps = 1e-5
    u = u/(maxrad + eps)
    v = v/(maxrad + eps)

    # color wheel
    colorwheel = make_color_wheel()
    ncols = colorwheel.shape[0]
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(int)
    k1 = (k0 + 1) % ncols
    f = fk - k0
    img = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)

    for i in range(3):
        col0 = colorwheel[k0, i] / 255.0
        col1 = colorwheel[k1, i] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] *= 0.75
        img[:, :, i] = np.floor(255 * col).astype(np.uint8)
    return img

def make_color_wheel():
    # color wheel as in Middlebury
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_to_image(flow):
    """
    flow: H x W x 2
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    img = compute_color(u.copy(), v.copy())
    return img

def write_flow_png(flow, path):
    img = flow_to_image(flow)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
