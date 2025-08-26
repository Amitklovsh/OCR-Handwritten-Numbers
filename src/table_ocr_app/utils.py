import os
import time
from contextlib import contextmanager
from typing import Iterator, Tuple

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@contextmanager
def time_block() -> Iterator[float]:
    start = time.perf_counter()
    yield start
    # context manager ensures caller can compute elapsed via now - start


def read_image_bgr(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # Fallback if imdecode fails due to path encoding issues
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def deskew_by_hough(gray: np.ndarray, angle_search_deg: Tuple[int, int] = (-10, 10)) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=150)
    if lines is None or len(lines) == 0:
        return gray
    angles = []
    for rho_theta in lines[:200]:
        rho, theta = rho_theta[0]
        angle_deg = (theta * 180.0 / np.pi) - 90.0
        if angle_search_deg[0] <= angle_deg <= angle_search_deg[1]:
            angles.append(angle_deg)
    if not angles:
        return gray
    median_angle = float(np.median(angles))
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def binarize_otsu(gray: np.ndarray, invert: bool = True) -> np.ndarray:
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, th = cv2.threshold(gray, 0, 255, thresh_type | cv2.THRESH_OTSU)
    return th


def binarize_adaptive(gray: np.ndarray, method: str = "gaussian", block_size: int = 15, C: int = 2, invert: bool = True) -> np.ndarray:
    if method == "mean":
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    return cv2.adaptiveThreshold(gray, 255, adaptive_method, thresh_type, block_size, C)


def median_denoise(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = max(3, ksize | 1)
    return cv2.medianBlur(gray, ksize)


def scale_image(gray: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if scale == 1.0:
        return gray
    h, w = gray.shape[:2]
    return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)