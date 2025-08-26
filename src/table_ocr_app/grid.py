from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class Grid:
    rows: int
    cols: int
    # List of (x, y, w, h) per cell in row-major order
    cells: List[Tuple[int, int, int, int]]


def _extract_lines(binary: np.ndarray, is_horizontal: bool, kernel_frac: int) -> np.ndarray:
    h, w = binary.shape[:2]
    if is_horizontal:
        kernel_length = max(8, w // kernel_frac)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    else:
        kernel_length = max(8, h // kernel_frac)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    temp = cv2.erode(binary, kernel, iterations=1)
    lines = cv2.dilate(temp, kernel, iterations=1)
    return lines


def _find_line_positions(lines_img: np.ndarray, is_horizontal: bool, thr_ratio: float) -> List[int]:
    # Sum projections and pick peaks, then merge contiguous runs
    if is_horizontal:
        profile = lines_img.sum(axis=1)
    else:
        profile = lines_img.sum(axis=0)
    max_val = float(profile.max()) if profile.size else 0.0
    if max_val <= 0.0:
        return []
    thr = max_val * thr_ratio
    positions: List[int] = []
    in_run = False
    run_start = 0
    for idx, val in enumerate(profile):
        if val >= thr and not in_run:
            in_run = True
            run_start = idx
        elif val < thr and in_run:
            in_run = False
            run_end = idx - 1
            positions.append((run_start + run_end) // 2)
    if in_run:
        positions.append((run_start + len(profile) - 1) // 2)
    return positions


def _build_grid_from_positions(binary: np.ndarray, y_positions: List[int], x_positions: List[int]) -> Grid:
    h, w = binary.shape[:2]
    if not y_positions or y_positions[0] > 5:
        y_positions = [0] + y_positions
    if not y_positions or y_positions[-1] < h - 5:
        y_positions = y_positions + [h - 1]
    if not x_positions or x_positions[0] > 5:
        x_positions = [0] + x_positions
    if not x_positions or x_positions[-1] < w - 5:
        x_positions = x_positions + [w - 1]

    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))

    cells: List[Tuple[int, int, int, int]] = []
    for r in range(len(y_positions) - 1):
        y1, y2 = y_positions[r], y_positions[r + 1]
        for c in range(len(x_positions) - 1):
            x1, x2 = x_positions[c], x_positions[c + 1]
            x = max(0, x1)
            y = max(0, y1)
            w_box = max(1, x2 - x1)
            h_box = max(1, y2 - y1)
            cells.append((x, y, w_box, h_box))

    return Grid(rows=len(y_positions) - 1, cols=len(x_positions) - 1, cells=cells)


def detect_grid_from_binary(binary: np.ndarray) -> Grid:
    horiz = _extract_lines(binary, is_horizontal=True, kernel_frac=30)
    vert = _extract_lines(binary, is_horizontal=False, kernel_frac=30)
    y_positions = _find_line_positions(horiz, is_horizontal=True, thr_ratio=0.5)
    x_positions = _find_line_positions(vert, is_horizontal=False, thr_ratio=0.5)
    return _build_grid_from_positions(binary, y_positions, x_positions)


def detect_candidate_grids(binary: np.ndarray) -> List[Tuple[Grid, Dict[str, object]]]:
    # Try a sweep of kernel sizes and threshold ratios
    kernel_fracs = [20, 30, 40]
    thr_ratios = [0.3, 0.5, 0.7]
    candidates: List[Tuple[Grid, Dict[str, object]]] = []
    for kf in kernel_fracs:
        horiz = _extract_lines(binary, is_horizontal=True, kernel_frac=kf)
        vert = _extract_lines(binary, is_horizontal=False, kernel_frac=kf)
        for thr in thr_ratios:
            y_positions = _find_line_positions(horiz, is_horizontal=True, thr_ratio=thr)
            x_positions = _find_line_positions(vert, is_horizontal=False, thr_ratio=thr)
            if len(y_positions) >= 2 and len(x_positions) >= 2:
                grid = _build_grid_from_positions(binary, y_positions, x_positions)
                meta = {"kernel_frac": kf, "thr_ratio": thr, "rows": grid.rows, "cols": grid.cols}
                candidates.append((grid, meta))
    # Deduplicate by (rows, cols)
    uniq: Dict[Tuple[int, int], Tuple[Grid, Dict[str, object]]] = {}
    for grid, meta in candidates:
        key = (grid.rows, grid.cols)
        if key not in uniq:
            uniq[key] = (grid, meta)
    return list(uniq.values())


def binary_from_bgr_for_grid(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Use Otsu on inverted to highlight dark lines as white
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return th