from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Tuple

import cv2
import numpy as np

from .utils import apply_clahe, binarize_adaptive, binarize_otsu, deskew_by_hough, median_denoise, scale_image, to_gray


@dataclass
class PreprocessVariant:
    name: str
    image: np.ndarray
    meta: Dict[str, object]


def generate_variants(image_bgr: np.ndarray, limit: int | None = None) -> Iterable[PreprocessVariant]:
    gray = to_gray(image_bgr)

    # Define option spaces
    deskew_opts = [False, True]
    clahe_opts: List[Tuple[float, Tuple[int, int]]] = [(0.0, (0, 0)), (2.0, (8, 8)), (3.0, (8, 8))]
    blur_opts = [0, 3]
    scale_opts = [1.0, 1.5]
    binarize_methods: List[Tuple[str, Dict[str, object]]] = [
        ("otsu", {"invert": True}),
        ("adaptive_gaussian", {"block_size": 15, "C": 2, "invert": True}),
        ("adaptive_gaussian", {"block_size": 21, "C": 2, "invert": True}),
        ("adaptive_mean", {"block_size": 15, "C": 2, "invert": True}),
    ]

    count = 0

    for do_deskew in deskew_opts:
        base = gray
        if do_deskew:
            base = deskew_by_hough(base)
        for clip_limit, grid_size in clahe_opts:
            g1 = base
            if clip_limit > 0:
                g1 = apply_clahe(g1, clip_limit=clip_limit, tile_grid_size=grid_size)
            for blur_k in blur_opts:
                g2 = g1
                if blur_k >= 3:
                    g2 = median_denoise(g2, ksize=blur_k)
                for scale in scale_opts:
                    g3 = scale_image(g2, scale=scale)
                    for method_name, params in binarize_methods:
                        if method_name == "otsu":
                            bin_img = binarize_otsu(g3, invert=bool(params.get("invert", True)))
                        else:
                            bin_img = binarize_adaptive(
                                g3,
                                method="gaussian" if "gaussian" in method_name else "mean",
                                block_size=int(params.get("block_size", 15)),
                                C=int(params.get("C", 2)),
                                invert=bool(params.get("invert", True)),
                            )
                        variant_name = f"deskew={do_deskew}|clahe={clip_limit}|blur={blur_k}|scale={scale}|bin={method_name}:{params}"
                        meta = {
                            "deskew": do_deskew,
                            "clahe_clip_limit": clip_limit,
                            "clahe_grid": grid_size,
                            "blur": blur_k,
                            "scale": scale,
                            "binarize": {"method": method_name, **params},
                        }
                        yield PreprocessVariant(name=variant_name, image=bin_img, meta=meta)
                        count += 1
                        if limit is not None and count >= limit:
                            return