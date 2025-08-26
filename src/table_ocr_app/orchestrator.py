from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from .grid import Grid, binary_from_bgr_for_grid, detect_candidate_grids, detect_grid_from_binary
from .ocr_engines import OcrResult, OcrRunner
from .preprocess import PreprocessVariant, generate_variants
from .utils import read_image_bgr


@dataclass
class RunConfig:
    image_path: str
    max_variants: int | None
    engines: List[str]  # ["tesseract", "easyocr"]
    tesseract_psm: List[int]
    tesseract_oem: List[int]
    allowlist: str


@dataclass
class OutputTable:
    table: List[List[str]]
    sheet_meta: Dict[str, object]


class RecognitionOrchestrator:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.ocr = OcrRunner(
            use_tesseract=("tesseract" in config.engines),
            use_easyocr=("easyocr" in config.engines),
        )

    def _segment_grids(self, image_bgr: np.ndarray, variant: PreprocessVariant) -> List[Tuple[Grid, Dict[str, object]]]:
        grid_binary = binary_from_bgr_for_grid(image_bgr)
        candidates = detect_candidate_grids(grid_binary)
        if not candidates:
            # Fallback to default detection
            g = detect_grid_from_binary(grid_binary)
            candidates = [(g, {"fallback": True, "rows": g.rows, "cols": g.cols})]
        return candidates

    def _extract_cell_image(self, source_gray_or_bin: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        cell = source_gray_or_bin[y : y + h, x : x + w]
        cell = cv2.copyMakeBorder(cell, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        return cell

    def run(self) -> List[OutputTable]:
        image_bgr = read_image_bgr(self.config.image_path)
        outputs: List[OutputTable] = []

        for variant in generate_variants(image_bgr, limit=self.config.max_variants):
            start_variant = time.perf_counter()

            grid_candidates = self._segment_grids(image_bgr, variant)

            # Engines/options sweep
            engine_options_list: List[Tuple[str, Dict[str, object]]] = []
            if "tesseract" in self.config.engines:
                for psm in self.config.tesseract_psm:
                    for oem in self.config.tesseract_oem:
                        engine_options_list.append(("tesseract", {"psm": psm, "oem": oem, "allowlist": self.config.allowlist}))
            if "easyocr" in self.config.engines:
                engine_options_list.append(("easyocr", {"allowlist": self.config.allowlist}))

            for (grid, grid_meta) in grid_candidates:
                rows, cols = grid.rows, grid.cols
                for engine_name, engine_opts in engine_options_list:
                    start_engine = time.perf_counter()
                    engine_table: List[List[str]] = [["" for _ in range(cols)] for _ in range(rows)]

                    for r in range(rows):
                        for c in range(cols):
                            bbox = grid.cells[r * cols + c]
                            cell_img = self._extract_cell_image(variant.image, bbox)
                            result: OcrResult = self.ocr.recognize_cell(engine_name, cell_img, engine_opts)
                            engine_table[r][c] = result.text

                    duration_ms = int((time.perf_counter() - start_engine) * 1000)
                    sheet_meta: Dict[str, object] = {
                        "image_name": self.config.image_path,
                        "engine": engine_name,
                        "engine_options": engine_opts,
                        "variant_name": variant.name,
                        "preprocess_meta": variant.meta,
                        "grid_meta": grid_meta,
                        "rows": rows,
                        "cols": cols,
                        "duration_ms": duration_ms,
                    }
                    outputs.append(OutputTable(table=engine_table, sheet_meta=sheet_meta))

            _ = int((time.perf_counter() - start_variant) * 1000)

        return outputs