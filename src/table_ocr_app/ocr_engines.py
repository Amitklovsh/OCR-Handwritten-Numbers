from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    pytesseract = None  # type: ignore

try:
    import easyocr  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    easyocr = None  # type: ignore


@dataclass
class OcrResult:
    text: str
    confidence: float


class OcrRunner:
    def __init__(self, use_tesseract: bool = True, use_easyocr: bool = True, tesseract_lang: str = "eng") -> None:
        self.use_tesseract = use_tesseract and (pytesseract is not None)
        self.use_easyocr = use_easyocr and (easyocr is not None)
        self._easy_reader = None
        if self.use_easyocr:
            # Initialize lazily to avoid heavy startup
            self._easy_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        self.tesseract_lang = tesseract_lang

    def recognize_cell(self, engine: str, cell_img: np.ndarray, options: Dict[str, object]) -> OcrResult:
        engine = engine.lower()
        if engine == "tesseract" and self.use_tesseract:
            return self._recognize_tesseract(cell_img, options)
        if engine == "easyocr" and self.use_easyocr:
            return self._recognize_easyocr(cell_img, options)
        return OcrResult(text="", confidence=0.0)

    def _recognize_tesseract(self, cell_img: np.ndarray, options: Dict[str, object]) -> OcrResult:
        if pytesseract is None:
            return OcrResult(text="", confidence=0.0)
        psm = int(options.get("psm", 7))
        oem = int(options.get("oem", 3))
        allowlist = str(options.get("allowlist", "0123456789"))
        config = f"--oem {oem} --psm {psm} -c tessedit_char_whitelist={allowlist}"
        try:
            # Use data to obtain confidence
            data = pytesseract.image_to_data(cell_img, lang=self.tesseract_lang, config=config, output_type=pytesseract.Output.DICT)
            texts = [t for t in data.get("text", []) if t and t.strip()]
            confs = [float(c) for t, c in zip(data.get("text", []), data.get("conf", [])) if t and t.strip() and c not in ("-1", -1)]
            if not texts:
                return OcrResult(text="", confidence=0.0)
            text = "".join(texts)
            conf = float(np.mean(confs)) if confs else 0.0
            # Keep only digits
            text = "".join([ch for ch in text if ch.isdigit()])
            return OcrResult(text=text, confidence=conf)
        except Exception:
            return OcrResult(text="", confidence=0.0)

    def _recognize_easyocr(self, cell_img: np.ndarray, options: Dict[str, object]) -> OcrResult:
        if self._easy_reader is None:
            return OcrResult(text="", confidence=0.0)
        allowlist = str(options.get("allowlist", "0123456789"))
        try:
            results = self._easy_reader.readtext(cell_img, detail=1, paragraph=False, allowlist=allowlist)
            if not results:
                return OcrResult(text="", confidence=0.0)
            texts = [r[1] for r in results]
            confs = [float(r[2]) for r in results if len(r) >= 3]
            text = "".join(texts)
            text = "".join([ch for ch in text if ch.isdigit()])
            conf = float(np.mean(confs)) if confs else 0.0
            return OcrResult(text=text, confidence=conf)
        except Exception:
            return OcrResult(text="", confidence=0.0)