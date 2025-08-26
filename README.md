# Table OCR to Excel (Brute Force)

Desktop/CLI tool that converts an image of a printed table filled with handwritten numbers into an Excel workbook. It generates many variants using multiple preprocessing options and OCR engines (Tesseract, EasyOCR) to minimize missed digits. Each Excel sheet documents engine/options and runtime.

## Quick start (CLI)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -m table_ocr_app.cli --help
python -m table_ocr_app.cli /path/to/image.jpg --out results.xlsx --max-variants 16 --engines tesseract,easyocr
```

Alternatively once installed as a package:

```bash
pip install .
table-ocr-cli /path/to/image.jpg --out results.xlsx
```

## Features

- Brute-force preprocessing variants (deskew, CLAHE, denoise, scaling, adaptive/Otsu)
- Multiple OCR engines and options
  - Tesseract OEM/PSM sweep with whitelist for digits
  - EasyOCR with allowlist
- Grid detection via morphological line extraction
- One Excel workbook with many candidate sheets + a Manifest sheet recording metadata

## Desktop UI (PySide6)

A Qt-based UI to select image and set all options is planned. For now use the CLI.

## Optional: PaddleOCR

You can extend `ocr_engines.py` to add PaddleOCR. It is not included by default to keep install light.

## Notes

- Install Tesseract runtime if not present so `pytesseract` can invoke it.
- For best results, scan images at 300-600 DPI, avoid shadows, and crop to the table borders if possible.