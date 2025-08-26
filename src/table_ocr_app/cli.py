from __future__ import annotations

import argparse
import os
from typing import List

from .excel_export import SheetInfo, export_tables_to_workbook
from .orchestrator import RecognitionOrchestrator, RunConfig
from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Table OCR to Excel (brute force variants)")
    p.add_argument("image", help="Path to input image containing a table")
    p.add_argument("--out", default="output.xlsx", help="Output Excel path")
    p.add_argument("--max-variants", type=int, default=12, help="Limit number of preprocessing variants")
    p.add_argument("--engines", default="tesseract,easyocr", help="Comma-separated list of engines to use")
    p.add_argument("--tesseract-psm", default="6,7,8,10,13", help="Comma-separated PSM values")
    p.add_argument("--tesseract-oem", default="1,3", help="Comma-separated OEM values")
    p.add_argument("--allowlist", default="0123456789", help="Characters allowed during OCR")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    engines: List[str] = [e.strip() for e in args.engines.split(",") if e.strip()]
    tpsm = [int(x) for x in args.tesseract_psm.split(",") if x]
    toem = [int(x) for x in args.tesseract_oem.split(",") if x]

    cfg = RunConfig(
        image_path=args.image,
        max_variants=args.max_variants,
        engines=engines,
        tesseract_psm=tpsm,
        tesseract_oem=toem,
        allowlist=args.allowlist,
    )

    orchestrator = RecognitionOrchestrator(cfg)
    outputs = orchestrator.run()

    tables = [ot.table for ot in outputs]
    sheet_infos = [SheetInfo(sheet_name=f"{i+1}-{ot.sheet_meta['engine']}", meta=ot.sheet_meta) for i, ot in enumerate(outputs)]

    ensure_dir(os.path.dirname(args.out) or ".")
    export_tables_to_workbook(args.out, tables, sheet_infos)


if __name__ == "__main__":
    main()