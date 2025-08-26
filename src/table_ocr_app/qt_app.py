from __future__ import annotations

import os
import sys
import time
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from PySide6 import QtCore, QtWidgets

from .orchestrator import RecognitionOrchestrator, RunConfig
from .excel_export import export_tables_to_workbook, SheetInfo


class WorkerSignals(QtCore.QObject):
    progress = QtCore.Signal(str)
    done = QtCore.Signal()
    error = QtCore.Signal(str)
    last_output = QtCore.Signal(str)


class Worker(QtCore.QRunnable):
    def __init__(self, images: List[str], out_dir: str, base_name: str, cfg_template: RunConfig) -> None:
        super().__init__()
        self.images = images
        self.out_dir = out_dir
        self.base_name = base_name
        self.cfg_template = cfg_template
        self.signals = WorkerSignals()
        # Back-compat aliases for older UI wiring
        self.sig_progress = self.signals.progress
        self.sig_done = self.signals.done
        self.sig_error = self.signals.error

    @QtCore.Slot()
    def run(self) -> None:  # type: ignore[override]
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            for idx, img in enumerate(self.images, start=1):
                t0 = time.perf_counter()
                cfg = RunConfig(
                    image_path=img,
                    max_variants=self.cfg_template.max_variants,
                    engines=self.cfg_template.engines,
                    tesseract_psm=self.cfg_template.tesseract_psm,
                    tesseract_oem=self.cfg_template.tesseract_oem,
                    allowlist=self.cfg_template.allowlist,
                )
                self.signals.progress.emit(f"Processing {os.path.basename(img)}...")
                orch = RecognitionOrchestrator(cfg)
                outputs = orch.run()
                tables = [ot.table for ot in outputs]
                sheet_infos = [SheetInfo(sheet_name=f"{i+1}-{ot.sheet_meta['engine']}", meta=ot.sheet_meta) for i, ot in enumerate(outputs)]
                out_path = os.path.join(self.out_dir, f"{self.base_name}-{os.path.splitext(os.path.basename(img))[0]}.xlsx")
                export_tables_to_workbook(out_path, tables, sheet_infos)
                dt = int((time.perf_counter() - t0) * 1000)
                self.signals.progress.emit(f"Saved {out_path} in {dt} ms ({len(tables)} tables)")
                self.signals.last_output.emit(out_path)
            self.signals.done.emit()
        except Exception as e:
            self.signals.error.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Table OCR to Excel - Brute Force")
        self._build_ui()
        self.pool = QtCore.QThreadPool.globalInstance()
        self.last_excel_path: Optional[str] = None

    def _build_ui(self) -> None:
        w = QtWidgets.QWidget(self)
        lay = QtWidgets.QVBoxLayout(w)

        # Image selection
        self.image_list = QtWidgets.QListWidget()
        self.btn_add_images = QtWidgets.QPushButton("Add Images...")
        self.btn_clear_images = QtWidgets.QPushButton("Clear")
        img_row = QtWidgets.QHBoxLayout()
        img_row.addWidget(self.btn_add_images)
        img_row.addWidget(self.btn_clear_images)

        # Output dir and base name
        self.out_dir_edit = QtWidgets.QLineEdit(os.getcwd())
        self.btn_browse_out = QtWidgets.QPushButton("Browse...")
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(QtWidgets.QLabel("Output directory"))
        out_row.addWidget(self.out_dir_edit)
        out_row.addWidget(self.btn_browse_out)
        self.base_name_edit = QtWidgets.QLineEdit("results")
        base_row = QtWidgets.QHBoxLayout()
        base_row.addWidget(QtWidgets.QLabel("Output base name"))
        base_row.addWidget(self.base_name_edit)

        # Engines
        self.chk_tesseract = QtWidgets.QCheckBox("Tesseract")
        self.chk_tesseract.setChecked(True)
        self.chk_easyocr = QtWidgets.QCheckBox("EasyOCR")
        engine_row = QtWidgets.QHBoxLayout()
        engine_row.addWidget(QtWidgets.QLabel("Engines"))
        engine_row.addWidget(self.chk_tesseract)
        engine_row.addWidget(self.chk_easyocr)
        engine_row.addStretch()

        # Tesseract options
        self.psm_edit = QtWidgets.QLineEdit("6,7,8,10,13")
        self.oem_edit = QtWidgets.QLineEdit("1,3")
        tess_layout = QtWidgets.QHBoxLayout()
        tess_layout.addWidget(QtWidgets.QLabel("Tesseract PSM"))
        tess_layout.addWidget(self.psm_edit)
        tess_layout.addWidget(QtWidgets.QLabel("OEM"))
        tess_layout.addWidget(self.oem_edit)

        # Variants and allowlist
        self.max_var_spin = QtWidgets.QSpinBox()
        self.max_var_spin.setRange(1, 999)
        self.max_var_spin.setValue(16)
        self.allow_edit = QtWidgets.QLineEdit("0123456789")
        var_row = QtWidgets.QHBoxLayout()
        var_row.addWidget(QtWidgets.QLabel("Max variants"))
        var_row.addWidget(self.max_var_spin)
        var_row.addWidget(QtWidgets.QLabel("Allowlist"))
        var_row.addWidget(self.allow_edit)

        # Start/Open buttons and log
        buttons_row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_open_last = QtWidgets.QPushButton("Open Last Excel")
        self.btn_open_last.setEnabled(False)
        buttons_row.addWidget(self.btn_start)
        buttons_row.addWidget(self.btn_open_last)
        buttons_row.addStretch()

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)

        lay.addWidget(QtWidgets.QLabel("Selected images"))
        lay.addWidget(self.image_list)
        lay.addLayout(img_row)
        lay.addSpacing(8)
        lay.addLayout(out_row)
        lay.addLayout(base_row)
        lay.addSpacing(8)
        lay.addLayout(engine_row)
        lay.addLayout(tess_layout)
        lay.addLayout(var_row)
        lay.addSpacing(8)
        lay.addLayout(buttons_row)
        lay.addWidget(self.log)

        self.setCentralWidget(w)

        # Connections
        self.btn_add_images.clicked.connect(self.on_add_images)
        self.btn_clear_images.clicked.connect(self.image_list.clear)
        self.btn_browse_out.clicked.connect(self.on_browse_out)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_open_last.clicked.connect(self.on_open_last)

    def on_add_images(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select images", os.getcwd(), "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        for f in files:
            self.image_list.addItem(f)

    def on_browse_out(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory", self.out_dir_edit.text())
        if d:
            self.out_dir_edit.setText(d)

    def on_start(self) -> None:
        images = [self.image_list.item(i).text() for i in range(self.image_list.count())]
        if not images:
            self._log("Add at least one image.")
            return
        out_dir = self.out_dir_edit.text().strip() or os.getcwd()
        base_name = self.base_name_edit.text().strip() or "results"
        engines: List[str] = []
        if self.chk_tesseract.isChecked():
            engines.append("tesseract")
        if self.chk_easyocr.isChecked():
            engines.append("easyocr")
        if not engines:
            self._log("Select at least one engine.")
            return
        try:
            psm = [int(x) for x in self.psm_edit.text().split(',') if x.strip()]
            oem = [int(x) for x in self.oem_edit.text().split(',') if x.strip()]
        except Exception:
            self._log("Invalid Tesseract PSM/OEM values")
            return
        allowlist = self.allow_edit.text()
        max_variants = int(self.max_var_spin.value())

        cfg_template = RunConfig(
            image_path="",
            max_variants=max_variants,
            engines=engines,
            tesseract_psm=psm,
            tesseract_oem=oem,
            allowlist=allowlist,
        )

        self.btn_start.setEnabled(False)
        self._log("Starting recognition. No further prompts will be shown.")
        worker = Worker(images, out_dir, base_name, cfg_template)
        # New-style connections
        worker.signals.progress.connect(self._log)
        worker.signals.done.connect(self._on_done)
        worker.signals.error.connect(self._on_error)
        worker.signals.last_output.connect(self._on_last_output)
        # Back-compat: if old code path uses sig_* aliases, they point to same signals
        self.pool.start(worker)

    def _log(self, msg: str) -> None:
        self.log.appendPlainText(msg)

    def _on_done(self) -> None:
        self._log("All tasks completed.")
        self.btn_start.setEnabled(True)

    def _on_error(self, msg: str) -> None:
        self._log(f"Error: {msg}")
        self.btn_start.setEnabled(True)

    def _on_last_output(self, path: str) -> None:
        self.last_excel_path = path
        self.btn_open_last.setEnabled(True)
        self._log(f"Last Excel: {path}")

    def on_open_last(self) -> None:
        if not self.last_excel_path or not os.path.exists(self.last_excel_path):
            self._log("No Excel file to open yet.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(self.last_excel_path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self.last_excel_path])
            else:
                subprocess.Popen(["xdg-open", self.last_excel_path])
        except Exception as e:
            self._log(f"Failed to open: {e}")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())