import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


FITS_SUFFIXES = {".fits", ".fit", ".fts"}
MIN_ZOOM = 0.05
MAX_ZOOM = 30.0
ZOOM_STEP = 1.15


def fits_to_qimage(file_path: Path) -> QImage:
    with fits.open(file_path, memmap=False) as hdul:
        data = hdul[0].data

    if data is None:
        raise ValueError(f"{file_path.name} 没有可显示的图像数据")

    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 2:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"{file_path.name} 不是二维图像")

    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError(f"{file_path.name} 全部是无效数值")

    valid = data[finite]
    p1, p99 = np.percentile(valid, [1, 99])
    if p99 <= p1:
        p1 = valid.min()
        p99 = valid.max()
    if p99 <= p1:
        p99 = p1 + 1.0

    clipped = np.clip(data, p1, p99)
    norm = ((clipped - p1) / (p99 - p1) * 255.0).astype(np.uint8)
    norm[~finite] = 0

    h, w = norm.shape
    bytes_per_line = w
    qimg = QImage(norm.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    return qimg.copy()


class ImageView(QGraphicsView):
    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)
        self._zoom = 1.0
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

    def set_image(self, image: QImage) -> None:
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(self._pixmap_item.boundingRect())
        self.set_zoom(self._zoom)

    def set_zoom(self, value: float) -> None:
        self._zoom = max(MIN_ZOOM, min(MAX_ZOOM, value))
        self.resetTransform()
        self.scale(self._zoom, self._zoom)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            new_zoom = self._zoom * ZOOM_STEP
        else:
            new_zoom = self._zoom / ZOOM_STEP
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, new_zoom))
        self.zoom_changed.emit(new_zoom)
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS 拖放浏览器（Tab 切换 + 同步缩放）")
        self.resize(1200, 780)
        self.setAcceptDrops(True)

        self._zoom = 1.0
        self._images = []

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_current_row_changed)

        self.view = ImageView()
        self.view.zoom_changed.connect(self._on_zoom_changed)

        self.info_label = QLabel("拖放一组 FITS 到窗口，或点击“打开文件”")
        self.info_label.setStyleSheet("padding: 6px;")

        open_btn = QPushButton("打开文件")
        open_btn.clicked.connect(self._open_files_dialog)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.view, 1)
        right_layout.addWidget(self.info_label, 0)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(open_btn, 0)
        left_layout.addWidget(self.list_widget, 1)

        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(splitter)
        self.setCentralWidget(root)

        next_shortcut = QShortcut(QKeySequence(Qt.Key_Tab), self)
        next_shortcut.activated.connect(self._next_image)
        prev_shortcut = QShortcut(QKeySequence("Shift+Tab"), self)
        prev_shortcut.activated.connect(self._prev_image)

        zoom_in_action = QAction(self)
        zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        zoom_in_action.triggered.connect(
            lambda: self._on_zoom_changed(self._zoom * ZOOM_STEP)
        )
        self.addAction(zoom_in_action)

        zoom_out_action = QAction(self)
        zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        zoom_out_action.triggered.connect(
            lambda: self._on_zoom_changed(self._zoom / ZOOM_STEP)
        )
        self.addAction(zoom_out_action)

        reset_zoom_action = QAction(self)
        reset_zoom_action.setShortcut(QKeySequence("Ctrl+0"))
        reset_zoom_action.triggered.connect(lambda: self._on_zoom_changed(1.0))
        self.addAction(reset_zoom_action)

    def _open_files_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择 FITS 文件",
            "",
            "FITS Files (*.fits *.fit *.fts);;All Files (*)",
        )
        if files:
            self.load_files([Path(f) for f in files])

    def _on_zoom_changed(self, new_zoom: float) -> None:
        self._zoom = max(MIN_ZOOM, min(MAX_ZOOM, new_zoom))
        self.view.set_zoom(self._zoom)
        self._refresh_info_label()

    def _refresh_info_label(self):
        count = len(self._images)
        current = self.list_widget.currentRow() + 1 if count else 0
        self.info_label.setText(
            f"图像: {current}/{count}    缩放: {self._zoom * 100:.1f}%    "
            "快捷键: Tab/Shift+Tab 切换, Ctrl+加减号缩放"
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                paths.append(Path(url.toLocalFile()))
        self.load_files(paths)
        event.acceptProposedAction()

    def load_files(self, paths):
        fits_files = []
        for p in paths:
            if p.is_file() and p.suffix.lower() in FITS_SUFFIXES:
                fits_files.append(p)
            elif p.is_dir():
                fits_files.extend(
                    sorted(
                        [
                            child
                            for child in p.iterdir()
                            if child.is_file() and child.suffix.lower() in FITS_SUFFIXES
                        ]
                    )
                )

        if not fits_files:
            QMessageBox.warning(self, "提示", "没有找到可用的 FITS 文件")
            return

        fits_files = sorted(dict.fromkeys(fits_files))

        self.list_widget.clear()
        self._images.clear()

        failed = []
        for path in fits_files:
            try:
                qimg = fits_to_qimage(path)
            except Exception as exc:
                failed.append(f"{path.name}: {exc}")
                continue

            self._images.append((path, qimg))
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            self.list_widget.addItem(item)

        if failed:
            QMessageBox.information(
                self,
                "部分文件加载失败",
                "以下文件未加载:\n\n" + "\n".join(failed[:12]),
            )

        if self._images:
            self.list_widget.setCurrentRow(0)
        else:
            QMessageBox.warning(self, "提示", "FITS 文件读取失败")
        self._refresh_info_label()

    def _on_current_row_changed(self, row: int):
        if row < 0 or row >= len(self._images):
            return
        _, qimg = self._images[row]
        self.view.set_image(qimg)
        self.view.set_zoom(self._zoom)
        self._refresh_info_label()

    def _next_image(self):
        count = self.list_widget.count()
        if count == 0:
            return
        current = self.list_widget.currentRow()
        self.list_widget.setCurrentRow((current + 1) % count)

    def _prev_image(self):
        count = self.list_widget.count()
        if count == 0:
            return
        current = self.list_widget.currentRow()
        self.list_widget.setCurrentRow((current - 1 + count) % count)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
