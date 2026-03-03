import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


FITS_SUFFIXES = {".fits", ".fit", ".fts"}
MIN_ZOOM = 0.05
MAX_ZOOM = 30.0
ZOOM_STEP = 1.15


def _read_fits_2d(file_path: Path) -> np.ndarray:
    with fits.open(file_path, memmap=False) as hdul:
        data = hdul[0].data

    if data is None:
        raise ValueError(f"{file_path.name} 没有可显示的图像数据")

    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 2:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"{file_path.name} 不是二维图像")
    return data


def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("图像全部是无效数值")

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
    return norm


def _gray_to_qimage(gray: np.ndarray) -> QImage:
    h, w = gray.shape
    qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
    return qimg.copy()


def _rgb_to_qimage(rgb: np.ndarray) -> QImage:
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return qimg.copy()


def fits_to_qimage(file_path: Path) -> tuple[QImage, np.ndarray]:
    data = _read_fits_2d(file_path)
    norm = _normalize_to_uint8(data)
    return _gray_to_qimage(norm), data


def rpca_decompose(matrix: np.ndarray, max_iter: int = 80, tol: float = 1e-6):
    lam = 1.0 / np.sqrt(max(matrix.shape))
    norm2 = np.linalg.norm(matrix, ord=2)
    norm_inf = np.linalg.norm(matrix.ravel(), ord=np.inf) / lam
    dual_norm = max(norm2, norm_inf)
    if dual_norm == 0:
        return np.zeros_like(matrix), np.zeros_like(matrix)

    y = matrix / dual_norm
    mu = 1.25 / (norm2 + 1e-12)
    mu_bar = mu * 1e7
    rho = 1.5

    l = np.zeros_like(matrix)
    s = np.zeros_like(matrix)
    matrix_norm = np.linalg.norm(matrix, ord="fro") + 1e-12

    for _ in range(max_iter):
        u, sigma, vt = np.linalg.svd(matrix - s + (1.0 / mu) * y, full_matrices=False)
        sigma_thresh = np.maximum(sigma - 1.0 / mu, 0.0)
        rank = int(np.sum(sigma_thresh > 0))
        if rank > 0:
            l = (u[:, :rank] * sigma_thresh[:rank]) @ vt[:rank, :]
        else:
            l.fill(0.0)

        residual = matrix - l + (1.0 / mu) * y
        s = np.sign(residual) * np.maximum(np.abs(residual) - lam / mu, 0.0)

        z = matrix - l - s
        y = y + mu * z
        mu = min(mu * rho, mu_bar)
        if np.linalg.norm(z, ord="fro") / matrix_norm < tol:
            break
    return l, s


def build_annotation_image(source: np.ndarray, sparse: np.ndarray, threshold: float) -> QImage:
    base = _normalize_to_uint8(source)
    rgb = np.stack([base, base, base], axis=2).astype(np.float32)

    change_mask = np.abs(sparse) > threshold
    bg_mask = ~change_mask

    # 背景区域加轻微绿色，变化区域标红，方便快速检查时序变化。
    rgb[bg_mask, 1] = np.clip(rgb[bg_mask, 1] * 0.75 + 45.0, 0, 255)
    rgb[change_mask, 0] = 255
    rgb[change_mask, 1] *= 0.25
    rgb[change_mask, 2] *= 0.15

    return _rgb_to_qimage(rgb.astype(np.uint8))


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
        old_rect = self.sceneRect()
        old_center = self.mapToScene(self.viewport().rect().center())
        keep_relative_center = old_rect.isValid() and old_rect.width() > 0 and old_rect.height() > 0
        rel_x = 0.5
        rel_y = 0.5
        if keep_relative_center:
            rel_x = (old_center.x() - old_rect.left()) / old_rect.width()
            rel_y = (old_center.y() - old_rect.top()) / old_rect.height()

        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        new_rect = self._pixmap_item.boundingRect()
        self.scene().setSceneRect(new_rect)
        self.set_zoom(self._zoom)
        if keep_relative_center and new_rect.width() > 0 and new_rect.height() > 0:
            rel_x = min(max(rel_x, 0.0), 1.0)
            rel_y = min(max(rel_y, 0.0), 1.0)
            target = QPointF(
                new_rect.left() + rel_x * new_rect.width(),
                new_rect.top() + rel_y * new_rect.height(),
            )
            self.centerOn(target)
        else:
            self.centerOn(new_rect.center())

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
        self._rpca_annotated = []
        self._is_rpca_view = False
        self._rpca_sparse = None
        self._rpca_abs = None
        self._rpca_threshold = None

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_current_row_changed)

        self.view = ImageView()
        self.view.zoom_changed.connect(self._on_zoom_changed)

        self.info_label = QLabel("拖放一组 FITS 到窗口，或点击“打开文件”")
        self.info_label.setStyleSheet("padding: 6px;")

        open_btn = QPushButton("打开文件")
        open_btn.clicked.connect(self._open_files_dialog)
        rpca_btn = QPushButton("RPCA标注")
        rpca_btn.clicked.connect(self._run_rpca_annotation)
        self.auto_threshold_checkbox = QCheckBox("自动阈值")
        self.auto_threshold_checkbox.setChecked(True)
        self.auto_threshold_checkbox.toggled.connect(self._on_threshold_mode_changed)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 100)
        self.threshold_slider.setValue(45)
        self.threshold_slider.valueChanged.connect(self._on_threshold_control_changed)
        self.threshold_hint_label = QLabel("阈值灵敏度: 45 (自动)")

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.view, 1)
        right_layout.addWidget(self.info_label, 0)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(open_btn, 0)
        left_layout.addWidget(rpca_btn, 0)
        left_layout.addWidget(self.auto_threshold_checkbox, 0)
        left_layout.addWidget(self.threshold_hint_label, 0)
        left_layout.addWidget(self.threshold_slider, 0)
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
        view_mode = "RPCA标注" if self._is_rpca_view else "原图"
        threshold_text = ""
        if self._rpca_threshold is not None:
            threshold_text = f"    阈值: {self._rpca_threshold:.4g}"
        self.info_label.setText(
            f"图像: {current}/{count}    缩放: {self._zoom * 100:.1f}%    "
            f"模式: {view_mode}{threshold_text}    快捷键: Tab/Shift+Tab 切换, Ctrl+加减号缩放"
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
        self._rpca_annotated.clear()
        self._is_rpca_view = False
        self._rpca_sparse = None
        self._rpca_abs = None
        self._rpca_threshold = None

        failed = []
        for path in fits_files:
            try:
                qimg, data = fits_to_qimage(path)
            except Exception as exc:
                failed.append(f"{path.name}: {exc}")
                continue

            self._images.append((path, qimg, data))
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

        _, qimg, _ = self._images[row]
        if self._is_rpca_view and row < len(self._rpca_annotated):
            self.view.set_image(self._rpca_annotated[row])
        else:
            self.view.set_image(qimg)
            self._is_rpca_view = False

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

    def _run_rpca_annotation(self):
        if len(self._images) < 2:
            QMessageBox.information(self, "提示", "至少需要 2 张 FITS 图像才能做 RPCA 标注")
            return

        h, w = self._images[0][2].shape
        for path, _, data in self._images:
            if data.shape != (h, w):
                QMessageBox.warning(
                    self,
                    "尺寸不一致",
                    f"{path.name} 尺寸与首张不同，无法执行 RPCA（需要所有图像同尺寸）",
                )
                return

        stack = []
        for _, _, data in self._images:
            frame = np.array(data, dtype=np.float32, copy=True)
            finite = np.isfinite(frame)
            if finite.any():
                med = float(np.median(frame[finite]))
                frame[~finite] = med
            else:
                frame.fill(0.0)
            stack.append(frame)

        matrix = np.stack([x.reshape(-1) for x in stack], axis=1)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            _, sparse = rpca_decompose(matrix)
        except Exception as exc:
            QMessageBox.critical(self, "RPCA失败", f"RPCA 计算失败: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._rpca_sparse = sparse
        self._rpca_abs = np.abs(sparse)
        self._rebuild_rpca_annotations()

        self._is_rpca_view = True
        current = self.list_widget.currentRow()
        if current >= 0:
            self._on_current_row_changed(current)
        QMessageBox.information(self, "完成", "RPCA 标注已完成：红色为变化区域，绿色增强为背景区域")

    def _compute_current_threshold(self) -> float:
        if self._rpca_abs is None:
            return 0.0
        sensitivity = self.threshold_slider.value() / 100.0
        if self.auto_threshold_checkbox.isChecked():
            med = float(np.median(self._rpca_abs))
            mad = float(np.median(np.abs(self._rpca_abs - med)))
            # 灵敏度越高，阈值越低，标出的变化区域越多。
            k = 8.0 - 7.5 * sensitivity
            threshold = med + k * (mad + 1e-8)
        else:
            max_val = float(np.max(self._rpca_abs))
            threshold = max_val * (1.0 - sensitivity)
        if threshold <= 0:
            threshold = float(np.percentile(self._rpca_abs, 95))
        return threshold

    def _rebuild_rpca_annotations(self):
        if self._rpca_sparse is None:
            return
        self._rpca_threshold = self._compute_current_threshold()
        self._rpca_annotated = []
        h, w = self._images[0][2].shape
        self._rpca_annotated = []
        for idx, (_, _, data) in enumerate(self._images):
            sparse_i = self._rpca_sparse[:, idx].reshape(h, w)
            annotated = build_annotation_image(data, sparse_i, self._rpca_threshold)
            self._rpca_annotated.append(annotated)

    def _on_threshold_mode_changed(self):
        self._update_threshold_hint_label()
        self._on_threshold_control_changed()

    def _on_threshold_control_changed(self):
        self._update_threshold_hint_label()
        if self._rpca_sparse is None:
            self._refresh_info_label()
            return
        self._rebuild_rpca_annotations()
        if self._is_rpca_view:
            current = self.list_widget.currentRow()
            if current >= 0:
                self._on_current_row_changed(current)
        else:
            self._refresh_info_label()

    def _update_threshold_hint_label(self):
        value = self.threshold_slider.value()
        mode = "自动" if self.auto_threshold_checkbox.isChecked() else "手动"
        self.threshold_hint_label.setText(f"阈值灵敏度: {value} ({mode})")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
