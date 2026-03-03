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


def build_mask_subtracted_image(
    source: np.ndarray, current_sparse: np.ndarray, previous_sparse: np.ndarray, threshold: float
) -> QImage:
    base = _normalize_to_uint8(source).astype(np.float32) * 0.35
    rgb = np.stack([base, base, base], axis=2)

    # mask减图: 当前帧mask - 前一帧mask，红色=新增变化区，青色=消失变化区。
    curr_mask = np.abs(current_sparse) > threshold
    prev_mask = np.abs(previous_sparse) > threshold
    added = curr_mask & (~prev_mask)
    removed = prev_mask & (~curr_mask)
    overlap = curr_mask & prev_mask

    rgb[overlap, :] = np.clip(rgb[overlap, :] + 30.0, 0, 255)
    rgb[added, 0] = 255
    rgb[added, 1] *= 0.2
    rgb[added, 2] *= 0.2
    rgb[removed, 0] *= 0.25
    rgb[removed, 1] = 220
    rgb[removed, 2] = 255

    return _rgb_to_qimage(rgb.astype(np.uint8))


def _sanitize_frame(data: np.ndarray) -> np.ndarray:
    frame = np.array(data, dtype=np.float32, copy=True)
    finite = np.isfinite(frame)
    if finite.any():
        med = float(np.median(frame[finite]))
        frame[~finite] = med
    else:
        frame.fill(0.0)
    return frame


def detect_point_sources(data: np.ndarray, sensitivity: float) -> np.ndarray:
    frame = _sanitize_frame(data)
    med = float(np.median(frame))
    mad = float(np.median(np.abs(frame - med)))
    sigma = 1.4826 * mad + 1e-6
    z = (frame - med) / sigma

    # 灵敏度越高，阈值越低，检测点越多。
    threshold = 6.0 - 4.5 * np.clip(sensitivity, 0.01, 1.0)
    mask = z > threshold
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    points_with_score = []

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            pixels = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny in range(max(1, cy - 1), min(h - 1, cy + 2)):
                    for nx in range(max(1, cx - 1), min(w - 1, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            # 每个连通域只保留一个代表点，使用几何质心法。
            comp_scores = np.array([z[py, px] for py, px in pixels], dtype=np.float32)
            best_idx = int(np.argmax(comp_scores))
            ys = np.array([py for py, _ in pixels], dtype=np.float32)
            xs = np.array([px for _, px in pixels], dtype=np.float32)
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))

            points_with_score.append((cx, cy, float(comp_scores[best_idx])))

    if not points_with_score:
        return np.zeros((0, 2), dtype=np.float32)

    points_with_score.sort(key=lambda t: t[2], reverse=True)
    points = np.array([[p[0], p[1]] for p in points_with_score[:250]], dtype=np.float32)
    return points


def _score_translation(ref_points: np.ndarray, tgt_points: np.ndarray, shift: np.ndarray, radius: float) -> int:
    if len(ref_points) == 0 or len(tgt_points) == 0:
        return 0
    aligned = tgt_points + shift[None, :]
    diff = aligned[:, None, :] - ref_points[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    min_d2 = np.min(d2, axis=1)
    return int(np.sum(min_d2 <= radius * radius))


def estimate_translation(ref_points: np.ndarray, tgt_points: np.ndarray, radius: float = 3.0) -> np.ndarray:
    if len(ref_points) == 0 or len(tgt_points) == 0:
        return np.zeros(2, dtype=np.float32)

    ref_small = ref_points[:80]
    tgt_small = tgt_points[:80]
    best_shift = np.zeros(2, dtype=np.float32)
    best_score = -1
    for rp in ref_small:
        for tp in tgt_small:
            shift = rp - tp
            score = _score_translation(ref_points, tgt_points, shift, radius)
            if score > best_score:
                best_score = score
                best_shift = shift
    return best_shift.astype(np.float32)


def match_points(
    ref_points: np.ndarray, tgt_points: np.ndarray, shift: np.ndarray, radius: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    if len(ref_points) == 0 or len(tgt_points) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    aligned = tgt_points + shift[None, :]
    diff = aligned[:, None, :] - ref_points[None, :, :]
    d2 = np.sum(diff * diff, axis=2)

    t_idx, r_idx = np.where(d2 <= radius * radius)
    if len(t_idx) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    order = np.argsort(d2[t_idx, r_idx])
    used_t = set()
    used_r = set()
    matched_t = []
    matched_r = []
    for o in order:
        ti = int(t_idx[o])
        ri = int(r_idx[o])
        if ti in used_t or ri in used_r:
            continue
        used_t.add(ti)
        used_r.add(ri)
        matched_t.append(ti)
        matched_r.append(ri)
    return np.array(matched_r, dtype=np.int32), np.array(matched_t, dtype=np.int32)


def _draw_cross(rgb: np.ndarray, x: float, y: float, color: tuple[int, int, int], size: int = 3):
    h, w, _ = rgb.shape
    cx = int(round(x))
    cy = int(round(y))
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return
    for d in range(-size, size + 1):
        xx = cx + d
        yy = cy + d
        if 0 <= xx < w:
            rgb[cy, xx, :] = color
        if 0 <= yy < h:
            rgb[yy, cx, :] = color


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _convex_hull(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return points
    pts = np.unique(np.round(points).astype(np.int32), axis=0)
    if len(pts) < 3:
        return pts.astype(np.float32)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    lower = []
    for p in pts:
        p = p.astype(np.float32)
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        p = p.astype(np.float32)
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.vstack([lower[:-1], upper[:-1]])
    return hull.astype(np.float32)


def _draw_line(rgb: np.ndarray, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int]):
    h, w, _ = rgb.shape
    x0 = int(round(x0))
    y0 = int(round(y0))
    x1 = int(round(x1))
    y1 = int(round(y1))

    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            rgb[y0, x0, :] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _draw_polygon(rgb: np.ndarray, points: np.ndarray, color: tuple[int, int, int]):
    if len(points) < 3:
        return
    hull = _convex_hull(points)
    if len(hull) < 3:
        return
    for i in range(len(hull)):
        p0 = hull[i]
        p1 = hull[(i + 1) % len(hull)]
        _draw_line(rgb, p0[0], p0[1], p1[0], p1[1], color)


def build_point_change_image(
    source: np.ndarray,
    matched_points: np.ndarray,
    added_points: np.ndarray,
    missing_points: np.ndarray,
) -> QImage:
    base = _normalize_to_uint8(source)
    rgb = np.stack([base, base, base], axis=2)

    # 绿=匹配点, 红=新增点, 蓝=消失点(在当前帧预测位置)。
    for p in matched_points:
        _draw_cross(rgb, p[0], p[1], (40, 220, 40), size=2)
    for p in added_points:
        _draw_cross(rgb, p[0], p[1], (255, 60, 60), size=3)
    for p in missing_points:
        _draw_cross(rgb, p[0], p[1], (80, 150, 255), size=3)
    _draw_polygon(rgb, matched_points, (40, 220, 40))
    _draw_polygon(rgb, added_points, (255, 60, 60))
    _draw_polygon(rgb, missing_points, (80, 150, 255))

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
        self._rpca_mask_diff = []
        self._is_rpca_view = False
        self._rpca_show_mask_diff = False
        self._annotation_mode_label = ""
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
        fixed_bg_rpca_btn = QPushButton("固定背景RPCA")
        fixed_bg_rpca_btn.clicked.connect(self._run_fixed_background_rpca)
        self.rpca_view_toggle_btn = QPushButton("显示: 原始红绿mask")
        self.rpca_view_toggle_btn.clicked.connect(self._toggle_rpca_view_mode)
        point_change_btn = QPushButton("点源对齐检测")
        point_change_btn.clicked.connect(self._run_point_source_change_detection)
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
        left_layout.addWidget(fixed_bg_rpca_btn, 0)
        left_layout.addWidget(self.rpca_view_toggle_btn, 0)
        left_layout.addWidget(point_change_btn, 0)
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
        view_mode = self._annotation_mode_label if self._is_rpca_view else "原图"
        threshold_text = ""
        if self._rpca_threshold is not None:
            threshold_text = f"    阈值: {self._rpca_threshold:.4g}"
        self.info_label.setText(
            f"图像: {current}/{count}    缩放: {self._zoom * 100:.1f}%    "
            f"模式: {view_mode}{threshold_text}    快捷键: Tab/Shift+Tab 切换, Ctrl+加减号缩放"
        )

    def _toggle_rpca_view_mode(self):
        if self._rpca_sparse is None:
            QMessageBox.information(self, "提示", "请先执行 RPCA 标注后再切换显示模式")
            return
        self._rpca_show_mask_diff = not self._rpca_show_mask_diff
        self._update_rpca_view_toggle_text()
        if self._is_rpca_view:
            current = self.list_widget.currentRow()
            if current >= 0:
                self._on_current_row_changed(current)

    def _update_rpca_view_toggle_text(self):
        text = "显示: mask减图(当前-前一帧)" if self._rpca_show_mask_diff else "显示: 原始红绿mask"
        self.rpca_view_toggle_btn.setText(text)

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
        self._rpca_mask_diff.clear()
        self._is_rpca_view = False
        self._rpca_show_mask_diff = False
        self._annotation_mode_label = ""
        self._rpca_sparse = None
        self._rpca_abs = None
        self._rpca_threshold = None
        self._update_rpca_view_toggle_text()

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
            if self._rpca_show_mask_diff and row < len(self._rpca_mask_diff):
                self.view.set_image(self._rpca_mask_diff[row])
            else:
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
        self._annotation_mode_label = "RPCA标注"
        self._rpca_show_mask_diff = False
        self._update_rpca_view_toggle_text()
        current = self.list_widget.currentRow()
        if current >= 0:
            self._on_current_row_changed(current)
        QMessageBox.information(self, "完成", "RPCA 标注已完成：红色为变化区域，绿色增强为背景区域")

    def _run_fixed_background_rpca(self):
        if len(self._images) < 2:
            QMessageBox.information(self, "提示", "至少需要 2 张 FITS 图像才能做固定背景 RPCA")
            return

        ref_idx = self.list_widget.currentRow()
        if ref_idx < 0 or ref_idx >= len(self._images):
            QMessageBox.information(self, "提示", "请先用 Tab 或鼠标选中一张图作为参考背景")
            return

        h, w = self._images[0][2].shape
        for path, _, data in self._images:
            if data.shape != (h, w):
                QMessageBox.warning(
                    self,
                    "尺寸不一致",
                    f"{path.name} 尺寸与首张不同，无法执行固定背景 RPCA（需要所有图像同尺寸）",
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
        reference_vec = matrix[:, [ref_idx]]
        centered_matrix = matrix - reference_vec

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            _, sparse = rpca_decompose(centered_matrix)
        except Exception as exc:
            QMessageBox.critical(self, "固定背景RPCA失败", f"计算失败: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        # 参考帧相对于自身的变化应为 0，抑制数值残差带来的误检。
        sparse[:, ref_idx] = 0.0
        self._rpca_sparse = sparse
        self._rpca_abs = np.abs(sparse)
        self._rebuild_rpca_annotations()

        ref_name = self._images[ref_idx][0].name
        self._is_rpca_view = True
        self._annotation_mode_label = f"固定背景RPCA(参考: {ref_name})"
        self._rpca_show_mask_diff = False
        self._update_rpca_view_toggle_text()
        current = self.list_widget.currentRow()
        if current >= 0:
            self._on_current_row_changed(current)
        QMessageBox.information(
            self,
            "完成",
            f"固定背景 RPCA 标注完成。\n参考背景: {ref_name}\n红色为相对参考的变化区域。",
        )

    def _run_point_source_change_detection(self):
        if len(self._images) < 2:
            QMessageBox.information(self, "提示", "至少需要 2 张 FITS 图像才能做点源对齐检测")
            return

        ref_idx = self.list_widget.currentRow()
        if ref_idx < 0 or ref_idx >= len(self._images):
            QMessageBox.information(self, "提示", "请先选中一张图作为参考图")
            return

        h, w = self._images[0][2].shape
        for path, _, data in self._images:
            if data.shape != (h, w):
                QMessageBox.warning(
                    self,
                    "尺寸不一致",
                    f"{path.name} 尺寸与首张不同，无法执行点源对齐检测（需要所有图像同尺寸）",
                )
                return

        sensitivity = self.threshold_slider.value() / 100.0
        match_radius = 3.0

        frames = [_sanitize_frame(data) for _, _, data in self._images]
        point_sets = [detect_point_sources(frame, sensitivity) for frame in frames]
        ref_points = point_sets[ref_idx]
        if len(ref_points) == 0:
            QMessageBox.warning(self, "检测失败", "参考图未检测到有效点源，请提高灵敏度后重试")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            annotated_images = []
            for idx, frame in enumerate(frames):
                tgt_points = point_sets[idx]
                if len(tgt_points) == 0:
                    missing_on_tgt = ref_points.copy()
                    qimg = build_point_change_image(
                        frame,
                        matched_points=np.zeros((0, 2), dtype=np.float32),
                        added_points=np.zeros((0, 2), dtype=np.float32),
                        missing_points=missing_on_tgt,
                    )
                    annotated_images.append(qimg)
                    continue

                shift = estimate_translation(ref_points, tgt_points, radius=match_radius)
                matched_ref, matched_tgt = match_points(
                    ref_points, tgt_points, shift=shift, radius=match_radius
                )

                ref_mask = np.ones(len(ref_points), dtype=bool)
                tgt_mask = np.ones(len(tgt_points), dtype=bool)
                ref_mask[matched_ref] = False
                tgt_mask[matched_tgt] = False

                matched_points = tgt_points[matched_tgt] if len(matched_tgt) else np.zeros((0, 2), dtype=np.float32)
                added_points = tgt_points[tgt_mask]
                missing_points = ref_points[ref_mask] - shift[None, :]

                qimg = build_point_change_image(
                    frame,
                    matched_points=matched_points,
                    added_points=added_points,
                    missing_points=missing_points,
                )
                annotated_images.append(qimg)
        finally:
            QApplication.restoreOverrideCursor()

        self._rpca_sparse = None
        self._rpca_abs = None
        self._rpca_threshold = None
        self._rpca_annotated = annotated_images
        self._rpca_mask_diff = []
        self._rpca_show_mask_diff = False
        self._update_rpca_view_toggle_text()
        self._is_rpca_view = True
        ref_name = self._images[ref_idx][0].name
        self._annotation_mode_label = f"点源对齐检测(参考: {ref_name})"
        current = self.list_widget.currentRow()
        if current >= 0:
            self._on_current_row_changed(current)
        QMessageBox.information(
            self,
            "完成",
            "点源对齐检测完成：绿色=匹配点，红色=新增点，蓝色=消失点。",
        )

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
        self._rpca_mask_diff = []
        h, w = self._images[0][2].shape
        for idx, (_, _, data) in enumerate(self._images):
            sparse_i = self._rpca_sparse[:, idx].reshape(h, w)
            annotated = build_annotation_image(data, sparse_i, self._rpca_threshold)
            self._rpca_annotated.append(annotated)
            if idx == 0:
                prev_sparse_i = np.zeros_like(sparse_i)
            else:
                prev_sparse_i = self._rpca_sparse[:, idx - 1].reshape(h, w)
            mask_diff = build_mask_subtracted_image(
                data, sparse_i, prev_sparse_i, self._rpca_threshold
            )
            self._rpca_mask_diff.append(mask_diff)

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
