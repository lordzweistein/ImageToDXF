"""
ImageToDXF - GUI-Anwendung (Qt / qtpy)

Liest Bilder ein, erkennt Konturen automatisch (mit einstellbaren
Parametern), erlaubt eine Pixel->mm Kalibrierung per Mausklick und
exportiert die erkannten Konturen als DXF-Datei fuer CAD-Programme.

Die GUI basiert auf qtpy (abstrahiert PyQt5/PySide2/PyQt6/PySide6).
Standardmaessig wird PySide6 verwendet (siehe QT_API weiter unten).

Start:
    python3 app.py
"""
from __future__ import annotations

import os

# PySide6 als Qt-Backend erzwingen, bevor qtpy geladen wird.
os.environ.setdefault("QT_API", "pyside6")

import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QImage, QPen, QPixmap, QPolygonF
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from contour_core import (
    ContourParams,
    DetectedContour,
    PreprocessParams,
    compute_scale_mm_per_pixel,
    contours_to_dxf,
    find_contours,
    load_image,
    preprocess,
)


OUTER_COLOR = QColor("#ff9d00")
HOLE_COLOR = QColor("#00c2ff")
CALIB_COLOR = QColor("#ff3333")


def cv_to_qimage(img_bgr: np.ndarray) -> QImage:
    """Konvertiert ein OpenCV BGR-Bild in ein QImage (RGB888)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return qimg.copy()  # eigene Kopie, damit der numpy-Puffer freigegeben werden kann


class ImageGraphicsView(QGraphicsView):
    """QGraphicsView, die Bild-Pixelkoordinaten 1:1 als Szenen-Koordinaten
    verwendet und bei Linksklick die Szenen-Position als Signal sendet.

    Unterstuetzt zusaetzlich einen "ROI-Modus": Statt einzelner Klicks
    (Kalibrierung) kann der Nutzer dann per Maus-Drag ein Rechteck
    aufziehen (Qt eingebautes Rubber-Band), dessen Endkoordinaten als
    roiFinished-Signal (in Bild-Pixelkoordinaten) gemeldet werden.
    """

    clicked = Signal(float, float)
    roiFinished = Signal(float, float, float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRenderHints(self.renderHints())
        self.setBackgroundBrush(QColor("#2b2b2b"))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._roi_mode = False
        self._last_rubber_rect: Optional[QRectF] = None
        self.rubberBandChanged.connect(self._on_rubber_band_changed)

    def set_roi_mode(self, enabled: bool) -> None:
        self._roi_mode = enabled
        self.setDragMode(
            QGraphicsView.DragMode.RubberBandDrag if enabled else QGraphicsView.DragMode.NoDrag
        )

    def _on_rubber_band_changed(self, viewport_rect, from_scene, to_scene) -> None:
        if viewport_rect.isNull():
            return
        self._last_rubber_rect = QRectF(from_scene, to_scene).normalized()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self._roi_mode:
            scene_pos = self.mapToScene(event.pos())
            self.clicked.emit(scene_pos.x(), scene_pos.y())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if (
            self._roi_mode
            and event.button() == Qt.MouseButton.LeftButton
            and self._last_rubber_rect is not None
        ):
            r = self._last_rubber_rect
            self._last_rubber_rect = None
            self.roiFinished.emit(r.left(), r.top(), r.right(), r.bottom())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.scene() is not None and not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class LabeledSlider(QWidget):
    """Schieberegler mit Beschriftung, der auch Fliesskommawerte und eine
    Schrittweite (z.B. fuer ungerade Kernelgroessen) unterstuetzt."""

    valueChanged = Signal(float)

    def __init__(
        self,
        title: str,
        minimum: float,
        maximum: float,
        value: float,
        step: float = 1.0,
        decimals: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._minimum = minimum
        self._step = step
        self._decimals = decimals
        self._title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(0)

        self.label = QLabel()
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        n_steps = max(1, round((maximum - minimum) / step))
        self.slider.setRange(0, n_steps)
        layout.addWidget(self.slider)

        self.set_value(value, emit=False)
        self.slider.valueChanged.connect(self._on_slider_changed)

    def _on_slider_changed(self, raw: int) -> None:
        value = self._minimum + raw * self._step
        self._update_label(value)
        self.valueChanged.emit(value)

    def _update_label(self, value: float) -> None:
        if self._decimals > 0:
            self.label.setText(f"{self._title}: {value:.{self._decimals}f}")
        else:
            self.label.setText(f"{self._title}: {int(round(value))}")

    def value(self) -> float:
        return self._minimum + self.slider.value() * self._step

    def set_value(self, value: float, emit: bool = True) -> None:
        raw = round((value - self._minimum) / self._step)
        self._update_label(self._minimum + raw * self._step)
        if emit:
            self.slider.setValue(raw)
        else:
            self.slider.blockSignals(True)
            self.slider.setValue(raw)
            self.slider.blockSignals(False)


class ImageToDxfWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ImageToDXF - Konturerkennung")
        self.resize(1280, 820)

        # --- Zustand ---------------------------------------------------
        self.image_bgr: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        self.contours: List[DetectedContour] = []
        self.mm_per_pixel: Optional[float] = None
        self._last_binary: Optional[np.ndarray] = None
        self.view_mode = "original"  # "original" | "processed"

        self.calibration_mode = False
        self.calib_points_img: List[Tuple[float, float]] = []
        self.calib_marker_items: list = []

        self.roi: Optional[Tuple[float, float, float, float]] = None
        self.roi_rect_item = None

        self.scene = QGraphicsScene(self)
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.contour_items: list = []

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Aufbau
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        # Linke Seite: Toolbar + Canvas
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)

        toolbar = QHBoxLayout()
        load_btn = QPushButton("Bild laden...")
        load_btn.clicked.connect(self.on_load_image)
        toolbar.addWidget(load_btn)

        toolbar.addSpacing(20)
        self.view_mode_group = QButtonGroup(self)
        rb_original = QRadioButton("Original")
        rb_original.setChecked(True)
        rb_original.toggled.connect(lambda checked: checked and self._set_view_mode("original"))
        rb_processed = QRadioButton("Verarbeitet (Schwarz/Weiss)")
        rb_processed.toggled.connect(lambda checked: checked and self._set_view_mode("processed"))
        self.view_mode_group.addButton(rb_original)
        self.view_mode_group.addButton(rb_processed)
        toolbar.addWidget(rb_original)
        toolbar.addWidget(rb_processed)
        toolbar.addStretch(1)
        left_layout.addLayout(toolbar)

        self.view = ImageGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.clicked.connect(self._on_view_clicked)
        self.view.roiFinished.connect(self._on_roi_finished)
        left_layout.addWidget(self.view, stretch=1)

        root_layout.addWidget(left, stretch=1)

        # Rechte Seite: Steuerungs-Panel (scrollbar)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(340)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        scroll.setWidget(panel)
        root_layout.addWidget(scroll)

        self._build_preprocess_panel(panel_layout)
        self._build_roi_panel(panel_layout)
        self._build_contour_panel(panel_layout)
        self._build_scale_panel(panel_layout)
        self._build_export_panel(panel_layout)
        panel_layout.addStretch(1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Kein Bild geladen.")

    def _build_preprocess_panel(self, parent_layout) -> None:
        box = QGroupBox("1. Vorverarbeitung")
        layout = QVBoxLayout(box)
        parent_layout.addWidget(box)

        method_row = QHBoxLayout()
        self.method_group = QButtonGroup(self)
        self.rb_binary = QRadioButton("Binaer")
        self.rb_binary.setChecked(True)
        self.rb_adaptive = QRadioButton("Adaptiv")
        self.rb_canny = QRadioButton("Canny")
        for rb in (self.rb_binary, self.rb_adaptive, self.rb_canny):
            self.method_group.addButton(rb)
            rb.toggled.connect(lambda checked: checked and self._on_param_change())
            method_row.addWidget(rb)
        layout.addLayout(method_row)

        self.invert_cb = QCheckBox("Invertieren (helle Kontur auf dunklem Grund)")
        self.invert_cb.toggled.connect(self._on_param_change)
        layout.addWidget(self.invert_cb)

        self.blur_slider = LabeledSlider("Weichzeichnen", 0, 21, 3, step=2)
        self.blur_slider.valueChanged.connect(self._on_param_change)
        layout.addWidget(self.blur_slider)

        self.thresh_slider = LabeledSlider("Schwelle (Binaer)", 0, 255, 128, step=1)
        self.thresh_slider.valueChanged.connect(self._on_param_change)
        layout.addWidget(self.thresh_slider)

        self.block_slider = LabeledSlider("Blockgroesse (Adaptiv)", 3, 199, 35, step=2)
        self.block_slider.valueChanged.connect(self._on_param_change)
        layout.addWidget(self.block_slider)

        self.c_slider = LabeledSlider("C-Wert (Adaptiv)", -20, 20, 5, step=1)
        self.c_slider.valueChanged.connect(self._on_param_change)
        layout.addWidget(self.c_slider)

        self.canny_low_slider = LabeledSlider("Canny niedrig", 0, 255, 50, step=1)
        self.canny_low_slider.valueChanged.connect(self._on_param_change)
        layout.addWidget(self.canny_low_slider)

        self.canny_high_slider = LabeledSlider("Canny hoch", 0, 255, 150, step=1)
        self.canny_high_slider.valueChanged.connect(self._on_param_change)
        layout.addWidget(self.canny_high_slider)

    def _build_roi_panel(self, parent_layout) -> None:
        box = QGroupBox("2. Auswahlbereich (optional)")
        layout = QVBoxLayout(box)
        parent_layout.addWidget(box)

        self.roi_draw_btn = QPushButton("Rahmen zeichnen")
        self.roi_draw_btn.setCheckable(True)
        self.roi_draw_btn.toggled.connect(self._on_toggle_roi_draw)
        layout.addWidget(self.roi_draw_btn)

        self.roi_active_cb = QCheckBox("Nur im Rahmen nach Konturen suchen")
        self.roi_active_cb.setEnabled(False)
        self.roi_active_cb.toggled.connect(self._on_param_change)
        layout.addWidget(self.roi_active_cb)

        roi_clear_btn = QPushButton("Rahmen entfernen")
        roi_clear_btn.clicked.connect(self.on_clear_roi)
        layout.addWidget(roi_clear_btn)

        self.roi_info_label = QLabel("Kein Auswahlbereich gesetzt.")
        self.roi_info_label.setWordWrap(True)
        self.roi_info_label.setStyleSheet("color: #888;")
        layout.addWidget(self.roi_info_label)

    def _build_contour_panel(self, parent_layout) -> None:
        box = QGroupBox("3. Konturerkennung")
        layout = QVBoxLayout(box)
        parent_layout.addWidget(box)

        self.min_area_slider = LabeledSlider("Mindestflaeche (px²)", 0, 5000, 50, step=10)
        self.min_area_slider.valueChanged.connect(self._on_param_change)
        layout.addWidget(self.min_area_slider)

        self.epsilon_slider = LabeledSlider(
            "Vereinfachung", 0.0, 0.02, 0.0, step=0.0005, decimals=4
        )
        self.epsilon_slider.valueChanged.connect(self._on_param_change)
        layout.addWidget(self.epsilon_slider)

        self.smooth_slider = LabeledSlider(
            "Glaettung / Abrunden", 0.0, 20.0, 0.0, step=0.5, decimals=1
        )
        self.smooth_slider.valueChanged.connect(self._on_param_change)
        layout.addWidget(self.smooth_slider)

        self.spacing_slider = LabeledSlider(
            "Max. Punktabstand (px, 0=aus)", 0.0, 100.0, 0.0, step=1.0, decimals=0
        )
        self.spacing_slider.valueChanged.connect(self._on_param_change)
        layout.addWidget(self.spacing_slider)

        self.holes_cb = QCheckBox("Loecher als eigene Konturen erkennen")
        self.holes_cb.setChecked(True)
        self.holes_cb.toggled.connect(self._on_param_change)
        layout.addWidget(self.holes_cb)

        self.live_preview_cb = QCheckBox("Live-Vorschau")
        self.live_preview_cb.setChecked(True)
        layout.addWidget(self.live_preview_cb)

        detect_btn = QPushButton("Konturen erkennen")
        detect_btn.clicked.connect(lambda: self.on_detect_contours(quiet=False))
        layout.addWidget(detect_btn)

        self.contour_info_label = QLabel("Noch keine Konturen erkannt.")
        self.contour_info_label.setWordWrap(True)
        self.contour_info_label.setStyleSheet("color: #2a6;")
        layout.addWidget(self.contour_info_label)

    def _build_scale_panel(self, parent_layout) -> None:
        box = QGroupBox("4. Skalierung (Pixel -> mm)")
        layout = QVBoxLayout(box)
        parent_layout.addWidget(box)

        row = QHBoxLayout()
        row.addWidget(QLabel("mm pro Pixel:"))
        self.scale_edit = QLineEdit()
        self.scale_edit.editingFinished.connect(self._on_scale_entry_change)
        row.addWidget(self.scale_edit)
        layout.addLayout(row)

        calib_btn = QPushButton("Kalibrieren (2 Punkte anklicken)")
        calib_btn.clicked.connect(self.on_start_calibration)
        layout.addWidget(calib_btn)

        self.scale_info_label = QLabel("Kein Skalierungsfaktor gesetzt.")
        self.scale_info_label.setWordWrap(True)
        self.scale_info_label.setStyleSheet("color: #06c;")
        layout.addWidget(self.scale_info_label)

    def _build_export_panel(self, parent_layout) -> None:
        box = QGroupBox("5. DXF Export")
        layout = QVBoxLayout(box)
        parent_layout.addWidget(box)

        export_btn = QPushButton("Als DXF exportieren...")
        export_btn.clicked.connect(self.on_export_dxf)
        layout.addWidget(export_btn)

    # ------------------------------------------------------------------
    # Bild laden
    # ------------------------------------------------------------------
    def on_load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Bild auswaehlen",
            "",
            "Bilder (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Alle Dateien (*.*)",
        )
        if not path:
            return
        try:
            image = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Fehler beim Laden", str(exc))
            return

        self.image_bgr = image
        self.image_path = path
        self.contours = []
        self.calib_points_img = []
        self.calibration_mode = False
        self._last_binary = None
        self.roi = None
        self.roi_rect_item = None
        self.roi_active_cb.setChecked(False)
        self.roi_active_cb.setEnabled(False)
        self.roi_draw_btn.setChecked(False)
        self.roi_info_label.setText("Kein Auswahlbereich gesetzt.")
        self.contour_info_label.setText("Noch keine Konturen erkannt.")
        self.status_bar.showMessage(
            f"Geladen: {os.path.basename(path)}  ({image.shape[1]} x {image.shape[0]} px)"
        )
        self._show_image(image)

    def _show_image(self, image_bgr: np.ndarray) -> None:
        qimg = cv_to_qimage(image_bgr)
        pixmap = QPixmap.fromImage(qimg)

        self.scene.clear()
        self.contour_items = []
        self.calib_marker_items = []
        self.roi_rect_item = None
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        self._draw_contours()

    # ------------------------------------------------------------------
    # Parameter -> Bildverarbeitung
    # ------------------------------------------------------------------
    def _preprocess_params(self) -> PreprocessParams:
        method = "binary"
        if self.rb_adaptive.isChecked():
            method = "adaptive"
        elif self.rb_canny.isChecked():
            method = "canny"
        return PreprocessParams(
            method=method,
            blur_ksize=int(self.blur_slider.value()),
            invert=self.invert_cb.isChecked(),
            thresh=int(self.thresh_slider.value()),
            block_size=int(self.block_slider.value()),
            c_value=int(self.c_slider.value()),
            canny_low=int(self.canny_low_slider.value()),
            canny_high=int(self.canny_high_slider.value()),
        )

    def _contour_params(self) -> ContourParams:
        return ContourParams(
            min_area_px=self.min_area_slider.value(),
            approx_epsilon_factor=self.epsilon_slider.value(),
            retrieve_holes=self.holes_cb.isChecked(),
            smooth_sigma=self.smooth_slider.value(),
            max_point_spacing_px=self.spacing_slider.value(),
        )

    def _current_roi_px(self) -> Optional[Tuple[float, float, float, float]]:
        """Liefert den aktiven ROI in Pixelkoordinaten (x1,y1,x2,y2),
        sofern einer gesetzt und die Checkbox aktiviert ist."""
        if self.roi is not None and self.roi_active_cb.isChecked():
            x1, y1, x2, y2 = self.roi
            return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        return None

    def _on_param_change(self, *_args) -> None:
        if self.image_bgr is None:
            return
        if self.live_preview_cb.isChecked():
            self.on_detect_contours(quiet=True)
        elif self.view_mode == "processed":
            self._update_processed_preview()

    def on_detect_contours(self, quiet: bool = False) -> None:
        if self.image_bgr is None:
            if not quiet:
                QMessageBox.information(self, "Kein Bild", "Bitte zuerst ein Bild laden.")
            return
        binary = preprocess(self.image_bgr, self._preprocess_params())
        self._last_binary = binary
        self.contours = find_contours(binary, self._contour_params(), roi=self._current_roi_px())

        n_outer = sum(1 for c in self.contours if not c.is_hole)
        n_holes = sum(1 for c in self.contours if c.is_hole)
        self.contour_info_label.setText(
            f"{len(self.contours)} Kontur(en) gefunden "
            f"({n_outer} aussen, {n_holes} Loecher)."
        )

        if self.view_mode == "processed":
            self._update_processed_preview()
        self._draw_contours()

    def _set_view_mode(self, mode: str) -> None:
        self.view_mode = mode
        if self.image_bgr is None:
            return
        if mode == "processed":
            self._update_processed_preview()
        else:
            qimg = cv_to_qimage(self.image_bgr)
            self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self._draw_contours()

    def _update_processed_preview(self) -> None:
        if self.image_bgr is None or self.pixmap_item is None:
            return
        self._last_binary = preprocess(self.image_bgr, self._preprocess_params())
        bgr = cv2.cvtColor(self._last_binary, cv2.COLOR_GRAY2BGR)
        qimg = cv_to_qimage(bgr)
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))

    def _draw_contours(self) -> None:
        for item in self.contour_items:
            self.scene.removeItem(item)
        self.contour_items = []

        if self.view_mode != "original":
            return

        for contour in self.contours:
            polygon = QPolygonF([QPointF(x, y) for x, y in contour.points])
            item = QGraphicsPolygonItem(polygon)
            color = HOLE_COLOR if contour.is_hole else OUTER_COLOR
            pen = QPen(color)
            pen.setWidth(0)  # kosmetischer Stift: bleibt beim Zoomen 1px breit
            item.setPen(pen)
            item.setBrush(Qt.BrushStyle.NoBrush)
            item.setZValue(10)
            self.scene.addItem(item)
            self.contour_items.append(item)

    # ------------------------------------------------------------------
    # Skalierung
    # ------------------------------------------------------------------
    def _on_scale_entry_change(self) -> None:
        text = self.scale_edit.text().strip().replace(",", ".")
        if not text:
            return
        try:
            value = float(text)
            if value <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Ungueltiger Wert", "Bitte eine Zahl > 0 eingeben.")
            return
        self.mm_per_pixel = value
        self.scale_info_label.setText(f"Skalierung gesetzt: {value:.6f} mm/px")

    def on_start_calibration(self) -> None:
        if self.image_bgr is None:
            QMessageBox.information(self, "Kein Bild", "Bitte zuerst ein Bild laden.")
            return
        self.roi_draw_btn.setChecked(False)  # ROI-Zeichnen und Kalibrierung schliessen sich aus
        self.calibration_mode = True
        self.calib_points_img = []
        self._clear_calib_markers()
        self.status_bar.showMessage(
            "Kalibrierung: Klicke auf zwei Punkte im Bild mit bekanntem Abstand."
        )

    def _clear_calib_markers(self) -> None:
        for item in self.calib_marker_items:
            self.scene.removeItem(item)
        self.calib_marker_items = []

    # ------------------------------------------------------------------
    # Auswahlbereich (ROI)
    # ------------------------------------------------------------------
    def _on_toggle_roi_draw(self, checked: bool) -> None:
        if checked and self.image_bgr is None:
            self.roi_draw_btn.setChecked(False)
            QMessageBox.information(self, "Kein Bild", "Bitte zuerst ein Bild laden.")
            return
        if checked:
            self.calibration_mode = False  # ROI-Zeichnen und Kalibrierung schliessen sich aus
            self._clear_calib_markers()
        self.view.set_roi_mode(checked)
        if checked:
            self.status_bar.showMessage(
                "Rahmen ziehen: Mit gedrueckter Maustaste ein Rechteck im Bild aufziehen."
            )
        else:
            self.status_bar.showMessage("Bereit.")

    def _on_roi_finished(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if self.image_bgr is None:
            return
        img_h, img_w = self.image_bgr.shape[:2]
        x1 = max(0.0, min(x1, img_w))
        x2 = max(0.0, min(x2, img_w))
        y1 = max(0.0, min(y1, img_h))
        y2 = max(0.0, min(y2, img_h))
        self.roi_draw_btn.setChecked(False)
        self.view.set_roi_mode(False)

        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            self.status_bar.showMessage("Rahmen zu klein, bitte erneut aufziehen.")
            return

        self.roi = (x1, y1, x2, y2)
        self.roi_active_cb.setEnabled(True)
        self.roi_active_cb.setChecked(True)
        self._draw_roi_item()

        w, h = abs(x2 - x1), abs(y2 - y1)
        self.roi_info_label.setText(
            f"Auswahlbereich: {w:.0f} x {h:.0f} px bei "
            f"({min(x1, x2):.0f}, {min(y1, y2):.0f})"
        )
        self.status_bar.showMessage("Auswahlbereich gesetzt.")
        self._on_param_change()

    def on_clear_roi(self) -> None:
        self.roi = None
        self.roi_active_cb.setChecked(False)
        self.roi_active_cb.setEnabled(False)
        self.roi_info_label.setText("Kein Auswahlbereich gesetzt.")
        if self.roi_rect_item is not None:
            self.scene.removeItem(self.roi_rect_item)
            self.roi_rect_item = None
        self._on_param_change()

    def _draw_roi_item(self) -> None:
        if self.roi_rect_item is not None:
            self.scene.removeItem(self.roi_rect_item)
            self.roi_rect_item = None
        if self.roi is None:
            return
        x1, y1, x2, y2 = self.roi
        rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        pen = QPen(QColor("#00ff6a"))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(0)
        item = self.scene.addRect(rect, pen)
        item.setZValue(15)
        self.roi_rect_item = item

    def _on_view_clicked(self, x: float, y: float) -> None:
        if not self.calibration_mode or self.image_bgr is None:
            return
        img_h, img_w = self.image_bgr.shape[:2]
        if x < 0 or y < 0 or x > img_w or y > img_h:
            return
        self.calib_points_img.append((x, y))

        marker = self.scene.addEllipse(
            x - 4, y - 4, 8, 8, QPen(CALIB_COLOR, 2)
        )
        marker.setZValue(20)
        self.calib_marker_items.append(marker)

        if len(self.calib_points_img) == 1:
            self.status_bar.showMessage(
                "Erster Punkt gesetzt. Jetzt den zweiten Punkt anklicken."
            )
        elif len(self.calib_points_img) == 2:
            p1, p2 = self.calib_points_img
            line = self.scene.addLine(p1[0], p1[1], p2[0], p2[1], QPen(CALIB_COLOR, 2))
            line.setZValue(20)
            self.calib_marker_items.append(line)
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        self.calibration_mode = False
        distance = None
        ok = False
        while True:
            text, ok = QInputDialog.getText(
                self,
                "Kalibrierung",
                "Welcher reale Abstand (in mm) liegt zwischen den beiden Punkten?\n"
                "(Komma oder Punkt als Dezimaltrennzeichen)",
                text="10",
            )
            if not ok:
                break
            try:
                distance = float(text.strip().replace(",", "."))
                if distance <= 0:
                    raise ValueError
                break
            except ValueError:
                QMessageBox.critical(self, "Ungueltiger Wert", "Bitte eine Zahl > 0 eingeben.")
        self._clear_calib_markers()
        if not ok or distance is None:
            self.status_bar.showMessage("Kalibrierung abgebrochen.")
            return
        try:
            mm_per_px = compute_scale_mm_per_pixel(
                self.calib_points_img[0], self.calib_points_img[1], distance
            )
        except ValueError as exc:
            QMessageBox.critical(self, "Fehler", str(exc))
            return
        self.mm_per_pixel = mm_per_px
        self.scale_edit.setText(f"{mm_per_px:.6f}")
        self.scale_info_label.setText(f"Skalierung gesetzt: {mm_per_px:.6f} mm/px")
        self.status_bar.showMessage("Kalibrierung abgeschlossen.")

    # ------------------------------------------------------------------
    # DXF Export
    # ------------------------------------------------------------------
    def on_export_dxf(self) -> None:
        if self.image_bgr is None:
            QMessageBox.information(self, "Kein Bild", "Bitte zuerst ein Bild laden.")
            return
        if not self.contours:
            QMessageBox.information(
                self, "Keine Konturen", "Bitte zuerst 'Konturen erkennen' ausfuehren."
            )
            return
        if not self.mm_per_pixel:
            QMessageBox.information(
                self,
                "Keine Skalierung",
                "Bitte zuerst einen Skalierungsfaktor setzen (Eingabe oder Kalibrierung).",
            )
            return

        default_name = "kontur.dxf"
        if self.image_path:
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            default_name = f"{base}.dxf"

        out_path, _ = QFileDialog.getSaveFileName(
            self, "DXF speichern unter", default_name, "DXF Datei (*.dxf)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".dxf"):
            out_path += ".dxf"

        try:
            count = contours_to_dxf(
                self.contours,
                mm_per_pixel=self.mm_per_pixel,
                image_height_px=self.image_bgr.shape[0],
                output_path=out_path,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Fehler beim Export", str(exc))
            return

        self.status_bar.showMessage(f"DXF exportiert: {out_path} ({count} Polylinien)")
        QMessageBox.information(
            self,
            "Export erfolgreich",
            f"{count} Konturen wurden nach\n{out_path}\nexportiert.",
        )


def main() -> None:
    app = QApplication(sys.argv)
    window = ImageToDxfWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
