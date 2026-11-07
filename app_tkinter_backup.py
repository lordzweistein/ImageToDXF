"""
ImageToDXF - GUI-Anwendung

Liest Bilder ein, erkennt Konturen automatisch (mit einstellbaren
Parametern), erlaubt eine Pixel->mm Kalibrierung per Mausklick und
exportiert die erkannten Konturen als DXF-Datei fuer CAD-Programme.

Start:
    python3 app.py
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

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


CANVAS_BG = "#2b2b2b"


class ImageToDxfApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ImageToDXF - Konturerkennung")
        self.geometry("1200x780")
        self.minsize(980, 620)

        # --- Zustand -------------------------------------------------
        self.image_bgr: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        self.contours: List[DetectedContour] = []
        self.mm_per_pixel: Optional[float] = None

        self.view_mode = tk.StringVar(value="original")  # "original" | "processed"
        self.live_preview = tk.BooleanVar(value=True)

        # Canvas-Darstellung
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.tk_image: Optional[ImageTk.PhotoImage] = None

        # Kalibrierung
        self.calibration_mode = False
        self.calib_points_img: List[Tuple[float, float]] = []
        self.calib_marker_ids: List[int] = []

        self._build_ui()
        self.bind("<Configure>", self._on_resize)

    # ------------------------------------------------------------------
    # UI Aufbau
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)

        # Linke Seite: Canvas + Statusleiste
        left = ttk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(left)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Button(toolbar, text="Bild laden...", command=self.on_load_image).pack(side=tk.LEFT)
        ttk.Radiobutton(
            toolbar, text="Original", variable=self.view_mode, value="original",
            command=self.redraw
        ).pack(side=tk.LEFT, padx=(20, 4))
        ttk.Radiobutton(
            toolbar, text="Verarbeitet (Schwarz/Weiss)", variable=self.view_mode,
            value="processed", command=self.redraw
        ).pack(side=tk.LEFT, padx=4)

        self.canvas = tk.Canvas(left, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        self.status_var = tk.StringVar(value="Kein Bild geladen.")
        ttk.Label(left, textvariable=self.status_var, anchor="w").pack(
            side=tk.BOTTOM, fill=tk.X, padx=8, pady=4
        )

        # Rechte Seite: Steuerungs-Panel
        panel = ttk.Frame(root, width=320)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
        panel.pack_propagate(False)

        self._build_preprocess_panel(panel)
        self._build_contour_panel(panel)
        self._build_scale_panel(panel)
        self._build_export_panel(panel)

    def _slider(self, parent, label, var, frm, to, resolution=1, on_change=None):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
        scale = tk.Scale(
            row, from_=frm, to=to, orient=tk.HORIZONTAL, variable=var,
            resolution=resolution, showvalue=True,
            command=lambda _e: self._on_param_change()
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return scale

    def _build_preprocess_panel(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="1. Vorverarbeitung")
        box.pack(fill=tk.X, pady=(0, 10))

        self.method_var = tk.StringVar(value="binary")
        method_row = ttk.Frame(box)
        method_row.pack(fill=tk.X, pady=2)
        for text, value in [("Binaer", "binary"), ("Adaptiv", "adaptive"), ("Canny", "canny")]:
            ttk.Radiobutton(
                method_row, text=text, variable=self.method_var, value=value,
                command=self._on_param_change
            ).pack(side=tk.LEFT, padx=2)

        self.invert_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            box, text="Invertieren (helle Kontur auf dunklem Grund)",
            variable=self.invert_var, command=self._on_param_change
        ).pack(anchor="w", pady=2)

        self.blur_var = tk.IntVar(value=3)
        self._slider(box, "Weichzeichnen", self.blur_var, 0, 21, resolution=2)

        self.thresh_var = tk.IntVar(value=128)
        self._slider(box, "Schwelle (Binaer)", self.thresh_var, 0, 255)

        self.block_var = tk.IntVar(value=35)
        self._slider(box, "Blockgroesse (Adaptiv)", self.block_var, 3, 199, resolution=2)

        self.c_var = tk.IntVar(value=5)
        self._slider(box, "C-Wert (Adaptiv)", self.c_var, -20, 20)

        self.canny_low_var = tk.IntVar(value=50)
        self._slider(box, "Canny niedrig", self.canny_low_var, 0, 255)

        self.canny_high_var = tk.IntVar(value=150)
        self._slider(box, "Canny hoch", self.canny_high_var, 0, 255)

    def _build_contour_panel(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="2. Konturerkennung")
        box.pack(fill=tk.X, pady=(0, 10))

        self.min_area_var = tk.DoubleVar(value=50)
        self._slider(box, "Mindestflaeche (px²)", self.min_area_var, 0, 5000, resolution=10)

        self.epsilon_var = tk.DoubleVar(value=0.0)
        self._slider(box, "Vereinfachung", self.epsilon_var, 0.0, 0.02, resolution=0.0005)

        self.holes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            box, text="Loecher als eigene Konturen erkennen",
            variable=self.holes_var, command=self._on_param_change
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            box, text="Live-Vorschau", variable=self.live_preview
        ).pack(anchor="w", pady=2)

        ttk.Button(
            box, text="Konturen erkennen", command=self.on_detect_contours
        ).pack(fill=tk.X, pady=(6, 2))

        self.contour_info_var = tk.StringVar(value="Noch keine Konturen erkannt.")
        ttk.Label(box, textvariable=self.contour_info_var, wraplength=290, foreground="#2a6").pack(
            anchor="w", pady=2
        )

    def _build_scale_panel(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="3. Skalierung (Pixel -> mm)")
        box.pack(fill=tk.X, pady=(0, 10))

        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="mm pro Pixel:").pack(side=tk.LEFT)
        self.scale_var = tk.StringVar(value="")
        entry = ttk.Entry(row, textvariable=self.scale_var, width=12)
        entry.pack(side=tk.LEFT, padx=6)
        entry.bind("<Return>", self._on_scale_entry_change)
        entry.bind("<FocusOut>", self._on_scale_entry_change)

        ttk.Button(
            box, text="Kalibrieren (2 Punkte anklicken)",
            command=self.on_start_calibration
        ).pack(fill=tk.X, pady=(6, 2))

        self.scale_info_var = tk.StringVar(
            value="Kein Skalierungsfaktor gesetzt."
        )
        ttk.Label(box, textvariable=self.scale_info_var, wraplength=290, foreground="#06c").pack(
            anchor="w", pady=2
        )

    def _build_export_panel(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="4. DXF Export")
        box.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            box, text="Als DXF exportieren...", command=self.on_export_dxf
        ).pack(fill=tk.X, pady=4)

    # ------------------------------------------------------------------
    # Bild laden
    # ------------------------------------------------------------------
    def on_load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Bild auswaehlen",
            filetypes=[
                ("Bilder", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Alle Dateien", "*.*"),
            ],
        )
        if not path:
            return
        try:
            image = load_image(path)
        except Exception as exc:
            messagebox.showerror("Fehler beim Laden", str(exc))
            return

        self.image_bgr = image
        self.image_path = path
        self.contours = []
        self.calib_points_img = []
        self.calibration_mode = False
        self.contour_info_var.set("Noch keine Konturen erkannt.")
        self.status_var.set(
            f"Geladen: {os.path.basename(path)}  ({image.shape[1]} x {image.shape[0]} px)"
        )
        self.redraw()

    # ------------------------------------------------------------------
    # Parameter -> Bildverarbeitung
    # ------------------------------------------------------------------
    def _preprocess_params(self) -> PreprocessParams:
        return PreprocessParams(
            method=self.method_var.get(),
            blur_ksize=self.blur_var.get(),
            invert=self.invert_var.get(),
            thresh=self.thresh_var.get(),
            block_size=self.block_var.get(),
            c_value=self.c_var.get(),
            canny_low=self.canny_low_var.get(),
            canny_high=self.canny_high_var.get(),
        )

    def _contour_params(self) -> ContourParams:
        return ContourParams(
            min_area_px=self.min_area_var.get(),
            approx_epsilon_factor=self.epsilon_var.get(),
            retrieve_holes=self.holes_var.get(),
        )

    def _on_param_change(self) -> None:
        if self.image_bgr is None:
            return
        if self.live_preview.get():
            self.on_detect_contours(quiet=True)
        else:
            self.redraw()

    def on_detect_contours(self, quiet: bool = False) -> None:
        if self.image_bgr is None:
            if not quiet:
                messagebox.showinfo("Kein Bild", "Bitte zuerst ein Bild laden.")
            return
        binary = preprocess(self.image_bgr, self._preprocess_params())
        self._last_binary = binary
        self.contours = find_contours(binary, self._contour_params())

        n_outer = sum(1 for c in self.contours if not c.is_hole)
        n_holes = sum(1 for c in self.contours if c.is_hole)
        self.contour_info_var.set(
            f"{len(self.contours)} Kontur(en) gefunden "
            f"({n_outer} aussen, {n_holes} Loecher)."
        )
        self.redraw()

    # ------------------------------------------------------------------
    # Skalierung
    # ------------------------------------------------------------------
    def _on_scale_entry_change(self, _event=None) -> None:
        text = self.scale_var.get().strip().replace(",", ".")
        if not text:
            return
        try:
            value = float(text)
            if value <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ungueltiger Wert", "Bitte eine Zahl > 0 eingeben.")
            return
        self.mm_per_pixel = value
        self.scale_info_var.set(f"Skalierung gesetzt: {value:.6f} mm/px")

    def on_start_calibration(self) -> None:
        if self.image_bgr is None:
            messagebox.showinfo("Kein Bild", "Bitte zuerst ein Bild laden.")
            return
        self.calibration_mode = True
        self.calib_points_img = []
        self._clear_calib_markers()
        self.status_var.set(
            "Kalibrierung: Klicke auf zwei Punkte im Bild mit bekanntem Abstand."
        )

    def _clear_calib_markers(self) -> None:
        for item_id in self.calib_marker_ids:
            self.canvas.delete(item_id)
        self.calib_marker_ids = []

    def _on_canvas_click(self, event) -> None:
        if not self.calibration_mode or self.image_bgr is None:
            return
        img_point = self._canvas_to_image_coords(event.x, event.y)
        if img_point is None:
            return
        self.calib_points_img.append(img_point)

        marker = self.canvas.create_oval(
            event.x - 4, event.y - 4, event.x + 4, event.y + 4,
            outline="#ff3333", width=2
        )
        self.calib_marker_ids.append(marker)

        if len(self.calib_points_img) == 1:
            self.status_var.set("Erster Punkt gesetzt. Jetzt den zweiten Punkt anklicken.")
        elif len(self.calib_points_img) == 2:
            self.canvas.create_line(
                *self._image_to_canvas_coords(*self.calib_points_img[0]),
                *self._image_to_canvas_coords(*self.calib_points_img[1]),
                fill="#ff3333", width=2
            )
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        self.calibration_mode = False
        distance = simpledialog.askfloat(
            "Kalibrierung",
            "Welcher reale Abstand (in mm) liegt zwischen den beiden Punkten?",
            minvalue=0.0001,
            parent=self,
        )
        self._clear_calib_markers()
        self.redraw()
        if distance is None:
            self.status_var.set("Kalibrierung abgebrochen.")
            return
        try:
            mm_per_px = compute_scale_mm_per_pixel(
                self.calib_points_img[0], self.calib_points_img[1], distance
            )
        except ValueError as exc:
            messagebox.showerror("Fehler", str(exc))
            return
        self.mm_per_pixel = mm_per_px
        self.scale_var.set(f"{mm_per_px:.6f}")
        self.scale_info_var.set(f"Skalierung gesetzt: {mm_per_px:.6f} mm/px")
        self.status_var.set("Kalibrierung abgeschlossen.")

    # ------------------------------------------------------------------
    # DXF Export
    # ------------------------------------------------------------------
    def on_export_dxf(self) -> None:
        if self.image_bgr is None:
            messagebox.showinfo("Kein Bild", "Bitte zuerst ein Bild laden.")
            return
        if not self.contours:
            messagebox.showinfo(
                "Keine Konturen", "Bitte zuerst 'Konturen erkennen' ausfuehren."
            )
            return
        if not self.mm_per_pixel:
            messagebox.showinfo(
                "Keine Skalierung",
                "Bitte zuerst einen Skalierungsfaktor setzen (Eingabe oder Kalibrierung).",
            )
            return

        default_name = "kontur.dxf"
        if self.image_path:
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            default_name = f"{base}.dxf"

        out_path = filedialog.asksaveasfilename(
            title="DXF speichern unter",
            defaultextension=".dxf",
            initialfile=default_name,
            filetypes=[("DXF Datei", "*.dxf")],
        )
        if not out_path:
            return

        try:
            count = contours_to_dxf(
                self.contours,
                mm_per_pixel=self.mm_per_pixel,
                image_height_px=self.image_bgr.shape[0],
                output_path=out_path,
            )
        except Exception as exc:
            messagebox.showerror("Fehler beim Export", str(exc))
            return

        self.status_var.set(f"DXF exportiert: {out_path} ({count} Polylinien)")
        messagebox.showinfo(
            "Export erfolgreich", f"{count} Konturen wurden nach\n{out_path}\nexportiert."
        )

    # ------------------------------------------------------------------
    # Zeichnen / Koordinaten-Umrechnung
    # ------------------------------------------------------------------
    def _on_resize(self, _event) -> None:
        self.redraw()

    def _current_display_image(self) -> np.ndarray:
        if self.view_mode.get() == "processed" and hasattr(self, "_last_binary"):
            return cv2.cvtColor(self._last_binary, cv2.COLOR_GRAY2BGR)
        return self.image_bgr

    def redraw(self) -> None:
        self.canvas.delete("all")
        if self.image_bgr is None:
            return

        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        img = self._current_display_image()
        img_h, img_w = img.shape[:2]

        scale = min(canvas_w / img_w, canvas_h / img_h)
        scale = min(scale, 1.0) if scale <= 0 else scale
        disp_w, disp_h = max(1, int(img_w * scale)), max(1, int(img_h * scale))
        offset_x = (canvas_w - disp_w) // 2
        offset_y = (canvas_h - disp_h) // 2

        self.display_scale = scale
        self.display_offset = (offset_x, offset_y)

        display_img = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.tk_image)

        if self.view_mode.get() == "original":
            for contour in self.contours:
                color = "#ff9d00" if not contour.is_hole else "#00c2ff"
                canvas_pts = []
                for x, y in contour.points:
                    cx, cy = self._image_to_canvas_coords(x, y)
                    canvas_pts.extend([cx, cy])
                if len(canvas_pts) >= 6:
                    self.canvas.create_polygon(
                        canvas_pts, outline=color, fill="", width=2
                    )

    def _image_to_canvas_coords(self, x: float, y: float) -> Tuple[float, float]:
        ox, oy = self.display_offset
        return ox + x * self.display_scale, oy + y * self.display_scale

    def _canvas_to_image_coords(self, cx: float, cy: float) -> Optional[Tuple[float, float]]:
        if self.image_bgr is None or self.display_scale == 0:
            return None
        ox, oy = self.display_offset
        x = (cx - ox) / self.display_scale
        y = (cy - oy) / self.display_scale
        img_h, img_w = self.image_bgr.shape[:2]
        if x < 0 or y < 0 or x > img_w or y > img_h:
            return None
        return x, y


def main() -> None:
    app = ImageToDxfApp()
    app.mainloop()


if __name__ == "__main__":
    main()
