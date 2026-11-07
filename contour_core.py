"""
Kernlogik fuer ImageToDXF.

Enthaelt alle Funktionen fuer:
- Bild laden
- Vorverarbeitung (Weichzeichnen, Schwellenwert / Canny / Adaptiv)
- Konturerkennung (inkl. Loecher ueber Hierarchie)
- Vereinfachung der Konturen (Polygon-Approximation)
- Skalierungsberechnung (Pixel -> mm)
- DXF-Export

Dieses Modul ist bewusst frei von GUI-Code, damit es unabhaengig
getestet werden kann (siehe test_core.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import ezdxf


PreprocessMethod = str  # "binary" | "adaptive" | "canny"


@dataclass
class PreprocessParams:
    method: PreprocessMethod = "binary"
    blur_ksize: int = 3          # ungerade Zahl, 0/1 = kein Weichzeichnen
    invert: bool = False
    # Binär
    thresh: int = 128
    # Adaptiv
    block_size: int = 35         # ungerade Zahl > 1
    c_value: int = 5
    # Canny
    canny_low: int = 50
    canny_high: int = 150


@dataclass
class ContourParams:
    min_area_px: float = 50.0
    approx_epsilon_factor: float = 0.0   # 0 = keine Vereinfachung
    retrieve_holes: bool = True          # Loecher als eigene Konturen behalten
    smooth_sigma: float = 0.0            # Glaettung (Gauss, in Punkten); 0 = aus
    max_point_spacing_px: float = 0.0    # Segmente laenger als dies werden unterteilt; 0 = aus


@dataclass
class DetectedContour:
    points: np.ndarray           # (N, 2) float, Pixelkoordinaten (x, y)
    is_hole: bool
    area_px: float


def load_image(path: str) -> np.ndarray:
    """Laedt ein Bild (BGR, wie OpenCV es erwartet)."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    return image


def _odd(n: int, minimum: int = 1) -> int:
    n = max(minimum, int(n))
    if n % 2 == 0:
        n += 1
    return n


def preprocess(image_bgr: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """Erzeugt ein binaeres Bild (0/255) aus dem Eingangsbild, auf dessen
    Basis die Konturen gesucht werden. Weisse Flaechen = Vordergrund."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if params.blur_ksize and params.blur_ksize > 1:
        k = _odd(params.blur_ksize)
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    if params.method == "canny":
        edges = cv2.Canny(gray, params.canny_low, params.canny_high)
        # Kanten leicht schliessen, damit Konturen durchgehend sind
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        binary = edges
    elif params.method == "adaptive":
        block = _odd(params.block_size, minimum=3)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, params.c_value,
        )
        binary = 255 - binary  # adaptiveThreshold liefert Hintergrund=weiss
    else:  # "binary"
        _, binary = cv2.threshold(
            gray, params.thresh, 255, cv2.THRESH_BINARY_INV
        )

    if params.invert:
        binary = 255 - binary

    return binary


def smooth_closed_contour(points: np.ndarray, sigma: float) -> np.ndarray:
    """Gauss-Glaettung einer geschlossenen Kontur (rundet Ecken/Treppen ab)."""
    n = len(points)
    if sigma <= 0 or n < 5:
        return points
    radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    out = np.empty_like(points, dtype=float)
    for axis in range(2):
        padded = np.take(points[:, axis], np.arange(-radius, n + radius), mode="wrap")
        out[:, axis] = np.convolve(padded, kernel, mode="valid")
    return out


def subdivide_closed_contour(points: np.ndarray, max_spacing: float) -> np.ndarray:
    """Unterteilt Segmente, die laenger als max_spacing sind (mehr Punkte)."""
    if max_spacing <= 0:
        return points
    result = []
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        parts = max(1, int(math.ceil(np.hypot(*(b - a)) / max_spacing)))
        for k in range(parts):
            result.append(a + (b - a) * (k / parts))
    return np.array(result, dtype=float)


def find_contours(
    binary_image: np.ndarray,
    params: ContourParams,
    roi: Optional[Tuple[float, float, float, float]] = None,
) -> List[DetectedContour]:
    """Findet Konturen inkl. Loch-Erkennung (ueber die Hierarchie von
    cv2.findContours) und filtert/vereinfacht sie.

    roi: optionaler Auswahlbereich (x1, y1, x2, y2) in Pixelkoordinaten.
    Wenn gesetzt, wird die Kontursuche auf diesen Ausschnitt beschraenkt;
    die zurueckgegebenen Konturpunkte liegen trotzdem wieder in den
    Koordinaten des vollstaendigen Bildes (Offset wird zurueckgerechnet).
    """
    h, w = binary_image.shape[:2]
    offset_x, offset_y = 0, 0
    search_image = binary_image

    if roi is not None:
        x1, y1, x2, y2 = roi
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = int(round(max(0, min(x1, w))))
        x2 = int(round(max(0, min(x2, w))))
        y1 = int(round(max(0, min(y1, h))))
        y2 = int(round(max(0, min(y2, h))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return []
        search_image = binary_image[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1

    mode = cv2.RETR_TREE if params.retrieve_holes else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(
        search_image, mode, cv2.CHAIN_APPROX_NONE
    )

    result: List[DetectedContour] = []
    if hierarchy is None:
        return result
    hierarchy = hierarchy[0]  # (N, 4): next, prev, first_child, parent

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < params.min_area_px:
            continue

        if params.smooth_sigma > 0:
            smoothed = smooth_closed_contour(
                cnt.reshape(-1, 2).astype(float), params.smooth_sigma
            )
            cnt = smoothed.astype(np.float32).reshape(-1, 1, 2)

        if params.approx_epsilon_factor > 0:
            epsilon = params.approx_epsilon_factor * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)

        points = cnt.reshape(-1, 2).astype(float)
        if len(points) < 3:
            continue
        points = subdivide_closed_contour(points, params.max_point_spacing_px)

        if offset_x or offset_y:
            points = points + np.array([offset_x, offset_y], dtype=float)

        parent = hierarchy[idx][3]
        # Gerade Verschachtelungstiefe (0, 2, 4, ...) = Aussenkontur,
        # ungerade (1, 3, 5, ...) = Loch.
        depth = 0
        p = parent
        while p != -1:
            depth += 1
            p = hierarchy[p][3]
        is_hole = params.retrieve_holes and (depth % 2 == 1)

        result.append(DetectedContour(points=points, is_hole=is_hole, area_px=area))

    return result


def compute_scale_mm_per_pixel(
    p1: Tuple[float, float], p2: Tuple[float, float], real_distance_mm: float
) -> float:
    """Berechnet mm pro Pixel aus zwei Bildpunkten und der bekannten
    realen Distanz zwischen ihnen."""
    pixel_distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if pixel_distance <= 0:
        raise ValueError("Die beiden Kalibrierpunkte duerfen nicht identisch sein.")
    if real_distance_mm <= 0:
        raise ValueError("Die reale Distanz muss groesser als 0 sein.")
    return real_distance_mm / pixel_distance


def contours_to_dxf(
    contours: List[DetectedContour],
    mm_per_pixel: float,
    image_height_px: int,
    output_path: str,
    layer_outer: str = "KONTUREN",
    layer_holes: str = "LOECHER",
    dxf_version: str = "R2010",
) -> int:
    """Schreibt die Konturen als geschlossene LWPOLYLINE-Entities in eine
    DXF-Datei. Pixel werden in mm umgerechnet, die Y-Achse wird gespiegelt
    (Bildursprung oben-links -> CAD-Ursprung unten-links).

    Gibt die Anzahl geschriebener Polylinien zurueck.
    """
    doc = ezdxf.new(dxf_version)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    if layer_outer not in doc.layers:
        doc.layers.add(layer_outer, color=7)
    if layer_holes not in doc.layers:
        doc.layers.add(layer_holes, color=1)

    count = 0
    for contour in contours:
        pts_mm = [
            (x * mm_per_pixel, (image_height_px - y) * mm_per_pixel)
            for x, y in contour.points
        ]
        layer = layer_holes if contour.is_hole else layer_outer
        msp.add_lwpolyline(pts_mm, close=True, dxfattribs={"layer": layer})
        count += 1

    doc.saveas(output_path)
    return count
