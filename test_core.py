"""
Headless-Test der Kernlogik (ohne GUI).

Erzeugt ein synthetisches Testbild (Rechteck mit einem Loch sowie ein
Kreis), fuehrt die komplette Pipeline aus (Vorverarbeitung ->
Konturerkennung -> Skalierung -> DXF-Export) und prueft das Ergebnis.
"""
import os
import tempfile

import cv2
import ezdxf
import numpy as np

from contour_core import (
    ContourParams,
    PreprocessParams,
    compute_scale_mm_per_pixel,
    contours_to_dxf,
    find_contours,
    preprocess,
)


def make_test_image() -> np.ndarray:
    # Weisser Hintergrund mit dunklen Formen (z.B. Tinte auf Papier,
    # der Standardfall fuer die "Binaer"-Vorverarbeitung).
    img = np.full((400, 500, 3), 255, dtype=np.uint8)
    # Gefuelltes Rechteck mit einem (hellen) quadratischen Loch darin
    cv2.rectangle(img, (50, 50), (250, 250), (0, 0, 0), thickness=-1)
    cv2.rectangle(img, (120, 120), (180, 180), (255, 255, 255), thickness=-1)
    # Zusaetzlicher Kreis (separate Aussenkontur)
    cv2.circle(img, (380, 180), 60, (0, 0, 0), thickness=-1)
    return img


def main() -> None:
    img = make_test_image()

    binary = preprocess(img, PreprocessParams(method="binary", thresh=127))
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})

    contours = find_contours(binary, ContourParams(min_area_px=20, retrieve_holes=True))
    n_outer = sum(1 for c in contours if not c.is_hole)
    n_holes = sum(1 for c in contours if c.is_hole)
    print(f"Gefundene Konturen: {len(contours)} (aussen={n_outer}, loecher={n_holes})")
    assert n_outer == 2, f"Erwartet 2 Aussenkonturen (Rechteck+Kreis), bekommen {n_outer}"
    assert n_holes == 1, f"Erwartet 1 Loch, bekommen {n_holes}"

    # ROI-Test: Auswahlbereich, der nur den Kreis (rechts im Bild) umschliesst
    roi_contours = find_contours(
        binary, ContourParams(min_area_px=20, retrieve_holes=True), roi=(300, 100, 460, 260)
    )
    print(f"Konturen im ROI: {len(roi_contours)}")
    assert len(roi_contours) == 1, f"Erwartet 1 Kontur im ROI (Kreis), bekommen {len(roi_contours)}"
    # Koordinaten muessen wieder im vollen Bild liegen (Kreis-Mittelpunkt ~ (380, 180))
    cx, cy = roi_contours[0].points.mean(axis=0)
    assert 350 < cx < 410 and 150 < cy < 210, f"ROI-Konturpunkte falsch zurueckgerechnet: ({cx}, {cy})"

    # Skalierung: 100 px Abstand entsprechen angenommen 50 mm -> 0.5 mm/px
    mm_per_px = compute_scale_mm_per_pixel((0, 0), (100, 0), 50.0)
    assert abs(mm_per_px - 0.5) < 1e-9
    print(f"mm pro Pixel: {mm_per_px}")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "test.dxf")
        count = contours_to_dxf(contours, mm_per_px, img.shape[0], out_path)
        assert count == len(contours)
        assert os.path.exists(out_path)

        doc = ezdxf.readfile(out_path)
        msp = doc.modelspace()
        polylines = list(msp.query("LWPOLYLINE"))
        print(f"DXF enthaelt {len(polylines)} LWPOLYLINE-Entities")
        assert len(polylines) == len(contours)
        for pl in polylines:
            assert pl.closed

    print("ALLE TESTS ERFOLGREICH")


if __name__ == "__main__":
    main()
