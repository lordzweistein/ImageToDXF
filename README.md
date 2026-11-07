# ImageToDXF

Ein kleines Python-Tool (läuft auf macOS) mit **Qt-GUI** (via
[qtpy](https://github.com/spyder-ide/qtpy) + PySide6) zum:

1. Einlesen von Bildern (Fotos/Scans von Bauteilen, Skizzen, etc.)
2. Automatischen Erkennen von Konturen (inkl. Löchern) mit einstellbaren
   Parametern (Schwellenwert, Adaptiv, Canny-Kantenerkennung) –
   optional eingeschränkt auf einen per Maus aufgezogenen
   Auswahlbereich (ROI)
3. Kalibrieren eines Skalierungsfaktors (Pixel → mm) per Mausklick auf
   zwei Punkte mit bekanntem Abstand – oder durch direkte Eingabe eines
   Wertes
4. Export der erkannten Konturen als **DXF-Datei** (mm-Einheiten), die
   sich in jedem CAD-Programm öffnen lässt


## Installation (einmalig)

Wichtig: Ein modernes Python verwenden (nicht das alte Xcode/System-
Python). Am einfachsten über Homebrew:

```bash
brew install python@3.12
```

Dann im Projektordner:

```bash
cd ImageToDXF
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Installation unter Debian/Ubuntu (Linux)

Unter Linux werden zusätzlich zu den Python-Paketen noch einige
System-Bibliotheken benötigt (Qt/X11 für PySide6, OpenGL für OpenCV),
die nicht automatisch über `pip` mitinstalliert werden. Dafür gibt es
das Skript [`install_debian.sh`](./install_debian.sh), das:

1. die nötigen Debian-Pakete per `apt-get` installiert (u. a.
   `python3-venv`, `libxcb-cursor0`, `libxkbcommon-x11-0`, `libgl1`, ...),
2. ein Python-venv unter `.venv` anlegt,
3. `requirements.txt` darin installiert.

```bash
cd ImageToDXF
chmod +x install_debian.sh
./install_debian.sh
```

Danach wie gewohnt starten (siehe unten). Getestet/gedacht für Debian 12
"Bookworm" (Kernel 6.1), funktioniert aber auch auf Debian 11/13 und
i. d. R. auf Ubuntu. Die GUI benötigt eine grafische Oberfläche
(lokaler Desktop, `ssh -X` oder VNC/RDP) – auf einem reinen
Headless-Server ohne Display startet sie nicht.

## Starten

```bash
source .venv/bin/activate
python3 app.py
```

## Bedienung

1. **Bild laden...** – Foto/Scan auswählen (PNG, JPG, BMP, TIFF ...).
2. **Vorverarbeitung** (rechtes Panel, Abschnitt 1):
   - Methode wählen: *Binär* (fester Schwellenwert, gut für Tinte auf
     Papier), *Adaptiv* (gut bei ungleichmäßiger Beleuchtung) oder
     *Canny* (reine Kantenerkennung).
   - "Invertieren" aktivieren, falls die Kontur heller als der
     Hintergrund ist.
   - Mit den Schiebereglern Weichzeichnen/Schwellenwerte anpassen; bei
     aktivierter **Live-Vorschau** werden die Konturen laufend neu
     berechnet. Über "Original"/"Verarbeitet" oben lässt sich das
     Schwarz-Weiß-Zwischenbild zur Kontrolle anzeigen.
3. **Auswahlbereich / ROI** (Abschnitt 2, optional):
   - **Rahmen zeichnen** klicken, dann mit gedrückter Maustaste ein
     Rechteck über dem Bildbereich aufziehen, in dem gesucht werden soll
     (z. B. um Störungen am Bildrand oder andere Objekte auszublenden).
   - Nach dem Loslassen wird der Rahmen als gestricheltes grünes Rechteck
     angezeigt und die Checkbox **"Nur im Rahmen nach Konturen suchen"**
     automatisch aktiviert.
   - Die Checkbox lässt sich jederzeit aus-/einschalten, um zwischen
     "ganzes Bild" und "nur Rahmen" zu wechseln, ohne den Rahmen neu
     zeichnen zu müssen. **Rahmen entfernen** löscht ihn komplett.
   - Beim Laden eines neuen Bildes wird der Rahmen automatisch
     zurückgesetzt.
4. **Konturerkennung** (Abschnitt 3):
   - *Mindestfläche* filtert Rauschen/kleine Flecken heraus.
   - *Vereinfachung* reduziert die Anzahl der Punkte pro Kontur
     (Douglas-Peucker-Approximation), nützlich für glattere DXF-Polylinien.
   - *Löcher als eigene Konturen erkennen* erkennt z. B. Bohrungen
     innerhalb eines Bauteils als separate (innere) Konturen – sie
     werden in DXF auf einen eigenen Layer "LOECHER" gelegt, Außenkonturen
     auf den Layer "KONTUREN".
   - Button **Konturen erkennen** löst die Erkennung aus (auch ohne
     Live-Vorschau). Ist ein Auswahlbereich aktiv, wird nur innerhalb
     davon gesucht.
5. **Skalierung** (Abschnitt 4):
   - Entweder direkt einen Wert "mm pro Pixel" eingeben, falls bekannt
     (z. B. aus Scan-Auflösung), oder
   - **Kalibrieren**: Button klicken, dann zwei Punkte im Bild anklicken,
     deren realer Abstand bekannt ist (z. B. ein aufgelegtes Lineal oder
     ein Referenzobjekt bekannter Größe). Danach den Abstand in mm
     eingeben – der Skalierungsfaktor wird automatisch berechnet.
6. **DXF Export** (Abschnitt 5):
   - **Als DXF exportieren...** speichert alle aktuell erkannten
     Konturen (maßstabsgetreu in mm) in eine `.dxf`-Datei, die sich in
     CAD-Programmen (Fusion 360, SolidWorks, AutoCAD, FreeCAD, ...)
     öffnen/importieren lässt.

## Hinweise zur Genauigkeit

- Je gerader/senkrechter das Foto zur Bauteilebene aufgenommen wird,
  desto genauer passt die Skalierung über das ganze Bild (keine
  perspektivische Verzerrung).
- Für beste Ergebnisse das Kalibrierobjekt (Lineal o. ä.) in der
  gleichen Ebene wie die zu vermessende Kontur platzieren.
- Die Koordinaten werden beim Export von Bild- auf CAD-Konvention
  umgerechnet (Y-Achse gespiegelt), sodass die DXF-Geometrie in CAD
  korrekt orientiert erscheint.

## Dateien

- `app.py` – Qt-GUI-Anwendung (Hauptprogramm, `python3 app.py` starten)
- `contour_core.py` – reine Bildverarbeitungs-/DXF-Logik (keine GUI-Abhängigkeiten)
- `test_core.py` – Headless-Test der Kernlogik (`python3 test_core.py`)
- `requirements.txt` – Python-Abhängigkeiten
- `install_debian.sh` – Installationsskript für Debian/Ubuntu (System-Pakete + venv)
- `app_tkinter_backup.py` – alte Tkinter-Variante (nur zur Referenz, nicht mehr aktiv genutzt)
