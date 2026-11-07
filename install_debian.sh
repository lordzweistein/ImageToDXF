#!/usr/bin/env bash
#
# install_debian.sh
#
# Installiert alle System- und Python-Abhaengigkeiten, die ImageToDXF
# (Qt/PySide6-GUI + OpenCV) auf Debian benoetigt, und legt ein
# Python-venv mit den noetigen Paketen an.
#
# Getestet gedacht fuer Debian 12 "Bookworm" (Kernel 6.1) sowie Debian 11
# "Bullseye" / Debian 13 "Trixie" (apt-basierte Systeme). Auf Ubuntu
# funktionieren dieselben Paketnamen in der Regel ebenfalls.
#
# Nutzung:
#   chmod +x install_debian.sh
#   ./install_debian.sh
#
# Das Skript fragt bei Bedarf per "sudo" nach dem Passwort fuer die
# Systempaket-Installation. Muss im Projektordner (ImageToDXF/) liegen
# bzw. ausgefuehrt werden, da hier auch das Python-venv ".venv" angelegt
# und requirements.txt installiert wird.
#
# WICHTIG: Dieses Skript braucht "bash" (wegen "set -o pipefail" und
# BASH_SOURCE). Falls es versehentlich mit "sh install_debian.sh"
# gestartet wird, startet es sich unten automatisch mit "bash" neu.
# Debian verlinkt /bin/sh auf "dash", welches "pipefail" nicht kennt
# und sonst mit "set: Illegal option -o pipefail" abbrechen wuerde.

if [ -z "${BASH_VERSION:-}" ]; then
    if command -v bash >/dev/null 2>&1; then
        exec bash "$0" "$@"
    else
        echo "FEHLER: 'bash' wurde nicht gefunden. Bitte zuerst installieren:" >&2
        echo "  apt-get update && apt-get install -y bash" >&2
        exit 1
    fi
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== ImageToDXF: Installation unter Debian/Ubuntu ==="
echo

if ! command -v apt-get >/dev/null 2>&1; then
    echo "FEHLER: 'apt-get' wurde nicht gefunden. Dieses Skript ist fuer" >&2
    echo "Debian/Ubuntu-basierte Systeme gedacht." >&2
    exit 1
fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

echo "--- 1/4: Paketquellen aktualisieren ---"
$SUDO apt-get update

echo
echo "--- 2/4: Python-Grundlagen installieren ---"
$SUDO apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential

echo
echo "--- 3/4: System-Bibliotheken fuer Qt (PySide6) und OpenCV installieren ---"
# Diese Pakete werden von PySide6 (Qt "xcb"-Plattform-Plugin) sowie von
# opencv-python zur Laufzeit benoetigt, sind aber nicht Teil der
# Python-Wheels. Ohne sie meldet die App typischerweise:
#   "qt.qpa.plugin: Could not load the Qt platform plugin "xcb""
$SUDO apt-get install -y \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libx11-xcb1 \
    libxrender1 \
    libxext6 \
    libxi6 \
    libsm6 \
    libice6 \
    libfontconfig1 \
    libfreetype6 \
    libdbus-1-3 \
    libglib2.0-0 \
    libgl1 \
    libegl1 \
    libopengl0 \
    libgomp1

echo
echo "--- 4/4: Python-venv anlegen und Projekt-Abhaengigkeiten installieren ---"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Neues venv unter .venv angelegt."
else
    echo "Vorhandenes venv unter .venv wird weiterverwendet."
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate

echo
echo "=== Fertig! ==="
echo
echo "Starten der App:"
echo "  cd \"$SCRIPT_DIR\""
echo "  source .venv/bin/activate"
echo "  python3 app.py"
echo
echo "Hinweis: Fuer die GUI wird eine grafische Oberflaeche benoetigt"
echo "(lokaler Desktop, X11-Forwarding per 'ssh -X', oder VNC/RDP)."
echo "Auf einem reinen Headless-Server ohne Display startet die GUI nicht."
