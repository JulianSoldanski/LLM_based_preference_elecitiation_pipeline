"""Bereinigt den SoSci-Export des Fragebogens.

Entfernt alle Teilnahmen, bei denen mindestens ein Paarvergleich fehlt oder
die auf einer der Fallseiten (Seite 4-6) weniger als 30 Sekunden verbracht
haben. Alles andere bleibt unverändert: gleiche Spalten, Beschriftungszeile,
Werte und Format (UTF-8 mit BOM, Semikolon, CRLF).

    python3 src/clean_interviews.py results/InterviewResults/data_....csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Je Richtungsfrage: Antwortcode -> Intensitätsitem dieser Richtung.
# 3 = gleich wichtig, dann wird keine Intensität abgefragt. Die Codes sind am
# Export geprüft; die Beschriftung "Kosten VS Liefertreue" in C203/C303 nennt
# die Kriterien andersherum als die Codes.
VERGLEICHE = {
    "C101": {"1": "C104", "2": "C105", "3": None},  # Liefertreue / Flexibilität
    "C102": {"1": "C107", "2": "C109", "3": None},  # Kosten / Flexibilität
    "C103": {"1": "C108", "2": "C106", "3": None},  # Liefertreue / Kosten
    "C201": {"1": "C204", "2": "C207", "3": None},
    "C202": {"1": "C209", "2": "C206", "3": None},
    "C203": {"1": "C205", "2": "C208", "3": None},
    "C301": {"1": "C307", "2": "C310", "3": None},
    "C302": {"1": "C308", "2": "C309", "3": None},
    "C303": {"1": "C304", "2": "C306", "3": None},
}

# Verweildauer (Sekunden) auf den drei Fallseiten. Wer auf einer davon kürzer
# blieb, kann die Aussagen zum Fall kaum gelesen haben.
FALLSEITEN = {4: "TIME004", 5: "TIME005", 6: "TIME006"}
MIN_SEKUNDEN = 30


def fehlende_vergleiche(zeile: dict[str, str]) -> list[str]:
    """Fehlende Items einer Teilnahme; leer, wenn alle Vergleiche vollständig sind.

    Geprüft wird nur das Intensitätsitem der gewählten Richtung. Das Item der
    Gegenrichtung kann einen Wert tragen, wenn die Richtung nachträglich
    geändert wurde, und zählt deshalb nicht.
    """
    fehlend = []
    for richtung, intensitaet in VERGLEICHE.items():
        code = zeile[richtung].strip()
        if code not in intensitaet:
            fehlend.append(richtung)
        elif intensitaet[code] and not zeile[intensitaet[code]].strip():
            fehlend.append(intensitaet[code])
    return fehlend


def zu_schnelle_seiten(zeile: dict[str, str]) -> list[str]:
    """Fallseiten mit weniger als MIN_SEKUNDEN Verweildauer; fehlende Zeit zählt als 0."""
    zeiten = {seite: int(zeile[spalte].strip() or 0) for seite, spalte in FALLSEITEN.items()}
    return [f"Seite {seite} ({sek} s)" for seite, sek in zeiten.items() if sek < MIN_SEKUNDEN]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Entfernt Teilnahmen mit fehlenden Paarvergleichen oder zu kurzer Verweildauer.")
    parser.add_argument("export", type=Path, help="SoSci-CSV (data_*.csv)")
    parser.add_argument("--out", type=Path, help="Zieldatei (Default <export>_clean.csv)")
    args = parser.parse_args(argv)

    with args.export.open(encoding="utf-8-sig", newline="") as datei:
        kopf, *zeilen = list(csv.reader(datei, delimiter=";"))

    beschriftung, behalten, entfernt = [], [], []
    for zeile in zeilen:
        werte = dict(zip(kopf, zeile))
        # Die Beschriftungszeile unter dem Kopf bleibt stehen.
        if not werte["CASE"].strip().isdigit():
            beschriftung.append(zeile)
            continue
        fehlend = fehlende_vergleiche(werte)
        zu_schnell = zu_schnelle_seiten(werte)
        if fehlend:
            entfernt.append((werte["CASE"], "fehlt: " + ", ".join(fehlend)))
        elif zu_schnell:
            entfernt.append((werte["CASE"], f"unter {MIN_SEKUNDEN} s: " + ", ".join(zu_schnell)))
        else:
            behalten.append(zeile)

    out = args.out or args.export.with_name(f"{args.export.stem}_clean.csv")
    with out.open("w", encoding="utf-8-sig", newline="") as datei:
        csv.writer(datei, delimiter=";", lineterminator="\r\n").writerows([kopf, *beschriftung, *behalten])

    print(f"{len(entfernt)} Teilnahmen entfernt:")
    for teilnahme, grund in entfernt:
        print(f"  {teilnahme:>5}  {grund}")
    print(f"{len(behalten)} Teilnahmen behalten -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
