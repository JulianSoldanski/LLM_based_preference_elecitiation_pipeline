"""Überführt den bereinigten SoSci-Export in das Format der LLM-Ergebnisse.

Eingabe ist die CSV aus clean_interviews.py, Ausgabe eine JSON-Datei mit
demselben Aufbau wie die Dateien von experiment.py (meta, prompts, summary,
responses). Eine Teilnahme entspricht einem Durchlauf, ihre drei Fälle den
drei Responses dieses Durchlaufs.

    python3 src/transform_interviews.py results/InterviewResults/data_..._clean.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from clean_interviews import VERGLEICHE

GLEICH = "gleich"
ZEITZONE = ZoneInfo("Europe/Berlin")

# Paare (a, b) in der Reihenfolge von config/criteria.yaml und je Antwortcode
# der Richtungsfrage das bevorzugte Kriterium.
PAARE = [
    (("flexibilitaet", "liefertreue"), {"1": "liefertreue", "2": "flexibilitaet", "3": GLEICH}),
    (("kosten", "flexibilitaet"), {"1": "kosten", "2": "flexibilitaet", "3": GLEICH}),
    (("liefertreue", "kosten"), {"1": "liefertreue", "2": "kosten", "3": GLEICH}),
]

# SoSci-Fallkennung -> Fall-ID und Richtungsfragen in der Reihenfolge von PAARE.
FAELLE = {
    "Case1": ("ketten", ["C101", "C102", "C103"]),
    "Case2": ("batteriepacks", ["C201", "C202", "C203"]),
    "Case3": ("konnektivitaetsmodule", ["C301", "C302", "C303"]),
}

# Fallkennung je Seitenposition (randomisierte Fallreihenfolge).
REIHENFOLGE = ["SN01_01", "SN02_01", "SN03_01"]


def utc(zeitpunkt: str) -> str:
    """SoSci-Zeitstempel (Europe/Berlin) im Format der LLM-Dateien (UTC)."""
    lokal = datetime.strptime(zeitpunkt, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZEITZONE)
    return lokal.astimezone(timezone.utc).isoformat(timespec="seconds")


def vergleich(zeile: dict[str, str], richtung: str, paar: tuple[str, str], codes: dict[str, str]):
    """(Vergleich, None) oder (None, Fehler).

    Bei gewählter Richtung zählt nur deren Intensitätsitem; ein Wert im Item der
    Gegenrichtung stammt aus einer nachträglich geänderten Richtung. Bei
    "gleich" ist die Intensität immer 1.
    """
    code = zeile[richtung].strip()
    if code not in codes:
        return None, f"{richtung}: keine gültige Richtung ({code!r})"
    praeferenz = codes[code]
    if praeferenz == GLEICH:
        intensitaet = 1
    else:
        item = VERGLEICHE[richtung][code]
        wert = zeile[item].strip()
        if not wert.isdigit():
            return None, f"{item}: keine Intensität ({wert!r})"
        intensitaet = int(wert)
    return {"a": paar[0], "b": paar[1], "preference": praeferenz, "intensity": intensitaet}, None


def responses(zeile: dict[str, str]) -> list[dict]:
    """Die drei Responses einer Teilnahme, in der Reihenfolge, in der sie die Fälle sah."""
    sequenz = [FAELLE[zeile[spalte].strip()] for spalte in REIHENFOLGE]
    ergebnis = []
    for position, (fall, richtungen) in enumerate(sequenz):
        vergleiche, fehler = [], []
        for richtung, (paar, codes) in zip(richtungen, PAARE):
            eintrag, problem = vergleich(zeile, richtung, paar, codes)
            if problem:
                fehler.append(problem)
            else:
                vergleiche.append(eintrag)

        # Anstelle der Modell-Rohantwort: die SoSci-Werte dieses Falls, auch
        # die nicht ausgewerteten Items der Gegenrichtung.
        items = [i for r in richtungen for i in (r, *(v for v in VERGLEICHE[r].values() if v))]
        roh = {i: zeile[i] for i in items if zeile[i].strip()}

        ergebnis.append({
            "timestamp": utc(zeile["STARTED"]),
            "provider": "sosci",
            "model": "human",
            "temperature": None,
            "condition": "human",
            "seed": None,
            "case_id": fall,
            "run": int(zeile["CASE"]),
            "position": position,
            "sequence": [f for f, _ in sequenz],
            "valid": not fehler,
            "comparisons": vergleiche,
            "errors": fehler,
            "raw": json.dumps(roh, ensure_ascii=False),
        })
    return ergebnis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SoSci-Export -> JSON im Format der LLM-Ergebnisse.")
    parser.add_argument("export", type=Path, help="bereinigte SoSci-CSV (data_*_clean.csv)")
    parser.add_argument("--out", type=Path, help="Zieldatei (Default <export>.json)")
    args = parser.parse_args(argv)

    with args.export.open(encoding="utf-8-sig", newline="") as datei:
        zeilen = [z for z in csv.DictReader(datei, delimiter=";") if z["CASE"].strip().isdigit()]

    antworten = [r for zeile in zeilen for r in responses(zeile)]
    gueltig = sum(1 for r in antworten if r["valid"])
    dokument = {
        "meta": {
            "started": min(utc(z["STARTED"]) for z in zeilen),
            "finished": max(utc(z["LASTDATA"]) for z in zeilen),
            "provider": "sosci",
            "model": "human",
            "temperature": None,
            "condition": "human",
            "seed": None,
            "runs": len(zeilen),
            "cases": [fall for fall, _ in FAELLE.values()],
            "sequences": [antworten[i]["sequence"] for i in range(0, len(antworten), len(FAELLE))],
            "source": args.export.name,
        },
        # Die Probanden haben den Fragebogen gesehen, keinen Prompt.
        "prompts": None,
        "summary": {"total": len(antworten), "valid": gueltig, "invalid": len(antworten) - gueltig},
        "responses": antworten,
    }

    out = args.out or args.export.with_suffix(".json")
    out.write_text(json.dumps(dokument, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{len(zeilen)} Teilnahmen, {len(antworten)} Responses ({gueltig} gültig) -> {out}")
    for r in antworten:
        if not r["valid"]:
            print(f"  run {r['run']} {r['case_id']}: {'; '.join(r['errors'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
