"""Einfrieren von Ergebnisdateien mit SHA-256 im Ereignislog.

Gemeinsam für Gate 1, Gate 2 und die fixe Gewichtsversion. Eine Datei gilt
nur als abgeschlossen, wenn sie existiert UND genau ein Abschluss-Ereignis im
Log ihren Hash trägt.

Reihenfolge beim Schreiben (Spec 02):
    1. Inhalt nach <ziel>.tmp schreiben, SHA-256 berechnen.
    2. Ereignisse an das Log anhängen, das Abschluss-Ereignis mit result_file
       und result_sha256; flush und fsync.
    3. <ziel>.tmp in <ziel> umbenennen und schreibschützen.

Ein Abbruch vor Schritt 2 hinterlässt nur die temporäre Datei; der nächste
Versuch überschreibt sie. Ein Abbruch zwischen 2 und 3 lässt sich mit
nachholen() abschließen, weil Hash und Log zusammenpassen. Gelöscht oder
überschrieben wird nie automatisch.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

OFFEN = "open"                 # weder Datei noch Log-Eintrag
ABGESCHLOSSEN = "done"         # Datei und passender Log-Eintrag
NACHHOLBAR = "recoverable"     # Log-Eintrag passt zur temporären Datei, Umbenennen fehlt
UNVOLLSTAENDIG = "incomplete"  # Datei ohne Log-Eintrag oder Log-Eintrag ohne Datei
VERAENDERT = "modified"        # Datei und Log-Eintrag, aber der Hash passt nicht


class LedgerError(ValueError):
    """Ergebnisdatei und Ereignislog passen nicht zusammen."""


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tmp_pfad(ziel: Path) -> Path:
    return ziel.with_name(ziel.name + ".tmp")


def einfrieren(
    ziel: Path, inhalt: dict[str, Any], log: Path, ereignisse: list[dict[str, Any]], abschluss: str
) -> str:
    """Schreibt `inhalt` eingefroren nach `ziel` und hängt `ereignisse` (fertige
    Logzeilen) an `log` an. Das eine Ereignis vom Typ `abschluss` bekommt
    result_file und result_sha256. Gibt den SHA-256 zurück."""
    ziel = Path(ziel)
    if sum(e["event_type"] == abschluss for e in ereignisse) != 1:
        raise LedgerError(f"Genau ein {abschluss}-Ereignis erwartet")
    if zustand(ziel, log, abschluss) != OFFEN:
        raise LedgerError(f"{ziel.name} ist bereits abgeschlossen oder unvollständig ({zustand(ziel, log, abschluss)})")

    tmp = tmp_pfad(ziel)
    tmp.write_text(json.dumps(inhalt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pruefsumme = sha256(tmp)
    _anhaengen(log, [
        {**e, "result_file": ziel.name, "result_sha256": pruefsumme} if e["event_type"] == abschluss else e
        for e in ereignisse
    ])
    _umbenennen(tmp, ziel)
    return pruefsumme


def anhaengen(log: Path, ereignisse: list[dict[str, Any]]) -> None:
    """Ereignisse ohne Ergebnisdatei (z.B. ein Abbruch) ans Log hängen."""
    _anhaengen(log, ereignisse)


def abschluesse(log: Path, ziel_name: str, abschluss: str, gate: int | None = None) -> list[dict[str, Any]]:
    """Alle Abschluss-Ereignisse im Log, die auf `ziel_name` verweisen."""
    if not Path(log).is_file():
        return []
    return [
        e
        for e in map(json.loads, Path(log).read_text(encoding="utf-8").splitlines())
        if e.get("event_type") == abschluss
        and e.get("result_file") == ziel_name
        and (gate is None or e.get("gate") == gate)
    ]


def zustand(ziel: Path, log: Path, abschluss: str, gate: int | None = None) -> str:
    ziel = Path(ziel)
    eintraege = abschluesse(log, ziel.name, abschluss, gate)
    if ziel.exists():
        if not eintraege:
            return UNVOLLSTAENDIG
        return ABGESCHLOSSEN if any(e["result_sha256"] == sha256(ziel) for e in eintraege) else VERAENDERT
    tmp = tmp_pfad(ziel)
    if eintraege:
        if tmp.exists() and any(e["result_sha256"] == sha256(tmp) for e in eintraege):
            return NACHHOLBAR
        return UNVOLLSTAENDIG
    return OFFEN


def beschreibung(ziel: Path, stand: str) -> str:
    """Meldung für Oberflächen; leer bei OFFEN und ABGESCHLOSSEN."""
    name = Path(ziel).name
    return {
        UNVOLLSTAENDIG: (
            f"unvollständig abgeschlossen: Datei {name} ohne Log-Eintrag"
            if Path(ziel).exists() else f"unvollständig abgeschlossen: Log-Eintrag für {name} ohne Datei"
        ),
        VERAENDERT: f"{name} wurde nach dem Abschluss verändert (SHA-256 passt nicht zum Log)",
        NACHHOLBAR: f"{name}: Abschluss steht im Log, die Datei liegt noch als {tmp_pfad(Path(ziel)).name} vor",
    }.get(stand, "")


def nachholen(ziel: Path, log: Path, abschluss: str, gate: int | None = None) -> None:
    """Schließt einen Abbruch zwischen Log und Umbenennen ab."""
    if zustand(ziel, log, abschluss, gate) != NACHHOLBAR:
        raise LedgerError(f"{Path(ziel).name}: nichts nachzuholen")
    _umbenennen(tmp_pfad(Path(ziel)), Path(ziel))


def pruefe(ziel: Path, log: Path, abschluss: str, gate: int | None = None) -> dict[str, Any]:
    """Liest eine eingefrorene Datei nur, wenn genau ein Abschluss-Ereignis im
    Log auf sie verweist und dessen Hash passt."""
    ziel = Path(ziel)
    if not Path(log).is_file():
        raise LedgerError(f"Ereignislog fehlt: {Path(log).name}")
    eintraege = abschluesse(log, ziel.name, abschluss, gate)
    if len(eintraege) != 1:
        raise LedgerError(f"{ziel.name}: {len(eintraege)} {abschluss}-Einträge im Log, erwartet genau einer")
    if eintraege[0]["result_sha256"] != sha256(ziel):
        raise LedgerError(f"{ziel.name} wurde nach dem Abschluss verändert (SHA-256 passt nicht zum Log)")
    return json.loads(ziel.read_text(encoding="utf-8"))


# Einzeln, damit Tests einen Abbruch genau an dieser Stelle erzwingen können.


def _anhaengen(log: Path, ereignisse: list[dict[str, Any]]) -> None:
    with Path(log).open("a", encoding="utf-8") as datei:
        for e in ereignisse:
            datei.write(json.dumps(e, ensure_ascii=False) + "\n")
        datei.flush()
        os.fsync(datei.fileno())


def _umbenennen(tmp: Path, ziel: Path) -> None:
    tmp.replace(ziel)
    ziel.chmod(0o444)  # eingefroren
