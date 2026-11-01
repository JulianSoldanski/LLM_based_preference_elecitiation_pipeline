"""Schritt 8: fixe Gewichtsversion und Replay.

    python3 src/weight_version.py freeze results/artifact/<lauf>_r1_ketten_gate2.json
    python3 src/weight_version.py replay results/artifact/<lauf>_r1_ketten_weights_v1.json

freeze() speichert den validierten Gewichtsvektor aus Gate 2 als
unveränderliche Version mit vollständiger Herkunft (Hash-Kette über Schritt 3,
Gate 1, Schritt 5 und Gate 2). replay() rechnet die Gewichte ohne
Modellaufruf aus den gespeicherten Urteilen neu und prüft die Hash-Kette.
Stimmen beide, ist belegt, dass die Gewichte aus dem Code stammen und nicht
aus dem Modell (DO1, DO4).

Ein neuer Lauf der generativen Schritte erzeugt einen neuen Vorschlag;
reproduzierbar ist die gespeicherte Entscheidung.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ahp
import gate1_cli
import gate2_cli
import ledger
from gate1 import GateError
from prompt_builder import ROOT, PromptBuilder

EVENT = "WeightVersionFrozen"
VERSION = 1
TOLERANZ = 1e-12


class WeightVersionError(GateError):
    """Version fehlt, ist doppelt, verändert oder ihre Herkunft passt nicht."""


@dataclass(frozen=True)
class ReplayResult:
    weights_match: bool
    cr_match: bool
    sources_ok: bool
    differences: list[str] = field(default_factory=list)
    # Kriterium, gespeichertes Gewicht, neu berechnetes Gewicht, Differenz
    rows: list[tuple[str, float, float, float]] = field(default_factory=list)
    cr_stored: float = 0.0
    cr_recomputed: float = 0.0

    @property
    def ok(self) -> bool:
        return self.weights_match and self.cr_match and self.sources_ok


def version_pfad(gate2_path: Path) -> Path:
    gate2_path = Path(gate2_path)
    if not gate2_path.name.endswith("_gate2.json"):
        raise WeightVersionError(f"{gate2_path.name} ist keine Gate-2-Datei")
    return gate2_path.with_name(gate2_path.name.replace("_gate2.json", f"_weights_v{VERSION}.json"))


def _log(pfad: Path, run_id: str) -> Path:
    return Path(pfad).with_name(f"{run_id}_events.jsonl")


def _absolut(pfad: str) -> Path:
    p = Path(pfad)
    return p if p.is_absolute() else ROOT / p


def _datei(pfad: Path) -> dict[str, str]:
    return {"file": pfad.name, "path": gate1_cli.relativer_pfad(pfad, ROOT), "sha256": ledger.sha256(pfad)}


def _git_commit() -> str | None:
    try:
        aus = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return aus.stdout.strip() or None if aus.returncode == 0 else None


def freeze(gate2_path: Path, config_path: Path = ahp.AHP_CONFIG) -> Path:
    """Schreibt <lauf>_r<n>_<fall>_weights_v1.json aus einer Gate-2-Datei mit accept."""
    gate2_path = Path(gate2_path)
    try:
        gate2 = gate2_cli.load_validated(gate2_path)
    except GateError as exc:
        raise WeightVersionError(f"Keine fixe Version: {exc}") from None
    ziel = version_pfad(gate2_path)
    log = _log(gate2_path, gate2["run_id"])
    if ledger.zustand(ziel, log, EVENT) != ledger.OFFEN:
        raise WeightVersionError(f"Für Fall {gate2['case_id']} in Lauf {gate2['run']} gibt es schon eine Version: {ziel.name}")

    # Hash-Kette: Schritt 5 -> Gate 1 -> Schritt 3 muss noch stimmen.
    step5_path = gate2_path.with_name(gate2["source_file"])
    gate1_path = _absolut(gate2["gate1_path"])
    for pfad, erwartet in ((step5_path, gate2["source_sha256"]), (gate1_path, gate2["gate1_sha256"])):
        if not pfad.is_file() or ledger.sha256(pfad) != erwartet:
            raise WeightVersionError(f"Quelle {pfad.name} fehlt oder hat sich seit Gate 2 verändert")
    gate1 = gate1_cli.load_validated(gate1_path)
    assignment_path = gate1_path.with_name(gate1["source_file"])
    if not assignment_path.is_file() or ledger.sha256(assignment_path) != gate1["source_sha256"]:
        raise WeightVersionError(f"Quelle {assignment_path.name} fehlt oder hat sich seit Gate 1 verändert")

    step5 = json.loads(step5_path.read_text(encoding="utf-8"))
    meta, prompts, case_id = step5["meta"], step5["prompts"], gate2["case_id"]
    criteria = gate2["criteria"]
    config = ahp.load_config(config_path)
    ergebnis = ahp.compute(gate2["final"]["judgments"], criteria)
    bewertung = ahp.evaluation(ergebnis, config)
    abweichung = max(abs(ergebnis.weights[c] - gate2["final"]["evaluation"]["weights"][c]) for c in criteria)
    if abweichung > TOLERANZ:
        raise WeightVersionError(f"Gewichte weichen von Gate 2 ab ({abweichung:.2e})")

    codes = PromptBuilder.from_paths().assignment_codes
    revisionen = [
        e for e in map(json.loads, log.read_text(encoding="utf-8").splitlines())
        if e.get("case_id") == case_id and e.get("run") == gate2["run"] and e.get("gate") == 2
        and e.get("event_type") in ("AgentRevision", "RepairAttempt")
    ]
    schritt5_prompt = f"{prompts['system']}\n\n{prompts['user'][case_id]}"
    inhalt = {
        "version_id": uuid.uuid4().hex,
        "version": VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": gate2["run_id"],
        "run": gate2["run"],
        "case_id": case_id,
        "criteria": criteria,
        "judgments": gate2["final"]["judgments"],
        "weights": ergebnis.weights,
        "cr": ergebnis.cr,
        "cr_threshold": bewertung["cr_threshold"],
        "consistent": bewertung["consistent"],
        "method": ergebnis.method,
        "ri": ergebnis.ri,
        "random_index_source": bewertung["random_index_source"],
        "provenance": {
            "files": {
                "assignment": _datei(assignment_path),
                "gate1": _datei(gate1_path),
                "step5": _datei(step5_path),
                "gate2": _datei(gate2_path),
            },
            "provider": meta.get("provider"),
            "model": meta.get("model"),
            "temperature": meta.get("temperature"),
            "seed": meta.get("seed"),
            "condition": meta.get("condition"),
            "justification": meta.get("justification"),
            "evidence": meta.get("evidence"),
            "prompts": {
                "assignment": gate1.get("prompt_version"),
                "step5": "system.txt + user.txt",
                "step5_sha256": hashlib.sha256(schritt5_prompt.encode("utf-8")).hexdigest(),
                "revision": sorted({v for e in revisionen for v in _liste(e.get("prompt_version")) if v}),
                "revision_sha256": sorted({h for e in revisionen for h in _liste(e.get("prompt_sha256")) if h}),
            },
            "passages": {codes.get(code, code): ids for code, ids in gate1["validated_evidence"].items()},
            "gate1": {"reviewer": gate1.get("reviewer"), "mode": gate1.get("mode")},
            "gate2": {"reviewer": gate2.get("reviewer"), "mode": gate2.get("mode"), "cr_0": gate2.get("cr_0"),
                      "repair_enabled": gate2.get("repair_enabled", False)},
        },
        "code_version": _git_commit(),
    }
    ereignis = {
        "event_id": uuid.uuid4().hex,
        "timestamp": inhalt["created"],
        "run_id": gate2["run_id"],
        "case_id": case_id,
        "run": gate2["run"],
        "actor": "code",
        "event_type": EVENT,
        "gate": None,
        "version_id": inhalt["version_id"],
        "weights": ergebnis.weights,
        "cr": ergebnis.cr,
    }
    ledger.einfrieren(ziel, inhalt, log, [ereignis], EVENT)
    return ziel


def _liste(wert: Any) -> list[Any]:
    return wert if isinstance(wert, list) else [wert]


def load_version(path: Path) -> dict[str, Any]:
    """Einziger Lesezugang für spätere Schritte: prüft den Hash gegen das Log."""
    path = Path(path)
    inhalt = json.loads(path.read_text(encoding="utf-8"))
    try:
        return ledger.pruefe(path, _log(path, inhalt["run_id"]), EVENT)
    except ledger.LedgerError as exc:
        raise WeightVersionError(str(exc)) from None


def replay(path: Path) -> ReplayResult:
    """Rechnet Gewichte und CR aus den gespeicherten Urteilen neu und prüft
    die Quellen. Kein Modellaufruf."""
    version = load_version(path)
    ergebnis = ahp.compute(version["judgments"], version["criteria"])
    zeilen = [(c, version["weights"][c], ergebnis.weights[c], ergebnis.weights[c] - version["weights"][c])
              for c in version["criteria"]]
    abweichungen = [f"Gewicht {c}: {d:+.2e}" for c, _, _, d in zeilen if abs(d) > TOLERANZ]
    cr_match = abs(ergebnis.cr - version["cr"]) <= TOLERANZ
    if not cr_match:
        abweichungen.append(f"CR: {ergebnis.cr - version['cr']:+.2e}")
    quellen_ok = True
    for rolle, datei in version["provenance"]["files"].items():
        pfad = _absolut(datei["path"])
        if not pfad.is_file():
            quellen_ok = False
            abweichungen.append(f"Quelle {rolle} fehlt: {datei['file']}")
        elif ledger.sha256(pfad) != datei["sha256"]:
            quellen_ok = False
            abweichungen.append(f"Quelle {rolle} verändert: {datei['file']}")
    return ReplayResult(
        weights_match=all(abs(d) <= TOLERANZ for *_, d in zeilen),
        cr_match=cr_match,
        sources_ok=quellen_ok,
        differences=abweichungen,
        rows=zeilen,
        cr_stored=version["cr"],
        cr_recomputed=ergebnis.cr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="weight_version.py", description="Fixe Gewichtsversion und Replay.")
    befehle = parser.add_subparsers(dest="befehl", required=True)
    befehle.add_parser("freeze", help="Version aus einer Gate-2-Datei mit accept").add_argument("gate2", type=Path)
    befehle.add_parser("replay", help="Gewichte ohne Modellaufruf neu rechnen").add_argument("version", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.befehl == "freeze":
            print(f"-> {freeze(args.gate2)}")
            return 0
        ergebnis = replay(args.version)
    except GateError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1
    print(f"{'Kriterium':<16}{'gespeichert':>14}{'neu':>14}{'Differenz':>12}")
    for c, alt, neu, d in ergebnis.rows:
        print(f"{c:<16}{alt:>14.10f}{neu:>14.10f}{d:>12.1e}")
    print(f"{'CR':<16}{ergebnis.cr_stored:>14.10f}{ergebnis.cr_recomputed:>14.10f}"
          f"{ergebnis.cr_recomputed - ergebnis.cr_stored:>12.1e}")
    for abweichung in ergebnis.differences:
        print(f"  {abweichung}")
    print("OK" if ergebnis.ok else "NICHT OK")
    return 0 if ergebnis.ok else 1


if __name__ == "__main__":
    sys.exit(main())
