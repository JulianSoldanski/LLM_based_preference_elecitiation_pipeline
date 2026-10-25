"""Oberfläche für Gate 1: interaktive CLI, Ereignislog und Einfrieren.

    python3 src/gate1_cli.py results/assignment/<lauf>.json --run 1 --reviewer JS
    python3 src/gate1_cli.py results/assignment/<lauf>.json --run 1 --auto
    python3 src/gate1_cli.py results/assignment/<lauf>.json --run 1 --reviewer JS --case ketten

Gate 1 läuft je Fall (doc_id) über genau einen Lauf der Zuordnung. Die Regeln
stehen in gate1.py; hier wird nur abgefragt, gestempelt und geschrieben.

Ausgabe je Fall neben der Ergebnisdatei:
    <run_id>_r<run>_<case_id>_gate1.json   Ergebnis, danach schreibgeschützt
    <run_id>_events.jsonl                  eine Zeile je Aktion und eine für
                                           die Fallentscheidung (nur anhängen)
Das GateDecision-Ereignis trägt den SHA-256 der Ergebnisdatei. Schritt 5 liest
ausschließlich über load_validated(), das Hash und Entscheidung prüft. Wer die
Datei später ändert oder Gate 1 für denselben Fall erneut durchführt, bringt
Schritt 5 damit zum Abbruch.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from assignment import NONE, load_criteria, sha256
from experiment import _schreibe_json
from gate1 import (
    CONFIRM,
    CONTINUE,
    DELETE,
    GATE,
    NEED_MORE_INPUT,
    REASSIGN,
    Action,
    Actions,
    Assignment,
    Gate1Result,
    GateError,
    apply_gate1,
    assignments_from_records,
    autonomous,
)
from prompt_builder import CONFIG_DIR

STATUS_PASSED = "passed_gate1"
STATUS_STOPPED = "stopped_at_gate1"
KUERZEL = {"c": CONFIRM, "d": DELETE, "r": REASSIGN}

Eingabe = Callable[[str], str]
Ausgabe = Callable[[str], None]


def gate_pfad(result_path: Path, run: int, case_id: str) -> Path:
    return result_path.with_name(f"{result_path.stem}_r{run}_{case_id}_gate1.json")


def events_pfad(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}_events.jsonl")


def _jetzt() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------- Durchlauf


def run_gate1(
    result_path: Path,
    run: int,
    *,
    reviewer: str | None = None,
    auto: bool = False,
    case: str | None = None,
    eingabe: Eingabe = input,
    ausgabe: Ausgabe = print,
    uhr: Callable[[], str] = _jetzt,
) -> list[Path]:
    """Führt Gate 1 für alle (oder einen) Fälle eines Laufs durch. Gibt die
    geschriebenen Ergebnisdateien zurück; schon abgeschlossene Fälle werden
    übersprungen. Ein Abbruch (Ctrl+C/EOF) schreibt für den Fall nichts."""
    result_path = Path(result_path)
    dokument = json.loads(result_path.read_text(encoding="utf-8"))
    hierarchy = dokument["meta"]["criteria"]
    records = [r for r in dokument["records"] if r["run"] == run]
    if not records:
        raise GateError(f"Lauf {run} ist nicht in {result_path.name}")

    faelle: dict[str, list[Assignment]] = {}
    for a in assignments_from_records(records):
        faelle.setdefault(a.doc_id, []).append(a)
    if case is not None:
        if case not in faelle:
            raise GateError(f"Fall {case!r} ist nicht im Lauf, bekannt: {sorted(faelle)}")
        faelle = {case: faelle[case]}

    namen = _kriteriennamen(dokument["meta"].get("criteria_file"))
    geschrieben = []
    for case_id, assignments in faelle.items():
        ziel = gate_pfad(result_path, run, case_id)
        if ziel.exists():
            ausgabe(f"Fall {case_id}: Gate 1 ist bereits abgeschlossen ({ziel.name}), übersprungen.")
            continue
        geoeffnet = uhr()
        if auto:
            ergebnis, zeiten = autonomous(assignments, hierarchy), {}
        else:
            ergebnis, zeiten = _interaktiv(case_id, run, assignments, hierarchy, namen, reviewer, eingabe, ausgabe, uhr)
        geschlossen = uhr()
        speichern(result_path, dokument["meta"], run, case_id, ergebnis, geoeffnet, geschlossen, zeiten)
        status = STATUS_PASSED if ergebnis.decision == CONTINUE else STATUS_STOPPED
        ausgabe(f"Fall {case_id}: {ergebnis.decision} → {status} · {ziel.name}")
        geschrieben.append(ziel)
    return geschrieben


def speichern(
    result_path: Path,
    meta: dict[str, Any],
    run: int,
    case_id: str,
    ergebnis: Gate1Result,
    geoeffnet: str,
    geschlossen: str,
    zeiten: dict[str, str],
) -> None:
    run_id = result_path.stem
    ziel = gate_pfad(result_path, run, case_id)
    inhalt = {
        "run_id": run_id,
        "run": run,
        "case_id": case_id,
        "gate": GATE,
        "status": STATUS_PASSED if ergebnis.decision == CONTINUE else STATUS_STOPPED,
        "source_file": result_path.name,
        "source_sha256": sha256(result_path),
        "criteria_file": meta.get("criteria_file"),
        "prompt_version": meta.get("prompt_version"),
        "gate_opened": geoeffnet,
        "gate_closed": geschlossen,
        **{k: v for k, v in asdict(ergebnis).items() if k != "events"},
    }
    _schreibe_json(ziel, inhalt)
    ziel.chmod(0o444)  # eingefroren
    pruefsumme = sha256(ziel)

    with events_pfad(result_path).open("a", encoding="utf-8") as log:
        for event in ergebnis.events:
            zeile = {
                "event_id": uuid.uuid4().hex,
                "timestamp": zeiten.get(event["passage_id"], geschlossen),
                "run_id": run_id,
                "case_id": case_id,
                **event,
            }
            if event["event_type"] == "GateDecision":
                zeile.update(result_file=ziel.name, result_sha256=pruefsumme)
            log.write(json.dumps(zeile, ensure_ascii=False) + "\n")


def load_validated(gate_path: Path) -> dict[str, Any]:
    """Einziger Lesezugang für Schritt 5: prüft Hash gegen das Ereignislog und
    gibt das Ergebnis nur bei Entscheidung continue heraus."""
    gate_path = Path(gate_path)
    inhalt = json.loads(gate_path.read_text(encoding="utf-8"))
    log = gate_path.with_name(f"{inhalt['run_id']}_events.jsonl")
    if not log.is_file():
        raise GateError(f"Ereignislog fehlt: {log.name}")
    entscheidungen = [
        e
        for e in map(json.loads, log.read_text(encoding="utf-8").splitlines())
        if e["event_type"] == "GateDecision" and e["gate"] == GATE and e.get("result_file") == gate_path.name
    ]
    if len(entscheidungen) != 1:
        raise GateError(f"{gate_path.name}: {len(entscheidungen)} GateDecision-Einträge im Log, erwartet genau einer")
    if entscheidungen[0]["result_sha256"] != sha256(gate_path):
        raise GateError(f"{gate_path.name} wurde nach Gate 1 verändert (SHA-256 passt nicht zum Log)")
    if inhalt["decision"] != CONTINUE:
        raise GateError(f"Fall {inhalt['case_id']}: {STATUS_STOPPED}, an Schritt 5 wird nichts übergeben")
    return inhalt


def builder_from_gate1(paths: list[Path]) -> tuple[Any, list[dict[str, str]]]:
    """PromptBuilder für Schritt 5 aus validierten Gate-1-Dateien, dazu Name
    und SHA-256 jeder Datei für die Metadaten des Paarvergleichs."""
    from prompt_builder import PromptBuilder

    ergebnisse = [load_validated(p) for p in paths]
    faelle = [e["case_id"] for e in ergebnisse]
    if len(set(faelle)) != len(faelle):
        raise GateError(f"Mehrere Gate-1-Dateien für denselben Fall: {faelle}")
    quellen = [{"file": Path(p).name, "sha256": sha256(Path(p))} for p in paths]
    return PromptBuilder.from_paths().with_gate1(ergebnisse), quellen


# ------------------------------------------------------------- Abfrage


def _interaktiv(
    case_id: str,
    run: int,
    assignments: list[Assignment],
    hierarchy: list[str],
    namen: dict[str, str],
    reviewer: str | None,
    eingabe: Eingabe,
    ausgabe: Ausgabe,
    uhr: Callable[[], str],
) -> tuple[Gate1Result, dict[str, str]]:
    erlaubt = [*hierarchy, NONE]
    while True:
        _anzeigen(case_id, run, assignments, hierarchy, namen, ausgabe)
        actions, zeiten = [], {}
        for a in assignments:
            actions.append(_frage_aktion(a, erlaubt, eingabe))
            zeiten[a.passage_id] = uhr()

        vorschau = apply_gate1(assignments, Actions(tuple(actions), NEED_MORE_INPUT, reviewer), hierarchy)
        ausgabe("\nErgebnis: " + " · ".join(
            f"{code}: {', '.join(vorschau.validated_evidence.get(code, [])) or '–'}" for code in hierarchy))
        if vorschau.criteria_without_evidence:
            ausgabe(f"Kriterien ohne Evidenz: {', '.join(vorschau.criteria_without_evidence)} "
                    "– continue ist nicht möglich.")
            wahl = _frage(eingabe, "Entscheidung: [n] need_more_input  [z] Fall neu beginnen: ", ("n", "z"))
        else:
            wahl = _frage(eingabe, "Entscheidung: [w] continue  [n] need_more_input  [z] Fall neu beginnen: ", ("w", "n", "z"))
        if wahl == "z":
            continue
        grund = eingabe("Begründung zur Entscheidung (optional): ").strip()
        decision = CONTINUE if wahl == "w" else NEED_MORE_INPUT
        return apply_gate1(assignments, Actions(tuple(actions), decision, reviewer, decision_reason=grund), hierarchy), zeiten


def _frage_aktion(a: Assignment, erlaubt: list[str], eingabe: Eingabe) -> Action:
    wahl = KUERZEL[_frage(eingabe, f"\n{a.passage_id} → {a.criterion}   [c] confirm  [d] delete  [r] reassign: ", tuple(KUERZEL))]
    if wahl == CONFIRM:
        return Action(a.passage_id, CONFIRM)
    neu = None
    if wahl == REASSIGN:
        moeglich = tuple(c for c in erlaubt if c != a.criterion)
        neu = _frage(eingabe, f"  Neues Kriterium ({', '.join(moeglich)}): ", moeglich)
    grund = ""
    while not grund:
        grund = eingabe("  Begründung (Pflicht): ").strip()
    return Action(a.passage_id, wahl, neu, grund)


def _frage(eingabe: Eingabe, text: str, erlaubt: tuple[str, ...]) -> str:
    while True:
        antwort = eingabe(text).strip()
        if antwort in erlaubt:
            return antwort


def _anzeigen(
    case_id: str, run: int, assignments: list[Assignment], hierarchy: list[str], namen: dict[str, str], ausgabe: Ausgabe
) -> None:
    ausgabe(f"\n══ Gate 1 · Fall {case_id} · Lauf {run} · {len(assignments)} Passagen ══")
    for code in [*hierarchy, NONE]:
        gruppe = [a for a in assignments if a.criterion == code]
        titel = f"{code} {namen.get(code, '')}".strip()
        if code != NONE and not gruppe:
            ausgabe(f"\n{titel}  — KEINE EVIDENZ")
            continue
        if not gruppe:
            continue
        ausgabe(f"\n{titel}")
        for a in gruppe:
            ausgabe(f"  {a.passage_id}  ({a.doc_id}, Position {a.position})")
            ausgabe(textwrap.fill(" ".join(a.text.split()), width=88, initial_indent="    ", subsequent_indent="    "))
            ausgabe(textwrap.fill(f"Begründung: {a.reason}", width=88, initial_indent="    ", subsequent_indent="    "))


def _kriteriennamen(criteria_file: str | None) -> dict[str, str]:
    pfad = CONFIG_DIR / criteria_file if criteria_file else None
    if not pfad or not pfad.is_file():
        return {}
    return {c.id: c.name for c in load_criteria(pfad)}


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gate1_cli.py", description="Gate 1: Zuordnung durch einen Menschen prüfen.")
    parser.add_argument("result", type=Path, help="Ergebnisdatei aus assignment.py")
    parser.add_argument("--run", type=int, required=True, help="Lauf, der geprüft wird")
    parser.add_argument("--reviewer", help="Kürzel der prüfenden Person (Pflicht ohne --auto)")
    parser.add_argument("--auto", action="store_true", help="Autonomer Modus: alles bestätigen")
    parser.add_argument("--case", help="Nur diesen Fall (doc_id) prüfen")
    args = parser.parse_args(argv)
    if not args.auto and not (args.reviewer or "").strip():
        parser.error("--reviewer ist erforderlich (außer bei --auto)")

    try:
        run_gate1(args.result, args.run, reviewer=args.reviewer, auto=args.auto, case=args.case)
    except GateError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nAbgebrochen, für den offenen Fall wurde nichts gespeichert.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
