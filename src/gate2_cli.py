"""Oberfläche für Gate 2: interaktive CLI, Revision durch den Agenten,
Ereignislog und Einfrieren.

    python3 src/gate2_cli.py results/artifact/<lauf>.json --run 1 --reviewer JS
    python3 src/gate2_cli.py results/artifact/<lauf>.json --run 1 --auto
    python3 src/gate2_cli.py results/artifact/<lauf>.json --run 1 --reviewer JS --case ketten --repair

Gate 2 läuft je Fall über eine gültige Antwort aus Schritt 5 des Artefakts
(Aussagen aus Gate 1). Die Regeln stehen in gate2.py; hier wird angezeigt,
abgefragt, der Agent für Revisionen aufgerufen, gestempelt und geschrieben.

Ausgabe je Fall neben der Ergebnisdatei von Schritt 5:
    <lauf>_r<run>_<case_id>_gate2.json   Ergebnis, danach schreibgeschützt
    <lauf>_events.jsonl                  GateAction, AgentRevision,
                                         WeightsComputed, RepairAttempt und
                                         GateDecision (nur anhängen)
Schritt 8 liest ausschließlich über load_validated(): Hash gegen das Log,
nur accept.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import ahp
import ledger
from agent import AhpAgent, LLMClient, Revision, build_client
from gate1 import GateError
from gate1_cli import builder_from_gate1, load_validated as load_gate1, relativer_pfad
from gate2 import (
    ACCEPT,
    CONFIRM,
    GATE,
    REJECT,
    REVISE,
    SET,
    Gate2Result,
    Gate2Session,
    JudgmentAction,
    autonomous,
    judgments_from_response,
)
from prompt_builder import ROOT, PromptBuilder
from repair import RepairOutcome, run_repair
from runner import RESULTS_DIR
from schema import INDIFFERENCE, MAX_INTENSITY, MIN_INTENSITY, Comparison

GATE1_DIR = RESULTS_DIR / "assignment"
KUERZEL = {"c": CONFIRM, "r": REVISE, "s": SET}

Eingabe = Callable[[str], str]
Ausgabe = Callable[[str], None]
Pair = tuple[str, str]


def gate2_pfad(step5_path: Path, run: int, case_id: str) -> Path:
    return step5_path.with_name(f"{step5_path.stem}_r{run}_{case_id}_gate2.json")


def events_pfad(step5_path: Path) -> Path:
    return step5_path.with_name(f"{step5_path.stem}_events.jsonl")


def abschlussstand(step5_path: Path, run: int, case_id: str) -> str:
    return ledger.zustand(gate2_pfad(step5_path, run, case_id), events_pfad(step5_path), "GateDecision", GATE)


def _jetzt() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------- Fall laden


@dataclass(frozen=True)
class Fall:
    """Alles, was Gate 2 für einen Fall braucht."""

    step5_path: Path
    meta: dict[str, Any]
    run: int
    case_id: str
    response: dict[str, Any]
    judgments: list[Comparison]
    gate1_path: Path
    gate1: dict[str, Any]
    builder: PromptBuilder
    criteria: list[str]

    def passagen(self) -> dict[str, tuple[tuple[str, str], ...]]:
        """Passagen (ID, Text) je Kriterium aus Gate 1."""
        return self.builder.case(self.case_id).passages


def lade_fall(step5_path: Path, run: int, case_id: str, gate1_dir: Path | None = None) -> Fall:
    step5_path = Path(step5_path)
    dokument = json.loads(step5_path.read_text(encoding="utf-8"))
    meta = dokument["meta"]
    # Läufe vor meta.statements_source erkennt man an den Gate-1-Quellen.
    quelle_aussagen = meta.get("statements_source", "gate1" if meta.get("gate1") else "cases.yaml")
    if quelle_aussagen != "gate1" or not meta.get("gate1"):
        raise GateError(f"{step5_path.name}: Gate 2 gibt es nur im Artefakt (Aussagen aus Gate 1)")
    antworten = [r for r in dokument["responses"] if r["run"] == run and r["case_id"] == case_id]
    if len(antworten) != 1:
        raise GateError(f"Fall {case_id!r} in Lauf {run}: {len(antworten)} Antworten, erwartet genau eine")
    judgments = judgments_from_response(antworten[0])  # ungültig: GateError

    quellen = [q for q in meta["gate1"] if q["file"].endswith(f"_{case_id}_gate1.json")]
    if len(quellen) != 1:
        raise GateError(f"Fall {case_id!r}: keine eindeutige Gate-1-Datei in den Metadaten")
    quelle = quellen[0]
    gate1_path = _gate1_pfad(quelle, gate1_dir)
    if ledger.sha256(gate1_path) != quelle["sha256"]:
        raise GateError(f"{gate1_path.name} hat sich seit Schritt 5 verändert (SHA-256)")
    gate1 = load_gate1(gate1_path)
    builder, _ = builder_from_gate1([gate1_path])
    builder = replace(builder, justification=bool(meta.get("justification")), evidence=bool(meta.get("evidence")))
    return Fall(step5_path, meta, run, case_id, antworten[0], judgments, gate1_path, gate1, builder,
                [c.id for c in builder.criteria])


def _gate1_pfad(quelle: dict[str, Any], gate1_dir: Path | None) -> Path:
    kandidaten = []
    if quelle.get("path"):
        pfad = Path(quelle["path"])
        kandidaten.append(pfad if pfad.is_absolute() else ROOT / pfad)
    kandidaten.append(Path(gate1_dir or GATE1_DIR) / quelle["file"])
    for pfad in kandidaten:
        if pfad.is_file():
            return pfad
    raise GateError(f"Gate-1-Datei {quelle['file']} nicht gefunden (gesucht: {[str(k) for k in kandidaten]})")


# ------------------------------------------------------------- Revision


def agent_fuer(fall: Fall, client: LLMClient | None = None) -> AhpAgent:
    """Derselbe Anbieter, dasselbe Modell und dieselbe Temperatur wie in Schritt 5."""
    client = client or build_client(fall.meta["provider"], fall.meta["model"])
    return AhpAgent(builder=fall.builder, client=client, temperature=float(fall.meta.get("temperature") or 0.0))


def revise_mit_wiederholung(
    agent: AhpAgent, case_id: str, current: list[Comparison], pairs: list[Pair], comments: dict[Pair, str] | None
) -> tuple[dict[Pair, Comparison], list[dict[str, Any]]]:
    """Ein Revisionsaufruf, bei ungültiger Antwort eine identische Wiederholung.
    Danach bleibt das alte Urteil (leeres Ergebnis). Gibt die gültigen
    Revisionen und die Aufrufinfos für das Log zurück."""
    aufrufe = []
    for versuch in (1, 2):
        revision = agent.revise(case_id, current, pairs, comments)
        aufrufe.append(_aufruf(agent, revision, versuch))
        if revision.valid:
            return {(c.a, c.b): c for c in revision.comparisons}, aufrufe
    return {}, aufrufe


def _aufruf(agent: AhpAgent, revision: Revision, versuch: int) -> dict[str, Any]:
    return {
        "attempt": versuch,
        "pairs": [list(p) for p in revision.pairs],
        "provider": agent.client.provider,
        "model": agent.client.model_id,
        "prompt_version": revision.prompt_version,
        "prompt_sha256": revision.prompt_sha256,
        "raw": revision.raw,
        "valid": revision.valid,
        "errors": list(revision.errors),
    }


def reparieren(fall: Fall, config: dict[str, Any], agent_holen: Callable[[], AhpAgent]) -> RepairOutcome:
    """Reparaturschleife (Spec 06) für einen Fall, ohne Kommentar an den Agenten."""
    anzahl = {cid: len(p) for cid, p in fall.passagen().items()}
    return run_repair(
        fall.judgments, fall.criteria, config,
        lambda paare, aktuell: revise_mit_wiederholung(agent_holen(), fall.case_id, aktuell, paare, None),
        passages_per_criterion=anzahl,
    )


# ------------------------------------------------------------- Durchlauf


def run_gate2(
    step5_path: Path,
    run: int,
    *,
    reviewer: str | None = None,
    auto: bool = False,
    case: str | None = None,
    repair: bool = False,
    client: LLMClient | None = None,
    gate1_dir: Path | None = None,
    config_path: Path = ahp.AHP_CONFIG,
    eingabe: Eingabe = input,
    ausgabe: Ausgabe = print,
    uhr: Callable[[], str] = _jetzt,
) -> list[Path]:
    """Führt Gate 2 für alle (oder einen) Fälle eines Laufs aus Schritt 5 durch.
    Gibt die geschriebenen Ergebnisdateien zurück. Ungültige Antworten kommen
    nicht in Gate 2; abgeschlossene Fälle werden übersprungen."""
    step5_path = Path(step5_path)
    dokument = json.loads(step5_path.read_text(encoding="utf-8"))
    faelle = list(dict.fromkeys(r["case_id"] for r in dokument["responses"] if r["run"] == run))
    if not faelle:
        raise GateError(f"Lauf {run} ist nicht in {step5_path.name}")
    if case is not None:
        if case not in faelle:
            raise GateError(f"Fall {case!r} ist nicht im Lauf, bekannt: {faelle}")
        faelle = [case]
    config = ahp.load_config(config_path)

    geschrieben = []
    for case_id in faelle:
        ziel = gate2_pfad(step5_path, run, case_id)
        stand = abschlussstand(step5_path, run, case_id)
        if stand == ledger.NACHHOLBAR:
            ledger.nachholen(ziel, events_pfad(step5_path), "GateDecision", GATE)
            ausgabe(f"Fall {case_id}: Abschluss nachgeholt, Log und Datei passen ({ziel.name}).")
            geschrieben.append(ziel)
            continue
        if stand == ledger.ABGESCHLOSSEN:
            ausgabe(f"Fall {case_id}: Gate 2 ist bereits abgeschlossen ({ziel.name}), übersprungen.")
            continue
        if stand != ledger.OFFEN:
            ausgabe(f"Fall {case_id}: {ledger.beschreibung(ziel, stand)}. Übersprungen, bitte von Hand prüfen.")
            continue
        try:
            fall = lade_fall(step5_path, run, case_id, gate1_dir)
        except GateError as exc:
            ausgabe(f"Fall {case_id}: {exc}. Nicht in Gate 2.")
            continue

        agent: list[AhpAgent] = []

        def agent_holen() -> AhpAgent:  # erst beim ersten Bedarf: --auto ruft kein Modell
            if not agent:
                agent.append(agent_fuer(fall, client))
            return agent[0]

        geoeffnet = uhr()
        reparatur = reparieren(fall, config, agent_holen) if repair else None
        start = reparatur.judgments if reparatur else fall.judgments
        if auto:
            ergebnis, zeiten = autonomous(case_id, fall.criteria, start, config, repair=reparatur), {}
        else:
            sitzung = Gate2Session(case_id, fall.criteria, start, config, reviewer=reviewer, repair=reparatur)
            ergebnis, zeiten = _interaktiv(fall, sitzung, agent_holen, eingabe, ausgabe, uhr)
        speichern(fall, ergebnis, geoeffnet, uhr(), zeiten, repair)
        ausgabe(f"Fall {case_id}: {ergebnis.decision} → {ergebnis.status} · {ziel.name}")
        geschrieben.append(ziel)
    return geschrieben


def speichern(
    fall: Fall, ergebnis: Gate2Result, geoeffnet: str, geschlossen: str, zeiten: dict[int, str], repair: bool = False
) -> Path:
    step5_path = fall.step5_path
    run_id = step5_path.stem
    ziel = gate2_pfad(step5_path, fall.run, fall.case_id)
    reparatur = ergebnis.repair or {}
    inhalt = {
        "run_id": run_id,
        "run": fall.run,
        "case_id": fall.case_id,
        "gate": GATE,
        "status": ergebnis.status,
        "source_file": step5_path.name,
        "source_sha256": ledger.sha256(step5_path),
        "gate1_file": fall.gate1_path.name,
        "gate1_path": relativer_pfad(fall.gate1_path, ROOT),
        "gate1_sha256": ledger.sha256(fall.gate1_path),
        "criteria": fall.criteria,
        "provider": fall.meta.get("provider"),
        "model": fall.meta.get("model"),
        "justification": bool(fall.meta.get("justification")),
        "evidence": bool(fall.meta.get("evidence")),
        "initial": ergebnis.initial,
        "cycles": ergebnis.cycles,
        "final": ergebnis.final,
        "counts": ergebnis.counts,
        "hints": ergebnis.hints,
        "decision": ergebnis.decision,
        "reviewer": ergebnis.reviewer,
        "mode": ergebnis.mode,
        "gate_opened": geoeffnet,
        "gate_closed": geschlossen,
        # CR_0 steht immer drin; Reparaturfelder nur mit --repair gefüllt (Spec 06).
        "repair_enabled": repair,
        "cr_0": ergebnis.initial["evaluation"]["cr"],
        "repair_attempts": reparatur.get("repair_attempts", []),
        "cr_after_repair": reparatur.get("cr_after_repair"),
        "repair_warning": reparatur.get("repair_warning", False),
    }
    zeilen = [
        {
            "event_id": uuid.uuid4().hex,
            "timestamp": zeiten.get(event.get("cycle"), geschlossen) if event["event_type"] != "GateDecision" else geschlossen,
            "run_id": run_id,
            "case_id": fall.case_id,
            "run": fall.run,
            **event,
        }
        for event in ergebnis.events
    ]
    try:
        ledger.einfrieren(ziel, inhalt, events_pfad(step5_path), zeilen, "GateDecision")
    except ledger.LedgerError as exc:
        raise GateError(str(exc)) from None
    return ziel


def load_validated(gate_path: Path) -> dict[str, Any]:
    """Einziger Lesezugang für Schritt 8: prüft Hash gegen das Ereignislog und
    gibt das Ergebnis nur bei accept heraus."""
    gate_path = Path(gate_path)
    inhalt = json.loads(gate_path.read_text(encoding="utf-8"))
    log = gate_path.with_name(f"{inhalt['run_id']}_events.jsonl")
    try:
        inhalt = ledger.pruefe(gate_path, log, "GateDecision", GATE)
    except ledger.LedgerError as exc:
        raise GateError(str(exc).replace("nach dem Abschluss", "nach Gate 2")) from None
    if inhalt["decision"] != ACCEPT:
        raise GateError(f"Fall {inhalt['case_id']}: {inhalt['status']}, an Schritt 8 wird nichts übergeben")
    return inhalt


# ------------------------------------------------------------- Abfrage


def _interaktiv(
    fall: Fall,
    sitzung: Gate2Session,
    agent_holen: Callable[[], AhpAgent],
    eingabe: Eingabe,
    ausgabe: Ausgabe,
    uhr: Callable[[], str],
) -> tuple[Gate2Result, dict[int, str]]:
    zeiten: dict[int, str] = {}
    while not sitzung.can_decide:
        _anzeigen(fall, sitzung, ausgabe)
        actions = [_frage_aktion(fall, c, sitzung.revise_allowed, eingabe, ausgabe) for c in sitzung.judgments]
        zeiten[sitzung.cycle] = uhr()
        try:
            zurueck = sitzung.submit(actions)
        except GateError as exc:
            ausgabe(f"Fehler: {exc}. Zyklus bitte neu eingeben.")
            continue
        if zurueck:
            ausgabe(f"Revision durch den Agenten für {len(zurueck)} Urteil(e) …")
            revisionen, aufrufe = revise_mit_wiederholung(
                agent_holen(), fall.case_id, list(sitzung.judgments), [p for p, _ in zurueck], dict(zurueck)
            )
            sitzung.complete(revisionen, aufrufe)
    _anzeigen(fall, sitzung, ausgabe)
    wahl = _frage(eingabe, "Entscheidung: [a] accept  [x] reject: ", ("a", "x"))
    grund = eingabe("Begründung zur Entscheidung (optional): ").strip()
    return sitzung.decide(ACCEPT if wahl == "a" else REJECT, grund), zeiten


def _frage_aktion(fall: Fall, c: Comparison, revise_erlaubt: bool, eingabe: Eingabe, ausgabe: Ausgabe) -> JudgmentAction:
    namen = fall.builder.criteria_by_id()
    kuerzel = {k: v for k, v in KUERZEL.items() if revise_erlaubt or v != REVISE}
    optionen = "  ".join(f"[{k}] {v}" for k, v in kuerzel.items())
    wahl = kuerzel[_frage(eingabe, f"\n{namen[c.a].name} vs. {namen[c.b].name}: {_urteil(fall, c)}   {optionen}: ", tuple(kuerzel))]
    if wahl == CONFIRM:
        return JudgmentAction((c.a, c.b), CONFIRM)
    praeferenz = intensitaet = None
    if wahl == SET:
        richtung = _frage(eingabe, f"  Wichtiger: [1] {namen[c.a].name}  [2] {namen[c.b].name}  [g] {INDIFFERENCE}: ", ("1", "2", "g"))
        praeferenz = {"1": c.a, "2": c.b, "g": INDIFFERENCE}[richtung]
        if praeferenz == INDIFFERENCE:
            intensitaet = 0
        else:
            erlaubt = tuple(str(i) for i in range(MIN_INTENSITY, MAX_INTENSITY + 1))
            intensitaet = int(_frage(eingabe, f"  Intensität ({MIN_INTENSITY}-{MAX_INTENSITY}): ", erlaubt))
    grund = ""
    while not grund:
        grund = eingabe("  Kommentar an den Agenten (Pflicht): " if wahl == REVISE else "  Begründung (Pflicht): ").strip()
    return JudgmentAction((c.a, c.b), wahl, praeferenz, intensitaet, grund)


def _frage(eingabe: Eingabe, text: str, erlaubt: tuple[str, ...]) -> str:
    while True:
        antwort = eingabe(text).strip()
        if antwort in erlaubt:
            return antwort


def _urteil(fall: Fall, c: Comparison | dict[str, Any]) -> str:
    praeferenz, intensitaet = (c["preference"], c["intensity"]) if isinstance(c, dict) else (c.preference, c.intensity)
    if praeferenz == INDIFFERENCE:
        return "gleich wichtig"
    return f"{fall.builder.criteria_by_id()[praeferenz].name} wichtiger, Intensität {intensitaet}"


def _anzeigen(fall: Fall, sitzung: Gate2Session, ausgabe: Ausgabe) -> None:
    """Je Paar zuerst die Evidenz, dann der Vorschlag; unten Gewichte und CR
    (Reihenfolge gegen unkritisches Übernehmen, Parasuraman & Manzey 2010)."""
    namen = fall.builder.criteria_by_id()
    passagen = fall.passagen()
    fuellen = lambda s, einzug="    ": textwrap.fill(" ".join(s.split()), width=88, initial_indent=einzug, subsequent_indent=einzug)  # noqa: E731
    zyklus = min(sitzung.cycle, sitzung.max_cycles) if not sitzung.can_decide else len(sitzung.cycles)
    ausgabe(f"\n══ Gate 2 · Fall {fall.case_id} · Lauf {fall.run} · Zyklus {zyklus} von max. {sitzung.max_cycles} ══")
    for c in sitzung.judgments:
        ausgabe(f"\n{namen[c.a].name} vs. {namen[c.b].name}")
        # Belegstellen: zitierte Passagen nur markieren, nicht doppelt zeigen.
        zitiert = set(c.evidence or ())
        for cid in (c.a, c.b):
            ausgabe(f"  Evidenz {namen[cid].name}:")
            for pid, t in passagen.get(cid, ()):
                ausgabe(fuellen(f"[{pid}] {t}" + ("  · zitiert" if pid in zitiert else "")))
        if c.evidence is not None and not c.evidence:
            ausgabe("  Das Modell zitiert keine Passage (Urteil nur aus Modellwissen).")
        if c.justification:
            ausgabe(fuellen(f"Begründung des Modells: {c.justification}", "  "))
        for h in sitzung.history((c.a, c.b)):
            art = "zurück an Agent" if h["action"] == REVISE else "selbst gesetzt"
            ausgabe(fuellen(f"Zyklus {h['cycle']} · {art}: „{h['reason']}“ · {_urteil(fall, h['before'])} → "
                            f"{_urteil(fall, h['after'])}", "  "))
        ausgabe(f"  → Vorschlag: {_urteil(fall, c)}")
    bewertung = sitzung.evaluation()
    gewichte = " · ".join(f"{namen[k].name} {w:.3f}" for k, w in bewertung["weights"].items())
    ueber = "" if bewertung["consistent"] else "   ⚠ CR ÜBER DER SCHWELLE"
    ausgabe(f"\nGewichte: {gewichte}")
    ausgabe(f"CR {bewertung['cr']:.3f} · Schwelle {bewertung['cr_threshold']:.2f}{ueber}")
    if sitzung.repair is not None:
        r = sitzung.repair
        ausgabe(f"Reparatur vorab: CR_0 {r.cr_0:.3f} → {r.cr_after_repair:.3f} in {len(r.attempts)} Versuch(en)"
                + ("  ⚠ weiterhin über der Schwelle" if r.warning else ""))
    for hinweis in sitzung.hints:
        ausgabe(f"Hinweis: {hinweis}")


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gate2_cli.py", description="Gate 2: Paarvergleiche und Gewichte prüfen.")
    parser.add_argument("result", type=Path, help="Ergebnisdatei aus Schritt 5 des Artefakts (results/artifact/)")
    parser.add_argument("--run", type=int, required=True, help="Lauf aus Schritt 5, der geprüft wird")
    parser.add_argument("--reviewer", help="Kürzel der prüfenden Person (Pflicht ohne --auto)")
    parser.add_argument("--auto", action="store_true", help="Autonomer Modus: alles bestätigen, accept")
    parser.add_argument("--case", help="Nur diesen Fall prüfen")
    parser.add_argument("--repair", action="store_true",
                        help="Vorab automatische CR-Reparatur (Spec 06, nur Demo; in Evaluationsläufen aus)")
    parser.add_argument("--gate1-dir", type=Path, default=GATE1_DIR, help="Ordner der Gate-1-Dateien")
    args = parser.parse_args(argv)
    if not args.auto and not (args.reviewer or "").strip():
        parser.error("--reviewer ist erforderlich (außer bei --auto)")

    try:
        run_gate2(args.result, args.run, reviewer=args.reviewer, auto=args.auto, case=args.case,
                  repair=args.repair, gate1_dir=args.gate1_dir)
    except GateError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nAbgebrochen, für den offenen Fall wurde nichts gespeichert.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
