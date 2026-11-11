"""Minimales Frontend: Zuordnung starten und live verfolgen.

    python3 src/viewer.py            # dann http://127.0.0.1:8000 öffnen

Nur Standardbibliothek. Die Pipeline bearbeitet genau ein Produkt (Fall).
Schritt 1 "Aussagen suchen" ruft relevance.run_search() über alle Passagen der
gewählten Datei auf; Ergebnis in results/relevance/. Danach ruft "Zuordnen"
run_assignment() nur für die gefundenen Aussagen auf; Ergebnis wie bei der CLI
in results/assignment/. Beides läuft in einem Hintergrund-Thread; die Seite
fragt den Zustand zweimal je Sekunde ab und zeigt, welche Passage gerade dran
ist und wo sie landet.

Nach dem Lauf folgt Gate 1 auf derselben Seite, über genau einen Lauf. Die Regeln liegen in gate1.apply_gate1(), gespeichert wird mit
gate1_cli.speichern(): dieselben eingefrorenen Dateien und dasselbe
Ereignislog wie bei der CLI.

Danach startet die Seite Schritt 5 (Paarvergleich) mit run_experiment() für
den Fall, sofern Gate 1 ihn freigegeben hat, zustandslos (ein Fall hat keinen
Verlauf). Die Aussagen kommen ausschließlich über
gate1_cli.builder_from_gate1(); Ergebnis und Protokoll landen in
results/artifact/, getrennt von den Experimentläufen.

Danach folgt Gate 2 über einen Lauf aus Schritt 5. Die Regeln liegen
in gate2.Gate2Session, gespeichert wird mit gate2_cli.speichern(): dieselben
eingefrorenen Dateien und dasselbe Ereignislog wie bei der CLI. Revisionen
ruft die Seite mit demselben Modell wie Schritt 5 auf.

Als letzter Schritt zeigt die Seite die Rohdaten der Lieferanten für Teil B
(partb_data.load_partb(), nur lesend). Er steht bewusst hinter Gate 2: Die
Gewichte werden festgelegt, bevor Lieferanten verglichen werden (R11).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent import LLMClient, build_client
from experiment import run_experiment
from assignment import CRITERIA_FILE, PASSAGES_FILE, load_criteria, load_passages, run_assignment, sha256
from assignment_eval import evaluate_file
from relevance import OUT_DIR as SEARCH_DIR, run_search
from gate1 import SUPERVISED, Action, Actions, GateError, apply_gate1, assignments_from_records
from gate2 import Gate2Session, JudgmentAction
import ahp
import partb_data
import gate2_cli
import ledger
from gate1_cli import abschlussstand, builder_from_gate1, events_pfad, gate_pfad, speichern
from prompt_builder import ROOT, PromptBuilder
from runner import RESULTS_DIR
from schema import CONDITION_STATELESS, Response

# Paarvergleiche des Artefakts (Aussagen aus Gate 1) getrennt von den
# Experimentläufen in results/, damit sie nie zusammen ausgewertet werden.
ARTIFACT_SUBDIR = "artifact"

PAGE = Path(__file__).with_name("viewer.html")
# Wählbare Passagendateien: alle data/passages_*.csv, z.B. die Evaluation
# (nur die 9 Aussagen) und hidden (Aussagen zwischen Ablenkern).
PASSAGES_DIR = PASSAGES_FILE.parent
PROVIDERS = ("openai", "anthropic", "mistral")

# Zustand des laufenden bzw. letzten Durchlaufs. Ein Durchlauf zur Zeit.
_lock = threading.Lock()
_state: dict[str, Any] = {"running": False, "records": []}
# Schritt 1 "Aussagen suchen": eigener Zustand, überlebt den Start der Zuordnung.
_search: dict[str, Any] = {}


def passagendateien() -> dict[str, Path]:
    return {p.name: p for p in sorted(PASSAGES_DIR.glob("passages_*.csv"))}


def basis() -> dict[str, Any]:
    """Kriterienkatalog, Passagen je Datei, zu jeder Passage die Aussage des
    Fragebogens, falls sie eine ist, und die wählbaren Fälle (doc_id)."""
    criteria = load_criteria(CRITERIA_FILE)
    cases = PromptBuilder.from_paths().cases
    herkunft = {
        text.strip(): {"case": case.title, "statement": key}
        for case in cases
        for key, text in case.statements.items()
    }
    titel = {case.id: case.title for case in cases}
    dateien = {name: load_passages(pfad, [c.id for c in criteria]) for name, pfad in passagendateien().items()}
    alle = [p for passages in dateien.values() for p in passages]
    return {
        "criteria_file": CRITERIA_FILE.name,
        "criteria": [asdict(c) for c in criteria],
        "passages": {name: [{**p, **herkunft.get(p["text"].strip(), {})} for p in passages]
                     for name, passages in dateien.items()},
        "default_passages": PASSAGES_FILE.name,
        "cases": [{"id": d, "title": titel.get(d, d)} for d in dict.fromkeys(p["doc_id"] for p in alle)],
    }


def status() -> dict[str, Any]:
    with _lock:
        schritt5 = _state.get("step5")
        if schritt5:
            schritt5 = {**schritt5, "responses": list(schritt5["responses"])}
        rest = {k: v for k, v in _state.items() if k != "gate2_sessions"}
        suche = {**_search, "records": list(_search.get("records", []))} if _search else None
        return {**rest, "records": list(_state["records"]), "gate1": _gate_status(), "step5": schritt5,
                "gate2": _gate2_status(), "search": suche}


def _laeuft_etwas() -> bool:
    return bool(_state["running"] or (_state.get("step5") or {}).get("running") or _search.get("running"))


def suchen(params: dict[str, Any]) -> dict[str, Any]:
    """Schritt 1: alle Passagen der Datei für das Produkt des Falls prüfen.
    Eine neue Suche beginnt die Pipeline neu (die alte Zuordnung wird ausgeblendet)."""
    provider = params.get("provider")
    model = (params.get("model") or "").strip()
    pause = float(params.get("pause") or 0)
    case = params.get("case")
    datei = params.get("passages") or PASSAGES_FILE.name
    if provider not in PROVIDERS or not model or pause < 0:
        raise ValueError("Anbieter, Modell und eine Pause ab 0 s angeben.")
    with _lock:
        if _laeuft_etwas():
            raise ValueError("Es läuft bereits ein Durchlauf.")
        dateien = passagendateien()
        if datei not in dateien:
            raise ValueError(f"Passagendatei {datei!r} gibt es nicht, bekannt: {sorted(dateien)}")
        faelle = sorted({p["doc_id"] for p in basis()["passages"][datei]})
        if case not in faelle:
            raise ValueError(f"Bitte ein Produkt wählen, in {datei} bekannt: {faelle}")
        client = build_client(provider, model)
        _search.clear()
        _search.update(running=True, provider=provider, model=model, pause=pause, case=case,
                       passages_file=datei, current=None, records=[])
        _state.clear()
        _state.update(running=False, records=[])
    threading.Thread(target=_suche, args=(client, case, dateien[datei]), daemon=True).start()
    return status()


def _suche_fortschritt(passage: dict[str, Any], record: dict[str, Any] | None) -> None:
    with _lock:
        _search["current"] = None if record else passage["passage_id"]
        if record:
            _search["records"].append(_ohne_aufrufe(record))
        pause = _search["pause"] if record else 0
    time.sleep(pause)


def _suche(client: LLMClient, case: str, passagen: Path) -> None:
    try:
        json_path, _ = run_search(client, case_id=case, passages_path=passagen, quiet=True,
                                  progress=_suche_fortschritt, out_dir=SEARCH_DIR)
        dokument = json.loads(json_path.read_text(encoding="utf-8"))
        with _lock:
            _search.update(result=_anzeige(json_path), result_path=str(json_path),
                           evaluation=dokument.get("evaluation"), summary=dokument["summary"])
    except Exception as exc:
        with _lock:
            _search["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with _lock:
            _search.update(running=False, current=None)


def starten(params: dict[str, Any]) -> dict[str, Any]:
    """Schritt 3: ordnet die in Schritt 1 gefundenen Aussagen zu. Fall und
    Passagendatei kommen aus der Suche."""
    provider = params.get("provider")
    model = (params.get("model") or "").strip()
    runs = int(params.get("runs") or 1)
    pause = float(params.get("pause") or 0)
    if provider not in PROVIDERS or not model or runs < 1 or pause < 0:
        raise ValueError("Anbieter, Modell, mindestens 1 Lauf und eine Pause ab 0 s angeben.")
    with _lock:
        if _laeuft_etwas():
            raise ValueError("Es läuft bereits ein Durchlauf.")
        if not _search.get("result_path"):
            raise ValueError("Erst Aussagen suchen, dann zuordnen.")
        if not _search["summary"]["found"]:
            raise ValueError("Die Suche hat keine Aussagen gefunden, es gibt nichts zuzuordnen.")
        case, datei, suche = _search["case"], _search["passages_file"], Path(_search["result_path"])
        client = build_client(provider, model)
        _state.clear()
        _state.update(running=True, provider=provider, model=model, runs=runs, pause=pause, case=case,
                      passages_file=datei, run=1, current=None, records=[])
    threading.Thread(target=_durchlauf, args=(client, runs, suche), daemon=True).start()
    return status()


def _fortschritt(run: int, passage: dict[str, Any], record: dict[str, Any] | None) -> None:
    with _lock:
        _state["run"] = run
        if record is None:
            _state["current"] = passage["passage_id"]
        else:
            _state["current"] = None
            _state["records"].append(_ohne_aufrufe(record))
        pause = _state["pause"] if record else 0
    # Nur für die Anzeige: nach jeder Zuordnung kurz stehen bleiben, damit man
    # mitlesen kann. Außerhalb des Locks, sonst blockiert die Statusabfrage.
    time.sleep(pause)


def _ohne_aufrufe(record: dict[str, Any]) -> dict[str, Any]:
    """Datensatz ohne Rohantworten (bleiben in der Ergebnisdatei)."""
    return {k: v for k, v in record.items() if k != "calls"}


def _durchlauf(client: LLMClient, runs: int, suche: Path) -> None:
    try:
        json_path, _ = run_assignment(client, search_path=suche, runs=runs, quiet=True, progress=_fortschritt)
        eval_path, _ = evaluate_file(json_path)
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))["runs"] if eval_path else []
        with _lock:
            _state.update(result=_anzeige(json_path), result_path=str(json_path), evaluation=evaluation)
    except Exception as exc:  # auf der Seite anzeigen statt im Thread verschwinden
        with _lock:
            _state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with _lock:
            _state.update(running=False, current=None)


def laden(pfad: Path) -> None:
    """Übernimmt ein vorhandenes Ergebnis, z.B. nach einem Neustart des Servers,
    damit nicht jeder Schritt neu laufen muss. Die Art erkennt die Funktion am
    Inhalt, die vorangehenden Schritte lädt sie über die Verweise in den
    Metadaten mit:

    * Suche (results/relevance/)    → weiter mit "Zuordnen"
    * Zuordnung (results/assignment/) → weiter in Gate 1 bzw. Schritt 5
    * Paarvergleich (results/artifact/) → weiter in Gate 2
    """
    pfad = Path(pfad).resolve()
    dokument = json.loads(pfad.read_text(encoding="utf-8"))
    if dokument["meta"].get("step") == "relevance":
        with _lock:
            _search.clear()
            _search.update(_suche_zustand(pfad, dokument))
            _state.clear()
            _state.update(running=False, records=[])
    elif "responses" in dokument:
        _paarvergleich_laden(pfad, dokument)
    else:
        _zuordnung_laden(pfad, dokument)


def _suche_zustand(pfad: Path, dokument: dict[str, Any]) -> dict[str, Any]:
    meta = dokument["meta"]
    return dict(running=False, provider=meta["provider"], model=meta["model_requested"], case=meta["case_id"],
                passages_file=meta["passages_file"], current=None,
                records=[_ohne_aufrufe(r) for r in dokument["records"]], result=_anzeige(pfad),
                result_path=str(pfad), evaluation=dokument.get("evaluation"), summary=dokument["summary"])


def _zuordnung_laden(pfad: Path, dokument: dict[str, Any]) -> None:
    """Ein schon begonnenes Gate 1 ergibt sich aus den Gate-Dateien daneben.
    Ältere Läufe über alle Fälle bleiben ladbar (case=None)."""
    records = [_ohne_aufrufe(r) for r in dokument["records"]]
    eval_pfad = pfad.with_name(f"{pfad.stem}_eval.json")
    evaluation = json.loads(eval_pfad.read_text(encoding="utf-8"))["runs"] if eval_pfad.is_file() else []
    gate_laeufe = {int(p.name[len(pfad.stem) + 2:].split("_")[0]) for p in pfad.parent.glob(f"{pfad.stem}_r*_*_gate1.json")}
    faelle = {r["doc_id"] for r in records}
    suche = dokument["meta"].get("search_file")
    suche_pfad = SEARCH_DIR / suche if suche else None
    with _lock:
        _search.clear()
        if suche_pfad and suche_pfad.is_file():
            _search.update(_suche_zustand(suche_pfad, json.loads(suche_pfad.read_text(encoding="utf-8"))))
        _state.clear()
        _state.update(
            running=False, records=records, runs=dokument["meta"]["runs"], run=max(r["run"] for r in records),
            case=faelle.pop() if len(faelle) == 1 else None, passages_file=dokument["meta"]["passages_file"],
            provider=dokument["meta"]["provider"], model=dokument["meta"]["model_requested"], current=None,
            result=_anzeige(pfad), result_path=str(pfad), evaluation=evaluation,
            gate_run=min(gate_laeufe) if gate_laeufe else None,
        )


def _paarvergleich_laden(pfad: Path, dokument: dict[str, Any]) -> None:
    """Paarvergleich des Artefakts: über meta.gate1 die Gate-1-Dateien (Hash
    geprüft), über deren source_file den Zuordnungslauf. Ein schon begonnenes
    Gate 2 ergibt sich aus den Gate-2-Dateien daneben."""
    meta = dokument["meta"]
    if meta.get("statements_source") != "gate1" or not meta.get("gate1"):
        raise ValueError(f"{pfad.name} ist ein Experimentlauf (Aussagen aus cases.yaml), "
                         "kein Paarvergleich des Artefakts.")
    gate_pfade = []
    for quelle in meta["gate1"]:
        gate = Path(quelle["path"])
        gate = gate if gate.is_absolute() else ROOT / gate
        if not gate.is_file():
            raise ValueError(f"Gate-1-Datei {quelle['path']} fehlt.")
        if sha256(gate) != quelle["sha256"]:
            raise ValueError(f"Gate-1-Datei {quelle['path']} wurde nach dem Paarvergleich verändert.")
        gate_pfade.append(gate)
    zuordnungen = {gate.parent / json.loads(gate.read_text(encoding="utf-8"))["source_file"] for gate in gate_pfade}
    if len(zuordnungen) != 1:
        raise ValueError(f"Die Gate-1-Dateien stammen aus mehreren Zuordnungsläufen: {sorted(map(str, zuordnungen))}")
    zuordnung = zuordnungen.pop()
    _zuordnung_laden(zuordnung, json.loads(zuordnung.read_text(encoding="utf-8")))

    builder, _ = builder_from_gate1(gate_pfade)
    builder = replace(builder, justification=bool(meta.get("justification")), evidence=bool(meta.get("evidence")))
    namen = builder.criteria_by_id()
    responses = [{k: v for k, v in r.items() if k != "raw"} for r in dokument["responses"]]
    gate2_laeufe = {int(p.name[len(pfad.stem) + 2:].split("_")[0]) for p in pfad.parent.glob(f"{pfad.stem}_r*_*_gate2.json")}
    with _lock:
        _state["step5"] = {
            "running": False, "provider": meta["provider"], "model": meta["model"], "runs": meta["runs"], "pause": 0,
            "condition": meta["condition"], "justification": builder.justification, "evidence": builder.evidence,
            "run": max((r["run"] for r in responses), default=1), "current": None, "responses": responses,
            "cases": [
                {"id": c.id, "title": c.title,
                 "statements": [[namen[cid].name, text] for cid, text in c.ordered_statements()]}
                for c in builder.cases
            ],
            "names": {cid: k.name for cid, k in namen.items()},
            "result": _anzeige(pfad), "result_path": str(pfad),
        }
        if gate2_laeufe:
            _state["gate2_run"] = min(gate2_laeufe)


# ------------------------------------------------------------------ Gate 1


def _jetzt() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gate_status() -> dict[str, Any] | None:
    """Abgeschlossene Fälle stehen in den Gate-Dateien, nicht im Speicher:
    so überlebt der Stand auch ein Neuladen der Seite. Ein Abbruch zwischen
    Log und Umbenennen wird nachgeholt; ein unvollständig abgeschlossener Fall
    erscheint mit "problem" und blockiert Schritt 5."""
    run = _state.get("gate_run")
    if _state["running"] or not _state.get("result_path") or run is None:
        return None
    ergebnis = Path(_state["result_path"])
    faelle = {}
    for case_id in sorted({r["doc_id"] for r in _state["records"] if r["run"] == run}):
        pfad = gate_pfad(ergebnis, run, case_id)
        stand = abschlussstand(ergebnis, run, case_id)
        if stand == ledger.NACHHOLBAR:
            ledger.nachholen(pfad, events_pfad(ergebnis), "GateDecision", 1)
            stand = ledger.ABGESCHLOSSEN
        if stand in (ledger.UNVOLLSTAENDIG, ledger.VERAENDERT):
            faelle[case_id] = {"status": "incomplete", "decision": None, "problem": ledger.beschreibung(pfad, stand),
                               "file": _anzeige(pfad), "passages": {}}
        elif stand == ledger.ABGESCHLOSSEN:
            inhalt = json.loads(pfad.read_text(encoding="utf-8"))
            faelle[case_id] = {k: inhalt[k] for k in ("status", "decision", "counts", "criteria_without_evidence")}
            faelle[case_id]["file"] = _anzeige(pfad)
            faelle[case_id]["passages"] = {
                p["passage_id"]: {"action": p["action"], "criterion": p["criterion"], "reason": p["reason"]}
                for p in inhalt["passages"]
            }
    return {"run": run, "cases": faelle}


def gate_oeffnen(params: dict[str, Any]) -> dict[str, Any]:
    """Merkt sich, wann ein Fall zum ersten Mal geöffnet wurde (Aufwandsmessung).
    Ab dem ersten geöffneten Fall ist Gate 1 an diesen Lauf gebunden."""
    run, case_id = int(params.get("run") or 0), str(params.get("case_id") or "")
    with _lock:
        if _state["running"] or not _state.get("result_path"):
            raise GateError("Gate 1 ist erst nach einem abgeschlossenen Lauf möglich.")
        if _state.get("gate_run") not in (None, run):
            raise GateError(f"Gate 1 läuft bereits auf Lauf {_state['gate_run']}.")
        if not any(r["run"] == run and r["doc_id"] == case_id for r in _state["records"]):
            raise GateError(f"Fall {case_id!r} gibt es in Lauf {run} nicht.")
        _state["gate_run"] = run
        _state.setdefault("gate_opened", {}).setdefault(f"{run}|{case_id}", _jetzt())
    return status()


def gate_abschliessen(params: dict[str, Any]) -> dict[str, Any]:
    run, case_id = int(params.get("run") or 0), str(params.get("case_id") or "")
    with _lock:
        geoeffnet = (_state.get("gate_opened") or {}).get(f"{run}|{case_id}")
        if _state.get("gate_run") != run or geoeffnet is None:
            raise GateError(f"Fall {case_id!r} in Lauf {run} wurde nicht geöffnet.")
        ergebnis_pfad = Path(_state["result_path"])
        stand = abschlussstand(ergebnis_pfad, run, case_id)
        if stand != ledger.OFFEN:
            raise GateError(f"Fall {case_id!r}: Gate 1 ist bereits abgeschlossen."
                            if stand == ledger.ABGESCHLOSSEN else ledger.beschreibung(gate_pfad(ergebnis_pfad, run, case_id), stand))
        # Gelesen wird die Ergebnisdatei, nicht der Anzeigezustand.
        dokument = json.loads(ergebnis_pfad.read_text(encoding="utf-8"))
        records = [r for r in dokument["records"] if r["run"] == run and r["doc_id"] == case_id]
        eingaben = params.get("actions") or []
        actions = Actions(
            per_passage=tuple(
                Action(a["passage_id"], a["action"], a.get("new_criterion"), a.get("reason") or "") for a in eingaben
            ),
            decision=params.get("decision"),
            reviewer=params.get("reviewer"),
            mode=SUPERVISED,
            decision_reason=params.get("decision_reason") or "",
        )
        ergebnis = apply_gate1(assignments_from_records(records), actions, dokument["meta"]["criteria"])
        zeiten = {a["passage_id"]: _utc(a["timestamp"]) for a in eingaben if a.get("timestamp")}
        speichern(ergebnis_pfad, dokument["meta"], run, case_id, ergebnis, geoeffnet, _jetzt(), zeiten)
    return status()


def _utc(zeitpunkt: str) -> str:
    """Zeitstempel aus dem Browser (toISOString) im Format der übrigen Einträge."""
    return datetime.fromisoformat(zeitpunkt.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------- Schritt 5: Paarvergleich


def paarvergleich_starten(params: dict[str, Any]) -> dict[str, Any]:
    """Startet den Paarvergleich für den Fall, wenn er Gate 1 mit continue
    verlassen hat. Ein gestoppter Fall geht nicht an Schritt 5. Zustandslos:
    mit einem Fall gibt es keinen Verlauf, der History-Bedingung bliebe nur
    das Etikett. Immer mit Begründung und Belegstellen (Spec 07): Gate 2 zeigt
    je Urteil, worauf es sich stützt."""
    provider = params.get("provider")
    model = (params.get("model") or "").strip()
    runs = int(params.get("runs") or 1)
    pause = float(params.get("pause") or 0)
    condition = CONDITION_STATELESS
    if provider not in PROVIDERS or not model or runs < 1 or pause < 0:
        raise ValueError("Anbieter, Modell, mindestens 1 Lauf und eine Pause ab 0 s angeben.")
    with _lock:
        if _laeuft_etwas():
            raise ValueError("Es läuft bereits ein Durchlauf.")
        gate = _gate_status()
        if not gate or not gate["cases"]:
            raise GateError("Erst Gate 1 abschließen.")
        faelle = sorted({r["doc_id"] for r in _state["records"] if r["run"] == gate["run"]})
        if set(gate["cases"]) != set(faelle):
            raise GateError(f"Gate 1 ist noch offen für: {sorted(set(faelle) - set(gate['cases']))}")
        kaputt = [d for d in faelle if gate["cases"][d].get("problem")]
        if kaputt:
            raise GateError(f"Gate 1 unvollständig abgeschlossen für {kaputt}; bitte von Hand prüfen.")
        frei = [d for d in faelle if gate["cases"][d]["decision"] == "continue"]
        if not frei:
            raise GateError("Der Fall hat Gate 1 nicht mit continue verlassen; an Schritt 5 geht nichts.")
        ergebnis = Path(_state["result_path"])
        builder, quellen = builder_from_gate1([gate_pfad(ergebnis, gate["run"], d) for d in frei])
        builder = replace(builder, justification=True, evidence=True)
        client = build_client(provider, model)
        namen = builder.criteria_by_id()
        _state["step5"] = {
            "running": True, "provider": provider, "model": model, "runs": runs, "pause": pause,
            "condition": condition, "justification": True, "evidence": True,
            "run": 1, "current": None, "responses": [],
            "cases": [
                {"id": c.id, "title": c.title,
                 "statements": [[namen[cid].name, text] for cid, text in c.ordered_statements()]}
                for c in builder.cases
            ],
            "names": {cid: k.name for cid, k in namen.items()},
        }
    threading.Thread(target=_paarvergleich, args=(builder, client, runs, condition, quellen), daemon=True).start()
    return status()


def _paarvergleich_fortschritt(run: int, case_id: str, response: Response | None) -> None:
    with _lock:
        schritt = _state["step5"]
        schritt["run"] = run
        if response is None:
            schritt["current"] = case_id
        else:
            schritt["current"] = None
            schritt["responses"].append({k: v for k, v in response.to_dict().items() if k != "raw"})
        pause = schritt["pause"] if response else 0
    time.sleep(pause)  # nur für die Anzeige, wie bei der Zuordnung


def _paarvergleich(builder: PromptBuilder, client: LLMClient, runs: int, condition: str, quellen: list[dict[str, str]]) -> None:
    try:
        json_path, _ = run_experiment(
            builder, client, runs=runs, case_ids=builder.case_ids(), condition=condition,
            out_dir=RESULTS_DIR / ARTIFACT_SUBDIR, quiet=True, gate1=quellen, progress=_paarvergleich_fortschritt,
        )
        with _lock:
            _state["step5"].update(result=_anzeige(json_path), result_path=str(json_path))
            _state.pop("gate2_sessions", None)
    except Exception as exc:  # auf der Seite anzeigen statt im Thread verschwinden
        with _lock:
            _state["step5"]["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with _lock:
            _state["step5"].update(running=False, current=None)


# ------------------------------------------------------------------ Gate 2


def _gate2_status() -> dict[str, Any] | None:
    """Stand von Gate 2 je Fall: abgeschlossen (aus der Datei), in Arbeit (aus
    der Sitzung im Speicher) oder offen. Erst nach Schritt 5."""
    schritt5 = _state.get("step5") or {}
    if schritt5.get("running") or not schritt5.get("result_path"):
        return None
    pfad = Path(schritt5["result_path"])
    run = int(_state.get("gate2_run") or 1)
    sitzungen = _state.get("gate2_sessions") or {}
    faelle = {}
    for r in schritt5["responses"]:
        if r["run"] != run:
            continue
        case_id, ziel = r["case_id"], gate2_cli.gate2_pfad(pfad, run, r["case_id"])
        stand = gate2_cli.abschlussstand(pfad, run, case_id)
        if stand == ledger.NACHHOLBAR:
            ledger.nachholen(ziel, gate2_cli.events_pfad(pfad), "GateDecision", 2)
            stand = ledger.ABGESCHLOSSEN
        if stand == ledger.ABGESCHLOSSEN:
            inhalt = json.loads(ziel.read_text(encoding="utf-8"))
            faelle[case_id] = {"state": "done", "file": _anzeige(ziel),
                               **{k: inhalt[k] for k in ("status", "decision", "counts", "final", "cr_0", "repair_warning")},
                               "cycles": len(inhalt["cycles"])}
        elif stand != ledger.OFFEN:
            faelle[case_id] = {"state": "problem", "problem": ledger.beschreibung(ziel, stand), "file": _anzeige(ziel)}
        elif not r["valid"]:
            faelle[case_id] = {"state": "invalid", "errors": r["errors"]}
        elif case_id in sitzungen:
            faelle[case_id] = {"state": "active", **_gate2_ansicht(sitzungen[case_id])}
        else:
            faelle[case_id] = {"state": "open"}
    return {"run": run, "runs": schritt5.get("runs", 1), "cases": faelle}


def _gate2_ansicht(eintrag: dict[str, Any]) -> dict[str, Any]:
    """Je Paar zuerst die Evidenz, dann der Vorschlag; dazu Gewichte und CR."""
    sitzung, fall = eintrag["sitzung"], eintrag["fall"]
    namen = fall.builder.criteria_by_id()
    passagen = fall.passagen()
    return {
        "reviewer": sitzung.reviewer,
        "cycle": sitzung.cycle,
        "max_cycles": sitzung.max_cycles,
        "can_decide": sitzung.can_decide,
        "revise_allowed": sitzung.revise_allowed,
        "hints": sitzung.hints,
        "evaluation": sitzung.evaluation(),
        "initial_cr": sitzung.initial["evaluation"]["cr"],
        "repair": sitzung.repair.to_dict() if sitzung.repair is not None else None,
        "history": [{"cycle": z["cycle"], "actions": [a["action"] for a in z["actions"]], "cr": z["evaluation"]["cr"]}
                    for z in sitzung.cycles],
        "names": {cid: k.name for cid, k in namen.items()},
        "pairs": [
            {
                "pair": [c.a, c.b],
                "evidence": {cid: [list(p) for p in passagen.get(cid, ())] for cid in (c.a, c.b)},
                "cited": list(c.evidence) if c.evidence is not None else None,
                "history": sitzung.history((c.a, c.b)),
                "justification": c.justification,
                "preference": c.preference,
                "intensity": c.intensity,
            }
            for c in sitzung.judgments
        ],
    }


def gate2_oeffnen(params: dict[str, Any]) -> dict[str, Any]:
    """Öffnet Gate 2 für einen Fall. Mit repair läuft vorab die automatische
    CR-Reparatur (Spec 06, nur Demo). Revisionen laufen mit Anbieter, Modell
    und Temperatur aus Schritt 5."""
    case_id, reviewer = str(params.get("case_id") or ""), (params.get("reviewer") or "").strip()
    with _lock:
        stand = _gate2_status()
        if stand is None:
            raise GateError("Gate 2 ist erst nach einem abgeschlossenen Paarvergleich möglich.")
        run = int(params.get("run") or stand["run"])
        if _state.get("gate2_sessions") and _state.get("gate2_run") != run:
            raise GateError(f"Gate 2 läuft bereits auf Lauf {_state['gate2_run']}.")
        _state["gate2_run"] = run
        fall_stand = (_gate2_status()["cases"].get(case_id) or {}).get("state")
        if fall_stand not in ("open", "active"):
            raise GateError(f"Fall {case_id!r} ist in Gate 2 nicht offen.")
        neu = fall_stand == "open"
        if neu:
            if not reviewer:
                raise GateError("Bitte zuerst ein Kürzel als Prüfer eintragen.")
            fall = gate2_cli.lade_fall(Path(_state["step5"]["result_path"]), run, case_id)
    if neu:
        config = ahp.load_config()
        geoeffnet = _jetzt()
        agent = gate2_cli.agent_fuer(fall, build_client(fall.meta["provider"], fall.meta["model"]))
        reparatur = gate2_cli.reparieren(fall, config, lambda: agent) if params.get("repair") else None
        start = reparatur.judgments if reparatur else fall.judgments
        sitzung = Gate2Session(case_id, fall.criteria, start, config, reviewer=reviewer, repair=reparatur)
        with _lock:
            _state.setdefault("gate2_sessions", {})[case_id] = {
                "sitzung": sitzung, "fall": fall, "agent": agent, "geoeffnet": geoeffnet, "zeiten": {},
                "repair": bool(params.get("repair")),
            }
    return status()


def gate2_zyklus(params: dict[str, Any]) -> dict[str, Any]:
    """Ein Zyklus: Aktionen prüfen, set übernehmen, revise an den Agenten,
    neu rechnen."""
    case_id = str(params.get("case_id") or "")
    with _lock:
        eintrag = (_state.get("gate2_sessions") or {}).get(case_id)
        if eintrag is None:
            raise GateError(f"Fall {case_id!r} ist in Gate 2 nicht geöffnet.")
        sitzung = eintrag["sitzung"]
        actions = [
            JudgmentAction(tuple(a["pair"]), a["action"], a.get("preference"),
                           int(a["intensity"]) if a.get("intensity") not in (None, "") else None, a.get("reason") or "")
            for a in params.get("actions") or []
        ]
        eintrag["zeiten"][sitzung.cycle] = _jetzt()
        zurueck = sitzung.submit(actions)
        aktuell = list(sitzung.judgments)
    if zurueck:  # Modellaufruf außerhalb der Sperre: die Statusabfrage läuft weiter
        revisionen, aufrufe = gate2_cli.revise_mit_wiederholung(
            eintrag["agent"], case_id, aktuell, [p for p, _ in zurueck], dict(zurueck))
        with _lock:
            sitzung.complete(revisionen, aufrufe)
    return status()


def gate2_abschliessen(params: dict[str, Any]) -> dict[str, Any]:
    case_id = str(params.get("case_id") or "")
    with _lock:
        eintrag = (_state.get("gate2_sessions") or {}).get(case_id)
        if eintrag is None:
            raise GateError(f"Fall {case_id!r} ist in Gate 2 nicht geöffnet.")
        ergebnis = eintrag["sitzung"].decide(params.get("decision"), params.get("reason") or "")
        gate2_cli.speichern(eintrag["fall"], ergebnis, eintrag["geoeffnet"], _jetzt(), eintrag["zeiten"], eintrag["repair"])
        del _state["gate2_sessions"][case_id]
    return status()


def lieferanten() -> dict[str, Any]:
    """Rohdaten für Teil B, nur lesend."""
    return partb_data.load_partb()


def _anzeige(pfad: Path) -> str:
    return str(pfad.relative_to(ROOT) if pfad.is_relative_to(ROOT) else pfad)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        routen = {"/api/basis": basis, "/api/status": status, "/api/partb": lieferanten}
        if path not in routen:
            return self._json(404, {"error": f"Unbekannte Route {path}"})
        self._antworte(routen[path])

    def do_POST(self) -> None:
        routen = {
            "/api/search/start": suchen,
            "/api/start": starten,
            "/api/gate1/open": gate_oeffnen,
            "/api/gate1/close": gate_abschliessen,
            "/api/step5/start": paarvergleich_starten,
            "/api/gate2/open": gate2_oeffnen,
            "/api/gate2/cycle": gate2_zyklus,
            "/api/gate2/close": gate2_abschliessen,
        }
        funktion = routen.get(urlparse(self.path).path)
        if funktion is None:
            return self._json(404, {"error": "Unbekannte Route"})
        laenge = int(self.headers.get("Content-Length") or 0)
        params = json.loads(self.rfile.read(laenge) or b"{}")
        self._antworte(lambda: funktion(params))

    def _antworte(self, funktion: Any) -> None:
        try:
            self._json(200, funktion())
        except GateError as exc:  # Regelverstoß in Gate 1: Meldung reicht
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # Konfigurations- und Eingabefehler auf der Seite zeigen
            self._json(400, {"error": f"{type(exc).__name__}: {exc}"})

    def _json(self, status_code: int, daten: Any) -> None:
        body = json.dumps(daten, ensure_ascii=False).encode("utf-8")
        self._send(status_code, body, "application/json; charset=utf-8")

    def _send(self, status_code: int, body: bytes, content_type: str) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="viewer.py", description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--load", type=Path,
                        help="Vorhandenes Ergebnis laden und dort weitermachen: results/relevance/<suche>.json "
                             "(weiter mit Zuordnen), results/assignment/<lauf>.json (Gate 1, Schritt 5) "
                             "oder results/artifact/<lauf>.json (Gate 2)")
    args = parser.parse_args(argv)
    if args.load:
        laden(args.load)

    # Nur lokal erreichbar: die Seite startet kostenpflichtige Modellaufrufe.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Viewer läuft auf http://127.0.0.1:{args.port}  (Beenden mit Ctrl+C)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
