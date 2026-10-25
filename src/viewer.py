"""Minimales Frontend: Zuordnung starten und live verfolgen.

    python3 src/viewer.py            # dann http://127.0.0.1:8000 öffnen

Nur Standardbibliothek. "Start" ruft run_assignment() in einem Hintergrund-
Thread auf; die Seite fragt den Zustand zweimal je Sekunde ab und zeigt, welche
Passage gerade zugeordnet wird und in welcher Kriterienspalte sie landet.
Ergebnis- und Logdatei entstehen wie bei der CLI in results/assignment/.

Nach dem Lauf folgt Gate 1 auf derselben Seite, je Fall über genau einen
Lauf. Die Regeln liegen in gate1.apply_gate1(), gespeichert wird mit
gate1_cli.speichern(): dieselben eingefrorenen Dateien und dasselbe
Ereignislog wie bei der CLI.

Danach startet die Seite Schritt 5 (Paarvergleich) mit run_experiment() über
die freigegebenen Fälle. Die Aussagen kommen wie bei experiment.py
ausschließlich über gate1_cli.builder_from_gate1(); Ergebnis und Protokoll
landen wie dort in results/.
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
from assignment import CRITERIA_FILE, PASSAGES_FILE, load_criteria, load_passages, run_assignment
from assignment_eval import evaluate_file
from gate1 import SUPERVISED, Action, Actions, GateError, apply_gate1, assignments_from_records
from gate1_cli import builder_from_gate1, gate_pfad, speichern
from prompt_builder import ROOT, PromptBuilder
from runner import RESULTS_DIR
from schema import CONDITION_HISTORY, CONDITION_STATELESS, Response

PAGE = Path(__file__).with_name("viewer.html")
PROVIDERS = ("openai", "anthropic", "mistral")

# Zustand des laufenden bzw. letzten Durchlaufs. Ein Durchlauf zur Zeit.
_lock = threading.Lock()
_state: dict[str, Any] = {"running": False, "records": []}


def basis() -> dict[str, Any]:
    """Kriterienkatalog, Passagen und zu jeder Passage der Fall, aus dem sie stammt."""
    criteria = load_criteria(CRITERIA_FILE)
    herkunft = {
        text: {"case": case.title, "statement": key}
        for case in PromptBuilder.from_paths().cases
        for key, text in case.statements.items()
    }
    passages = load_passages(PASSAGES_FILE, [c.id for c in criteria])
    return {
        "criteria_file": CRITERIA_FILE.name,
        "criteria": [asdict(c) for c in criteria],
        "passages": [{**p, **herkunft.get(p["text"], {})} for p in passages],
    }


def status() -> dict[str, Any]:
    with _lock:
        schritt5 = _state.get("step5")
        if schritt5:
            schritt5 = {**schritt5, "responses": list(schritt5["responses"])}
        return {**_state, "records": list(_state["records"]), "gate1": _gate_status(), "step5": schritt5}


def starten(params: dict[str, Any]) -> dict[str, Any]:
    provider = params.get("provider")
    model = (params.get("model") or "").strip()
    runs = int(params.get("runs") or 1)
    pause = float(params.get("pause") or 0)
    if provider not in PROVIDERS or not model or runs < 1 or pause < 0:
        raise ValueError("Anbieter, Modell, mindestens 1 Lauf und eine Pause ab 0 s angeben.")
    with _lock:
        if _state["running"] or (_state.get("step5") or {}).get("running"):
            raise ValueError("Es läuft bereits ein Durchlauf.")
        # Fehlender API-Schlüssel fällt hier auf, bevor der Thread startet.
        client = build_client(provider, model)
        _state.clear()
        _state.update(running=True, provider=provider, model=model, runs=runs, pause=pause, run=1, current=None, records=[])
    threading.Thread(target=_durchlauf, args=(client, runs), daemon=True).start()
    return status()


def _fortschritt(run: int, passage: dict[str, Any], record: dict[str, Any] | None) -> None:
    with _lock:
        _state["run"] = run
        if record is None:
            _state["current"] = passage["passage_id"]
        else:
            _state["current"] = None
            _state["records"].append({k: v for k, v in record.items() if k != "calls"})
        pause = _state["pause"] if record else 0
    # Nur für die Anzeige: nach jeder Zuordnung kurz stehen bleiben, damit man
    # mitlesen kann. Außerhalb des Locks, sonst blockiert die Statusabfrage.
    time.sleep(pause)


def _durchlauf(client: LLMClient, runs: int) -> None:
    try:
        json_path, _ = run_assignment(client, runs=runs, quiet=True, progress=_fortschritt)
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
    """Übernimmt einen vorhandenen Lauf, z.B. nach einem Neustart des Servers.
    Ein schon begonnenes Gate 1 ergibt sich aus den Gate-Dateien daneben."""
    pfad = Path(pfad).resolve()
    dokument = json.loads(pfad.read_text(encoding="utf-8"))
    records = [{k: v for k, v in r.items() if k != "calls"} for r in dokument["records"]]
    eval_pfad = pfad.with_name(f"{pfad.stem}_eval.json")
    evaluation = json.loads(eval_pfad.read_text(encoding="utf-8"))["runs"] if eval_pfad.is_file() else []
    gate_laeufe = {int(p.name[len(pfad.stem) + 2:].split("_")[0]) for p in pfad.parent.glob(f"{pfad.stem}_r*_*_gate1.json")}
    with _lock:
        _state.clear()
        _state.update(
            running=False, records=records, runs=dokument["meta"]["runs"], run=max(r["run"] for r in records),
            provider=dokument["meta"]["provider"], model=dokument["meta"]["model_requested"], current=None,
            result=_anzeige(pfad), result_path=str(pfad), evaluation=evaluation,
            gate_run=min(gate_laeufe) if gate_laeufe else None,
        )


# ------------------------------------------------------------------ Gate 1


def _jetzt() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gate_status() -> dict[str, Any] | None:
    """Abgeschlossene Fälle stehen in den Gate-Dateien, nicht im Speicher:
    so überlebt der Stand auch ein Neuladen der Seite."""
    run = _state.get("gate_run")
    if _state["running"] or not _state.get("result_path") or run is None:
        return None
    ergebnis = Path(_state["result_path"])
    faelle = {}
    for case_id in sorted({r["doc_id"] for r in _state["records"] if r["run"] == run}):
        pfad = gate_pfad(ergebnis, run, case_id)
        if pfad.exists():
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
        if gate_pfad(ergebnis_pfad, run, case_id).exists():
            raise GateError(f"Fall {case_id!r}: Gate 1 ist bereits abgeschlossen.")
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
    """Startet den Paarvergleich über alle Fälle, die Gate 1 mit continue
    verlassen haben. Gestoppte Fälle gehen nicht an Schritt 5."""
    provider = params.get("provider")
    model = (params.get("model") or "").strip()
    runs = int(params.get("runs") or 1)
    pause = float(params.get("pause") or 0)
    condition = params.get("condition")
    justification = bool(params.get("justification"))
    if provider not in PROVIDERS or not model or runs < 1 or pause < 0:
        raise ValueError("Anbieter, Modell, mindestens 1 Lauf und eine Pause ab 0 s angeben.")
    if condition not in (CONDITION_HISTORY, CONDITION_STATELESS):
        raise ValueError(f"Bedingung {condition!r} gibt es nicht.")
    with _lock:
        if _state["running"] or (_state.get("step5") or {}).get("running"):
            raise ValueError("Es läuft bereits ein Durchlauf.")
        gate = _gate_status()
        if not gate or not gate["cases"]:
            raise GateError("Erst Gate 1 abschließen.")
        faelle = sorted({r["doc_id"] for r in _state["records"] if r["run"] == gate["run"]})
        if set(gate["cases"]) != set(faelle):
            raise GateError(f"Gate 1 ist noch offen für: {sorted(set(faelle) - set(gate['cases']))}")
        frei = [d for d in faelle if gate["cases"][d]["decision"] == "continue"]
        if not frei:
            raise GateError("Kein Fall hat Gate 1 mit continue verlassen; an Schritt 5 geht nichts.")
        ergebnis = Path(_state["result_path"])
        builder, quellen = builder_from_gate1([gate_pfad(ergebnis, gate["run"], d) for d in frei])
        builder = replace(builder, justification=justification)
        client = build_client(provider, model)
        namen = builder.criteria_by_id()
        _state["step5"] = {
            "running": True, "provider": provider, "model": model, "runs": runs, "pause": pause,
            "condition": condition, "justification": justification,
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
            out_dir=RESULTS_DIR, quiet=True, gate1=quellen, progress=_paarvergleich_fortschritt,
        )
        with _lock:
            _state["step5"]["result"] = _anzeige(json_path)
    except Exception as exc:  # auf der Seite anzeigen statt im Thread verschwinden
        with _lock:
            _state["step5"]["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with _lock:
            _state["step5"].update(running=False, current=None)


def _anzeige(pfad: Path) -> str:
    return str(pfad.relative_to(ROOT) if pfad.is_relative_to(ROOT) else pfad)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        routen = {"/api/basis": basis, "/api/status": status}
        if path not in routen:
            return self._json(404, {"error": f"Unbekannte Route {path}"})
        self._antworte(routen[path])

    def do_POST(self) -> None:
        routen = {
            "/api/start": starten,
            "/api/gate1/open": gate_oeffnen,
            "/api/gate1/close": gate_abschliessen,
            "/api/step5/start": paarvergleich_starten,
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
    parser.add_argument("--load", type=Path, help="Vorhandenen Lauf laden (results/assignment/<lauf>.json)")
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
