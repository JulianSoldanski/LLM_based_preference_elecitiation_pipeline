"""Zuordnung von Passagen zu Kriterien (Schritt 3, vor dem Paarvergleich).

Ein LLM ordnet jede Passage einzeln genau einem Kriterium eines geladenen
Katalogs zu oder "none": ein Aufruf je Passage, kein Fallkontext, keine
Gewichtung. Die Antwort ist über das API-Schema an die Codes des Katalogs
gebunden und wird zusätzlich im Code geprüft. Eine ungültige Antwort wird genau einmal mit identischer Anfrage
wiederholt; scheitert auch die, bleibt die Passage als "failed" im Ergebnis.

    python3 src/assignment.py --dry-run
    python3 src/assignment.py --model gpt-4o-2024-08-06 --runs 5

Jeder Datensatz ist eine flache Zeile je Passage und Lauf. "criteria" ist eine
Liste mit höchstens einem Code ([] = "none"), damit Auswertung und ältere
Läufe mit assignment_v1 (mehrere Codes) dasselbe Format haben. doc_id und
position werden aus der Passagendatei durchgereicht, aber nie in den Prompt
gesetzt. Gate 1 (gate1.py) liest die Datensätze, dieser Code nie zurück.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent import LLMClient, build_client
from experiment import ZEITFORMAT_DATEI, _schreibe_json, setup_logger
from prompt_builder import (
    _PLACEHOLDER,
    CONFIG_DIR,
    PROMPTS_DIR,
    ROOT,
    ConfigError,
    _clean,
    _load_yaml,
    _read_text,
)
from runner import RESULTS_DIR, model_slug
from schema import Criterion, strip_code_fence

CRITERIA_FILE = CONFIG_DIR / "criteria_wang2004.yaml"
PROMPT_FILE = PROMPTS_DIR / "assignment_v2.md"
PASSAGES_FILE = ROOT / "data" / "passages_evaluation.csv"
OUT_DIR = RESULTS_DIR / "assignment"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
# Ein Versuch plus genau eine Wiederholung mit identischer Anfrage.
MAX_ATTEMPTS = 2
# expected_criteria in der Passagendatei: leere Zelle = keine Ground Truth,
# "-" = erwartet ist kein Kriterium, sonst Codes durch ";" getrennt.
NO_CRITERION = "-"
# Antwort des Modells, wenn die Passage keinem Kriterium entspricht.
NONE = "none"

_SECTIONS = re.compile(r"\s*SYSTEM:\s*\n(?P<system>.*?)\n\s*USER:\s*\n(?P<user>.*)", re.DOTALL)


# ------------------------------------------------------------------ laden


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_criteria(path: Path) -> list[Criterion]:
    """Katalog aus YAML. code/name/definition landen in Criterion(id, name, description)."""
    criteria = [
        Criterion(id=str(entry["code"]), name=str(entry["name"]), description=_clean(entry["definition"]))
        for entry in _load_yaml(path).get("criteria") or []
    ]
    if not criteria:
        raise ConfigError(f"{path} enthält keine Kriterien.")
    codes = [c.id for c in criteria]
    if len(set(codes)) != len(codes):
        raise ConfigError(f"Doppelte Codes in {path}: {codes}")
    if NONE in codes:
        raise ConfigError(f"{path}: {NONE!r} ist als Code reserviert.")
    return criteria


def load_passages(path: Path, codes: list[str]) -> list[dict[str, Any]]:
    """Passagen aus CSV: passage_id, doc_id, position, source, text, optional
    expected_criteria. position ist die Stelle der Passage im Dokument (ab 1)."""
    if not path.is_file():
        raise ConfigError(f"Passagendatei fehlt: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    passages: list[dict[str, Any]] = []
    problems: list[str] = []
    for zeile, row in enumerate(rows, start=2):  # Zeile 1 ist der Kopf
        passage_id = (row.get("passage_id") or "").strip()
        pflicht = ("passage_id", "doc_id", "position", "source", "text")
        if any(not (row.get(feld) or "").strip() for feld in pflicht):
            problems.append(f"Zeile {zeile}: eines der Felder {', '.join(pflicht)} fehlt")
            continue
        position = row["position"].strip()
        if not position.isdigit() or int(position) < 1:
            problems.append(f"Zeile {zeile}: position {position!r} ist keine Zahl ab 1")
            continue
        expected = _parse_expected(row.get("expected_criteria"))
        if expected and set(expected) - set(codes):
            problems.append(f"Zeile {zeile}: unbekannte erwartete Kriterien {expected}")
        passages.append(
            {
                "passage_id": passage_id,
                "doc_id": row["doc_id"].strip(),
                "position": int(position),
                "source": row["source"].strip(),
                "text": row["text"],
                "expected_criteria": expected,
            }
        )

    ids = [p["passage_id"] for p in passages]
    if len(set(ids)) != len(ids):
        problems.append(f"Doppelte passage_id: {sorted(i for i, n in Counter(ids).items() if n > 1)}")
    stellen = [(p["doc_id"], p["position"]) for p in passages]
    if len(set(stellen)) != len(stellen):
        problems.append(f"Doppelte position im selben Dokument: {sorted(s for s, n in Counter(stellen).items() if n > 1)}")
    if not rows:
        problems.append("Die Datei enthält keine Passagen.")
    if problems:
        raise ConfigError(f"{path}:\n- " + "\n- ".join(problems))
    return passages


def _parse_expected(cell: str | None) -> list[str] | None:
    cell = (cell or "").strip()
    if not cell:
        return None
    if cell == NO_CRITERION:
        return []
    return [code.strip() for code in cell.split(";") if code.strip()]


def load_prompt(path: Path) -> tuple[str, str]:
    """Teilt die versionierte Vorlage in System- und User-Teil."""
    match = _SECTIONS.fullmatch(_read_text(path))
    if not match:
        raise ConfigError(f"{path}: Abschnitte 'SYSTEM:' und 'USER:' fehlen.")
    return match["system"].strip(), match["user"].strip()


def fill(template: str, values: dict[str, str]) -> str:
    """Wie prompt_builder._render, aber ohne Leerzeilen zusammenzuziehen:
    der Passagentext muss unverändert im Prompt stehen."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise ConfigError(f"Unbekannter Platzhalter {{{{{key}}}}} in der Vorlage")
        return values[key]

    return _PLACEHOLDER.sub(replace, template)


def output_schema(codes: list[str]) -> dict[str, Any]:
    """Antwortschema. justification steht vor criterion, damit die Begründung
    vor der Zuordnung erzeugt wird. criterion ist genau ein Code oder "none";
    ein Enum statt einer Liste mit maxItems, weil das alle Anbieter im
    strikten Modus unterstützen."""
    return {
        "type": "object",
        "properties": {
            "passage_id": {"type": "string"},
            "justification": {"type": "string"},
            "criterion": {"type": "string", "enum": [*codes, NONE]},
        },
        "required": ["passage_id", "justification", "criterion"],
        "additionalProperties": False,
    }


def prepare(passages_path: Path, criteria_path: Path, prompt_path: Path) -> dict[str, Any]:
    """Lädt alle Eingaben und rendert die Prompts, bevor ein Modell gerufen wird."""
    criteria = load_criteria(criteria_path)
    codes = [c.id for c in criteria]
    passages = load_passages(passages_path, codes)
    system_template, user_template = load_prompt(prompt_path)
    block = "\n".join(f"- {c.id} ({c.name}): {c.description}" for c in criteria)
    return {
        "codes": codes,
        "passages": passages,
        "system": fill(system_template, {"criteria_block": block}),
        "user": {
            p["passage_id"]: fill(user_template, {"passage_id": p["passage_id"], "passage_text": p["text"]})
            for p in passages
        },
        "schema": output_schema(codes),
    }


# ------------------------------------------------------------ zuordnen


def parse_assignment(raw: str, passage_id: str, codes: list[str]) -> tuple[list[str], str, list[str]]:
    """Prüft eine Rohantwort. Rückgabe (criteria, justification, errors);
    criteria ist [code] oder [] für "none"."""
    text = strip_code_fence(raw or "").strip()
    if not text:
        return [], "", ["Leere Antwort"]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], "", [f"Antwort ist kein gültiges JSON ({exc.msg}, Zeile {exc.lineno})"]
    if not isinstance(payload, dict):
        return [], "", ["Antwort ist kein JSON-Objekt"]

    errors: list[str] = []
    if payload.get("passage_id") != passage_id:
        errors.append(f"passage_id {payload.get('passage_id')!r} statt {passage_id!r}")
    justification = payload.get("justification")
    if not isinstance(justification, str):
        errors.append('Feld "justification" fehlt oder ist kein Text')
    criterion = payload.get("criterion")
    if criterion != NONE and criterion not in codes:
        errors.append(f"Unbekanntes Kriterium {criterion!r}, erlaubt sind {[*codes, NONE]}")

    if errors:
        return [], "", errors
    return ([] if criterion == NONE else [criterion]), justification, []


def assign(
    client: LLMClient,
    system: str,
    user: str,
    passage_id: str,
    codes: list[str],
    schema: dict[str, Any],
    temperature: float,
) -> dict[str, Any]:
    """Ein Aufruf je Passage; bei ungültiger Antwort eine identische Wiederholung."""
    calls: list[dict[str, Any]] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        call: dict[str, Any] = {"attempt": attempt, "timestamp": _jetzt(), "model_returned": None, "raw": ""}
        try:
            completion = client.complete_json(system, user, temperature, "assignment", schema)
        except Exception as exc:  # Netzfehler beenden den Lauf nicht
            call["errors"] = [f"Client-Fehler: {type(exc).__name__}: {exc}"]
        else:
            call["model_returned"] = completion.model
            call["raw"] = completion.text
            criteria, justification, call["errors"] = parse_assignment(completion.text, passage_id, codes)
        calls.append(call)
        if not call["errors"]:
            return {"criteria": criteria, "justification": justification, "status": STATUS_OK, "calls": calls}
    return {"criteria": None, "justification": None, "status": STATUS_FAILED, "calls": calls}


def summarize(records: list[dict[str, Any]], codes: list[str]) -> list[dict[str, Any]]:
    """Je Lauf: ok/failed und welche Kriterien keine Passage bekommen haben."""
    zusammenfassung = []
    for run in sorted({r["run"] for r in records}):
        lauf = [r for r in records if r["run"] == run]
        ok = [r for r in lauf if r["status"] == STATUS_OK]
        je_kriterium = Counter(code for r in ok for code in r["criteria"])
        zusammenfassung.append(
            {
                "run": run,
                "total": len(lauf),
                "ok": len(ok),
                "failed": len(lauf) - len(ok),
                "passages_per_criterion": {code: je_kriterium[code] for code in codes},
                "missing_criteria": [code for code in codes if not je_kriterium[code]],
            }
        )
    return zusammenfassung


# ------------------------------------------------------------ Durchlauf


def run_assignment(
    client: LLMClient,
    *,
    passages_path: Path = PASSAGES_FILE,
    criteria_path: Path = CRITERIA_FILE,
    prompt_path: Path = PROMPT_FILE,
    runs: int = 1,
    temperature: float = 0.0,
    out_dir: Path = OUT_DIR,
    quiet: bool = False,
    progress: Callable[[int, dict[str, Any], dict[str, Any] | None], None] | None = None,
) -> tuple[Path, Path]:
    """n Durchläufe über alle Passagen. Gibt (JSON-Pfad, Log-Pfad) zurück.

    progress(run, passage, record) wird je Passage zweimal gerufen: vor dem
    Modellaufruf mit record=None und danach mit dem fertigen Datensatz."""
    eingabe = prepare(passages_path, criteria_path, prompt_path)
    start = datetime.now(timezone.utc)
    run_id = f"{start.astimezone().strftime(ZEITFORMAT_DATEI)}_{client.provider}_{model_slug(client.model_id)}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, log_path = out_dir / f"{run_id}.json", out_dir / f"{run_id}.log"

    meta = {
        "started": start.isoformat(timespec="seconds"),
        "finished": None,
        "provider": client.provider,
        "model_requested": client.model_id,
        "temperature": temperature,
        "max_tokens": getattr(client, "max_tokens", None),
        "runs": runs,
        "prompt_version": prompt_path.stem,
        "prompt_sha256": sha256(prompt_path),
        "criteria_file": criteria_path.name,
        "criteria_sha256": sha256(criteria_path),
        "criteria": eingabe["codes"],
        "passages_file": passages_path.name,
        "passages_sha256": sha256(passages_path),
        "max_attempts": MAX_ATTEMPTS,
        "output_schema": eingabe["schema"],
    }
    dokument: dict[str, Any] = {
        "meta": meta,
        "prompts": {"system": eingabe["system"], "user": eingabe["user"]},
        "summary": [],
        "records": [],
    }
    # Ground Truth getrennt von den Datensätzen: im Human Gate soll sie nicht sichtbar sein.
    truth = {p["passage_id"]: p["expected_criteria"] for p in eingabe["passages"] if p["expected_criteria"] is not None}
    if truth:
        dokument["ground_truth"] = truth

    logger = setup_logger(log_path, quiet)
    try:
        logger.info(
            "Start %s | provider=%s model=%s temperature=%s runs=%d prompt=%s criteria=%s passages=%d",
            meta["started"], client.provider, client.model_id, temperature, runs,
            meta["prompt_version"], meta["criteria_file"], len(eingabe["passages"]),
        )
        logger.debug("---- SYSTEM-PROMPT ----\n%s", eingabe["system"])
        logger.debug("---- SCHEMA ----\n%s", json.dumps(eingabe["schema"], ensure_ascii=False))
        for passage_id, user in eingabe["user"].items():
            logger.debug("---- USER-PROMPT: %s ----\n%s", passage_id, user)

        for run in range(1, runs + 1):
            for passage in eingabe["passages"]:
                pid = passage["passage_id"]
                if progress:
                    progress(run, passage, None)
                ergebnis = assign(
                    client, eingabe["system"], eingabe["user"][pid], pid,
                    eingabe["codes"], eingabe["schema"], temperature,
                )
                record = {
                    "run_id": run_id,
                    "run": run,
                    "passage_id": pid,
                    "doc_id": passage["doc_id"],
                    "position": passage["position"],
                    "source": passage["source"],
                    "text": passage["text"],
                    "criteria": ergebnis["criteria"],
                    "justification": ergebnis["justification"],
                    "status": ergebnis["status"],
                    "errors": ergebnis["calls"][-1]["errors"],
                    "attempts": len(ergebnis["calls"]),
                    "provider": client.provider,
                    "model_requested": client.model_id,
                    "model_returned": ergebnis["calls"][-1]["model_returned"],
                    "temperature": temperature,
                    "prompt_version": meta["prompt_version"],
                    "criteria_file": meta["criteria_file"],
                    "calls": ergebnis["calls"],
                }
                dokument["records"].append(record)
                dokument["summary"] = summarize(dokument["records"], eingabe["codes"])
                _protokolliere(logger, record, runs)
                # Nach jeder Passage schreiben: ein Abbruch vernichtet nichts.
                _schreibe_json(json_path, dokument)
                if progress:
                    progress(run, passage, record)

        meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _schreibe_json(json_path, dokument)
        for lauf in dokument["summary"]:
            logger.info(
                "Lauf %d: %d ok, %d failed | Kriterien ohne Passage: %s",
                lauf["run"], lauf["ok"], lauf["failed"], ", ".join(lauf["missing_criteria"]) or "keine",
            )
        logger.info("Ergebnisse: %s", json_path)
        logger.info("Protokoll:  %s", log_path)
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    return json_path, log_path


def _protokolliere(logger: logging.Logger, record: dict[str, Any], runs: int) -> None:
    if record["status"] == STATUS_OK:
        ergebnis = ", ".join(record["criteria"]) or "kein Kriterium"
    else:
        ergebnis = "FAILED"
    logger.info("[Lauf %d/%d | %s] %s (%d Versuch(e))", record["run"], runs, record["passage_id"], ergebnis, record["attempts"])

    zeilen = []
    for call in record["calls"]:
        zeilen.append(f"  Versuch {call['attempt']} | Modell {call['model_returned']}")
        zeilen.extend(f"  Fehler: {fehler}" for fehler in call["errors"])
        zeilen.append(f"  Rohantwort: {call['raw']!r}")
    logger.debug("\n".join(zeilen))


def _jetzt() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="assignment.py",
        description="Ordnet Passagen den Kriterien eines Katalogs zu (ein LLM-Aufruf je Passage).",
    )
    parser.add_argument("--provider", choices=["openai", "anthropic", "mistral"], default="openai")
    parser.add_argument("--model", help="Modell-ID, möglichst als Snapshot (außer bei --dry-run Pflicht)")
    parser.add_argument("--runs", type=int, default=1, help="Wiederholungen über alle Passagen")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--passages", type=Path, default=PASSAGES_FILE, help="CSV mit den Passagen")
    parser.add_argument("--criteria", type=Path, default=CRITERIA_FILE, help="YAML mit dem Kriterienkatalog")
    parser.add_argument("--prompt", type=Path, default=PROMPT_FILE, help="Versionierte Prompt-Vorlage")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Zielverzeichnis")
    parser.add_argument("--quiet", action="store_true", help="Keine Konsolenausgabe")
    parser.add_argument("--dry-run", action="store_true", help="Prompts und Schema ausgeben, kein Modellaufruf")

    args = parser.parse_args(argv)
    if not args.dry_run and not args.model:
        parser.error("--model ist erforderlich (außer bei --dry-run)")
    if args.runs < 1:
        parser.error("--runs muss mindestens 1 sein")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        eingabe = prepare(args.passages, args.criteria, args.prompt)
        print(f"==== SYSTEM ====\n{eingabe['system']}\n")
        print(f"==== SCHEMA ====\n{json.dumps(eingabe['schema'], indent=2)}")
        for passage_id, user in eingabe["user"].items():
            print(f"\n==== USER: {passage_id} ====\n{user}")
        return 0

    json_path, _ = run_assignment(
        build_client(args.provider, args.model),
        passages_path=args.passages,
        criteria_path=args.criteria,
        prompt_path=args.prompt,
        runs=args.runs,
        temperature=args.temperature,
        out_dir=args.out_dir,
        quiet=args.quiet,
    )

    # Hier importiert, weil assignment_eval seinerseits dieses Modul importiert.
    from assignment_eval import evaluate_file

    eval_path, tabelle = evaluate_file(json_path)
    if eval_path and not args.quiet:
        print(f"\n{tabelle}\nAuswertung: {eval_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
