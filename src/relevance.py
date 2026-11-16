"""Aussagen suchen (Schritt 1 der Pipeline, vor der Zuordnung).

Für ein gewähltes Produkt prüft ein LLM jede Passage der ganzen Datei einzeln,
über alle Dokumente und Produkte hinweg: Ist sie eine bewertende Aussage zu
diesem Produkt? Der Filter kennt das Produkt (`component` aus cases.yaml), aber
weder den Kriterienkatalog noch doc_id oder position. Ein Aufruf je Passage;
eine ungültige Antwort wird genau einmal identisch wiederholt, scheitert auch
die, ist die Passage "failed".

Weiter an die Zuordnung (assignment.py --search) gehen nur die gefundenen
Passagen (relevant = true). Nicht relevante und gescheiterte Passagen bleiben
nur in der Ergebnisdatei.

    python3 src/relevance.py --dry-run --case batteriepacks
    python3 src/relevance.py --model gpt-4o-2024-08-06 --case batteriepacks --passages data/passages_hidden.csv

Ground Truth (nur zur Auswertung, nie im Prompt): gesucht sind die Passagen mit
doc_id == Fall und mindestens einem erwarteten Kriterium.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent import LLMClient, build_client
from assignment import (
    CRITERIA_FILE,
    MAX_ATTEMPTS,
    PASSAGES_FILE,
    STATUS_FAILED,
    STATUS_OK,
    _mit_wiederholung,
    fill,
    load_criteria,
    load_passages,
    load_prompt,
    sha256,
)
from experiment import ZEITFORMAT_DATEI, _schreibe_json, setup_logger
from prompt_builder import PROMPTS_DIR, ConfigError, PromptBuilder
from runner import RESULTS_DIR, model_slug
from schema import strip_code_fence

PROMPT_FILE = PROMPTS_DIR / "relevance_v2.md"
OUT_DIR = RESULTS_DIR / "relevance"


# ------------------------------------------------------------------ laden


def product_of(case_id: str) -> str:
    """Produkt eines Falls aus cases.yaml (Feld component). Später kommt hier
    eine Produktbeschreibung hinzu."""
    for case in PromptBuilder.from_paths().cases:
        if case.id == case_id:
            return case.component
    raise ConfigError(f"Fall {case_id!r} steht nicht in cases.yaml")


def schema() -> dict[str, Any]:
    """Antwortschema; justification vor der Entscheidung."""
    return {
        "type": "object",
        "properties": {
            "passage_id": {"type": "string"},
            "justification": {"type": "string"},
            "relevant": {"type": "boolean"},
        },
        "required": ["passage_id", "justification", "relevant"],
        "additionalProperties": False,
    }


def prepare(passages_path: Path, case_id: str, prompt_path: Path = PROMPT_FILE) -> dict[str, Any]:
    """Lädt alle Passagen der Datei und rendert die Prompts für das Produkt des Falls."""
    codes = [c.id for c in load_criteria(CRITERIA_FILE)]  # nur zur Prüfung von expected_criteria
    passages = load_passages(passages_path, codes)
    if case_id not in {p["doc_id"] for p in passages}:
        raise ConfigError(f"Fall {case_id!r} hat keine Passagen in {passages_path.name}")
    product = product_of(case_id)
    system_template, user_template = load_prompt(prompt_path)
    return {
        "product": product,
        "passages": passages,
        "system": fill(system_template, {"product": product}),
        "user": {
            p["passage_id"]: fill(user_template, {"passage_id": p["passage_id"], "passage_text": p["text"]})
            for p in passages
        },
        "schema": schema(),
    }


# ------------------------------------------------------------------ prüfen


def parse_relevance(raw: str, passage_id: str) -> tuple[bool | None, str, list[str]]:
    """Prüft eine Rohantwort. Rückgabe (relevant, justification, errors)."""
    text = strip_code_fence(raw or "").strip()
    if not text:
        return None, "", ["Leere Antwort"]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, "", [f"Antwort ist kein gültiges JSON ({exc.msg}, Zeile {exc.lineno})"]
    if not isinstance(payload, dict):
        return None, "", ["Antwort ist kein JSON-Objekt"]

    errors: list[str] = []
    if payload.get("passage_id") != passage_id:
        errors.append(f"passage_id {payload.get('passage_id')!r} statt {passage_id!r}")
    justification = payload.get("justification")
    if not isinstance(justification, str):
        errors.append('Feld "justification" fehlt oder ist kein Text')
    relevant = payload.get("relevant")
    if not isinstance(relevant, bool):
        errors.append(f'Feld "relevant" ist {relevant!r}, erwartet true oder false')

    if errors:
        return None, "", errors
    return relevant, justification, []


def screen(client: LLMClient, system: str, user: str, passage_id: str, temperature: float) -> dict[str, Any]:
    """Ein Aufruf je Passage; bei ungültiger Antwort eine identische Wiederholung."""
    relevant, justification, calls = _mit_wiederholung(
        client, system, user, "relevance", schema(), temperature,
        lambda raw: parse_relevance(raw, passage_id),
    )
    if justification is None:
        return {"relevant": None, "justification": None, "status": STATUS_FAILED, "calls": calls}
    return {"relevant": relevant, "justification": justification, "status": STATUS_OK, "calls": calls}


def ground_truth(passages: list[dict[str, Any]], case_id: str) -> dict[str, bool]:
    """Gesucht: Passagen dieses Falls mit mindestens einem erwarteten Kriterium.
    Passagen ohne expected_criteria haben keine Ground Truth."""
    return {
        p["passage_id"]: p["doc_id"] == case_id and bool(p["expected_criteria"])
        for p in passages
        if p["expected_criteria"] is not None
    }


def evaluate(dokument: dict[str, Any]) -> dict[str, Any] | None:
    """Precision/Recall der Suche. failed zählt als nicht gefunden."""
    truth = dokument.get("ground_truth") or {}
    bewertet = [r for r in dokument["records"] if r["passage_id"] in truth]
    if not bewertet:
        return None
    z = Counter()
    for r in bewertet:
        gefunden, gesucht = r["relevant"] is True, truth[r["passage_id"]]
        z[("t" if gefunden == gesucht else "f") + ("p" if gefunden else "n")] += 1
    quote = lambda a, b: a / b if b else None  # noqa: E731
    return {
        "passages": len(bewertet),
        "precision": quote(z["tp"], z["tp"] + z["fp"]),
        "recall": quote(z["tp"], z["tp"] + z["fn"]),
        "tp": z["tp"], "fp": z["fp"], "fn": z["fn"], "tn": z["tn"],
        "missed": sorted(r["passage_id"] for r in bewertet if truth[r["passage_id"]] and r["relevant"] is not True),
    }


def load_search(path: Path) -> dict[str, Any]:
    """Liest das Ergebnis einer Suche für die Zuordnung. Bricht ab, wenn die Suche
    nicht fertig ist."""
    dokument = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = dokument.get("meta") or {}
    if meta.get("step") != "relevance" or not meta.get("finished"):
        raise ConfigError(f"{path}: kein abgeschlossenes Ergebnis von 'Aussagen suchen'")
    return dokument


def found_ids(dokument: dict[str, Any]) -> list[str]:
    return [r["passage_id"] for r in dokument["records"] if r["relevant"] is True]


# ------------------------------------------------------------ Durchlauf


def run_search(
    client: LLMClient,
    *,
    case_id: str,
    passages_path: Path = PASSAGES_FILE,
    prompt_path: Path = PROMPT_FILE,
    temperature: float = 0.0,
    out_dir: Path = OUT_DIR,
    quiet: bool = False,
    progress: Callable[[dict[str, Any], dict[str, Any] | None], None] | None = None,
) -> tuple[Path, Path]:
    """Prüft alle Passagen der Datei für das Produkt des Falls. Gibt (JSON, Log) zurück.

    progress(passage, record) wird je Passage zweimal gerufen: vor dem Aufruf
    mit record=None und danach mit dem fertigen Datensatz."""
    eingabe = prepare(passages_path, case_id, prompt_path)
    start = datetime.now(timezone.utc)
    run_id = f"{start.astimezone().strftime(ZEITFORMAT_DATEI)}_{client.provider}_{model_slug(client.model_id)}_{case_id}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, log_path = out_dir / f"{run_id}.json", out_dir / f"{run_id}.log"

    meta = {
        "step": "relevance",
        "started": start.isoformat(timespec="seconds"),
        "finished": None,
        "provider": client.provider,
        "model_requested": client.model_id,
        "temperature": temperature,
        "case_id": case_id,
        "product": eingabe["product"],
        "prompt_version": prompt_path.stem,
        "prompt_sha256": sha256(prompt_path),
        "passages_file": passages_path.name,
        "passages_sha256": sha256(passages_path),
        "max_attempts": MAX_ATTEMPTS,
        "output_schema": eingabe["schema"],
    }
    dokument: dict[str, Any] = {
        "meta": meta,
        "prompts": {"system": eingabe["system"], "user": eingabe["user"]},
        "summary": {},
        "records": [],
    }
    truth = ground_truth(eingabe["passages"], case_id)
    if truth:
        dokument["ground_truth"] = truth

    logger = setup_logger(log_path, quiet)
    try:
        logger.info("Aussagen suchen %s | provider=%s model=%s prompt=%s produkt=%s passagen=%d (%s)",
                    meta["started"], client.provider, client.model_id, meta["prompt_version"],
                    eingabe["product"], len(eingabe["passages"]), passages_path.name)
        logger.debug("---- SYSTEM-PROMPT ----\n%s", eingabe["system"])
        for passage in eingabe["passages"]:
            pid = passage["passage_id"]
            if progress:
                progress(passage, None)
            ergebnis = screen(client, eingabe["system"], eingabe["user"][pid], pid, temperature)
            record = {
                "passage_id": pid,
                "doc_id": passage["doc_id"],
                "position": passage["position"],
                "source": passage["source"],
                "text": passage["text"],
                "relevant": ergebnis["relevant"],
                "justification": ergebnis["justification"],
                "status": ergebnis["status"],
                "errors": ergebnis["calls"][-1]["errors"],
                "attempts": len(ergebnis["calls"]),
                "model_returned": ergebnis["calls"][-1]["model_returned"],
                "calls": ergebnis["calls"],
            }
            dokument["records"].append(record)
            dokument["summary"] = _zusammenfassung(dokument["records"])
            _protokolliere(logger, record)
            _schreibe_json(json_path, dokument)  # nach jeder Passage: ein Abbruch vernichtet nichts
            if progress:
                progress(passage, record)

        meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        dokument["evaluation"] = evaluate(dokument)
        _schreibe_json(json_path, dokument)
        s = dokument["summary"]
        logger.info("Gefunden: %d von %d, nicht relevant: %d, failed: %d", s["found"], s["total"],
                    s["not_relevant"], s["failed"])
        if dokument["evaluation"]:
            e = dokument["evaluation"]
            logger.info("Gegen Ground Truth: Precision %s, Recall %s, übersehen: %s",
                        _zahl(e["precision"]), _zahl(e["recall"]), ", ".join(e["missed"]) or "keine")
        logger.info("Ergebnisse: %s", json_path)
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
    return json_path, log_path


def _zusammenfassung(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(records),
        "found": sum(r["relevant"] is True for r in records),
        "not_relevant": sum(r["relevant"] is False for r in records),
        "failed": sum(r["status"] == STATUS_FAILED for r in records),
    }


def _protokolliere(logger: logging.Logger, record: dict[str, Any]) -> None:
    ergebnis = {True: "relevant", False: "nicht relevant", None: "FAILED"}[record["relevant"]]
    logger.info("[%s] %s (%d Versuch(e))", record["passage_id"], ergebnis, record["attempts"])
    zeilen = []
    for call in record["calls"]:
        zeilen.append(f"  Versuch {call['attempt']} | Modell {call['model_returned']}")
        zeilen.extend(f"  Fehler: {fehler}" for fehler in call["errors"])
        zeilen.append(f"  Rohantwort: {call['raw']!r}")
    logger.debug("\n".join(zeilen))


def _zahl(wert: float | None) -> str:
    return "-" if wert is None else f"{wert:.2f}"


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="relevance.py", description="Sucht die bewertenden Aussagen zu einem Produkt.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "mistral"], default="openai")
    parser.add_argument("--model", help="Modell-ID (außer bei --dry-run Pflicht)")
    parser.add_argument("--case", required=True, help="Fall (doc_id); das Produkt kommt aus cases.yaml")
    parser.add_argument("--passages", type=Path, default=PASSAGES_FILE, help="CSV mit den Passagen")
    parser.add_argument("--prompt", type=Path, default=PROMPT_FILE, help="Versionierte Prompt-Vorlage")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Prompts ausgeben, kein Modellaufruf")
    args = parser.parse_args(argv)
    if not args.dry_run and not args.model:
        parser.error("--model ist erforderlich (außer bei --dry-run)")

    if args.dry_run:
        eingabe = prepare(args.passages, args.case, args.prompt)
        print(f"==== SYSTEM ====\n{eingabe['system']}\n")
        print(f"==== SCHEMA ====\n{json.dumps(eingabe['schema'], indent=2)}")
        for passage_id, user in eingabe["user"].items():
            print(f"\n==== USER: {passage_id} ====\n{user}")
        return 0

    json_path, _ = run_search(build_client(args.provider, args.model), case_id=args.case,
                              passages_path=args.passages, prompt_path=args.prompt,
                              temperature=args.temperature, out_dir=args.out_dir, quiet=args.quiet)
    if not args.quiet:
        print(f"\nWeiter mit: python3 src/assignment.py --model {args.model} --search {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
