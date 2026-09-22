"""Dokumentierter Durchlauf: n Wiederholungen über alle Fälle.

runner.py sammelt Rohurteile als JSONL. Dieses Skript ergänzt die
Dokumentation, die für die Arbeit nachvollziehbar sein muss:

* eine Logdatei mit den verwendeten Prompts und jeder einzelnen Durchführung,
* eine JSON-Datei mit Metadaten, Prompts, Zusammenfassung und allen Antworten.

Beide Dateien tragen denselben Zeitstempel im Namen, sodass Protokoll und
Ergebnisse eines Laufs zusammenbleiben. Ausgewertet wird auch hier nichts:
gespeichert werden ausschließlich Rohurteile.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent import AhpAgent, LLMClient, build_client
from prompt_builder import PromptBuilder
from runner import (
    RESULTS_DIR,
    case_sequences,
    model_slug,
    resolve_cases,
    resolve_seed,
    warne_ohne_unabhaengigkeitshinweis,
)
from schema import CONDITION_HISTORY, CONDITION_STATELESS, Response

ZEITFORMAT_DATEI = "%Y%m%d-%H%M%S"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="experiment.py",
        description="Führt n dokumentierte Durchläufe über alle Fälle aus.",
    )
    parser.add_argument("--provider", choices=["openai", "anthropic", "mistral"], default="openai")
    parser.add_argument("--model", required=True, help="Modell-ID des Anbieters")
    parser.add_argument(
        "--condition",
        choices=[CONDITION_HISTORY, CONDITION_STATELESS],
        default=CONDITION_HISTORY,
        help="history: alle Fälle in einer Konversation. stateless: ein Fall pro Aufruf.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed für die Fallreihenfolge. Nur in der History-Bedingung wirksam.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Wiederholungen je Fall")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Fall-ID; mehrfach angebbar. Ohne Angabe werden alle Fälle erhoben.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=RESULTS_DIR, help="Zielverzeichnis (Default results/)"
    )
    parser.add_argument("--quiet", action="store_true", help="Keine Konsolenausgabe")

    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs muss mindestens 1 sein")
    return args


def setup_logger(log_path: Path, quiet: bool = False) -> logging.Logger:
    """Datei bekommt alles, Konsole nur den Fortschritt.

    Eigener Logger je Lauf statt des Root-Loggers, damit mehrere Durchläufe im
    selben Prozess (z.B. in Tests) sich nicht gegenseitig protokollieren.
    """
    logger = logging.getLogger(f"experiment.{log_path.stem}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    datei = logging.FileHandler(log_path, encoding="utf-8")
    datei.setLevel(logging.DEBUG)
    datei.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(datei)

    if not quiet:
        konsole = logging.StreamHandler(sys.stdout)
        konsole.setLevel(logging.INFO)
        konsole.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(konsole)

    return logger


def run_experiment(
    builder: PromptBuilder,
    client: LLMClient,
    *,
    runs: int,
    case_ids: list[str],
    temperature: float = 0.0,
    condition: str = CONDITION_HISTORY,
    seed: int | None = None,
    out_dir: Path = RESULTS_DIR,
    quiet: bool = False,
) -> tuple[Path, Path]:
    """Führt den Durchlauf aus und gibt (JSON-Pfad, Log-Pfad) zurück."""
    start = datetime.now(timezone.utc)
    seed = resolve_seed(condition, seed)
    folgen = case_sequences(case_ids, runs, condition, seed)
    # Dateiname in lokaler Zeit (leichter wiederzufinden), Metadaten in UTC.
    stamm = (
        f"{start.astimezone().strftime(ZEITFORMAT_DATEI)}_"
        f"{client.provider}_{model_slug(client.model_id)}_{condition}"
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{stamm}.log"
    json_path = out_dir / f"{stamm}.json"

    agent = AhpAgent(builder=builder, client=client, temperature=temperature)
    prompts = {
        "system": builder.build_system(),
        "user": {case_id: builder.build_user(case_id) for case_id in case_ids},
    }
    # Alles, was unverändert in jede Zwischenfassung der JSON gehört.
    kontext = {
        "provider": client.provider,
        "model": client.model_id,
        "temperature": temperature,
        "condition": condition,
        "seed": seed,
        "runs": runs,
        "cases": list(case_ids),
        "sequences": folgen,
        "prompts": prompts,
    }
    antworten: list[Response] = []
    logger = setup_logger(log_path, quiet)

    try:
        logger.info(
            "Start %s | provider=%s model=%s temperature=%s condition=%s seed=%s "
            "runs=%d cases=%s",
            start.isoformat(timespec="seconds"),
            client.provider,
            client.model_id,
            temperature,
            condition,
            seed,
            runs,
            ", ".join(case_ids),
        )
        logger.debug("---- SYSTEM-PROMPT ----\n%s", prompts["system"])
        for case_id in case_ids:
            logger.debug("---- USER-PROMPT: %s ----\n%s", case_id, prompts["user"][case_id])

        # Wiederholungen außen, Fälle innen: ein Abbruch hinterlässt
        # vollständige Durchgänge über alle Fälle.
        for run, folge in enumerate(folgen, start=1):
            if condition == CONDITION_HISTORY:
                logger.info("Lauf %d/%d Reihenfolge: %s", run, runs, ", ".join(folge))
                lauf = agent.judge_sequence(folge, run=run, seed=seed)
            else:
                lauf = [agent.judge(case_id, run=run) for case_id in folge]

            for response in lauf:
                antworten.append(response)
                _protokolliere(logger, response, run, runs)
                # Nach jeder Antwort schreiben: ein Abbruch vernichtet nichts.
                _schreibe_json(
                    json_path,
                    _dokument(start, None, kontext, antworten),
                )

        ende = datetime.now(timezone.utc)
        _schreibe_json(json_path, _dokument(start, ende, kontext, antworten))
        zusammenfassung = _zusammenfassung(antworten)
        logger.info(
            "Ende %s | %d Antworten, davon %d gültig, %d ungültig | Dauer %.1fs",
            ende.isoformat(timespec="seconds"),
            zusammenfassung["total"],
            zusammenfassung["valid"],
            zusammenfassung["invalid"],
            (ende - start).total_seconds(),
        )
        logger.info("Ergebnisse: %s", json_path)
        logger.info("Protokoll:  %s", log_path)
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    return json_path, log_path


def _protokolliere(logger: logging.Logger, response: Response, run: int, runs: int) -> None:
    kopf = f"[Lauf {run}/{runs} | {response.case_id}] " + (
        "gültig" if response.valid else "UNGÜLTIG"
    )
    logger.info(kopf)

    zeilen = [
        f"  {index}. {c.a} vs. {c.b} -> {c.preference} ({c.intensity})"
        for index, c in enumerate(response.comparisons, start=1)
    ]
    for fehler in response.errors:
        zeilen.append(f"  Fehler: {fehler}")
    zeilen.append(f"  Rohantwort: {response.raw!r}")
    logger.debug("\n".join(zeilen))


def _zusammenfassung(antworten: list[Response]) -> dict[str, int]:
    gueltig = sum(1 for r in antworten if r.valid)
    return {"total": len(antworten), "valid": gueltig, "invalid": len(antworten) - gueltig}


def _dokument(
    start: datetime, ende: datetime | None, kontext: dict, antworten: list[Response]
) -> dict:
    """Prompts stehen mit in der JSON: die sprachliche Eingabe ist Teil des Messwerts."""
    meta = {
        "started": start.isoformat(timespec="seconds"),
        "finished": ende.isoformat(timespec="seconds") if ende else None,
        **{k: v for k, v in kontext.items() if k != "prompts"},
    }
    return {
        "meta": meta,
        "prompts": kontext["prompts"],
        "summary": _zusammenfassung(antworten),
        "responses": [r.to_dict() for r in antworten],
    }


def _schreibe_json(path: Path, dokument: dict) -> None:
    """Erst in eine temporäre Datei, dann umbenennen: nie eine halbe JSON."""
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(dokument, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    builder = PromptBuilder.from_paths()
    case_ids = resolve_cases(builder, args.cases)
    warne_ohne_unabhaengigkeitshinweis(builder, args.condition)
    client = build_client(args.provider, args.model)

    run_experiment(
        builder,
        client,
        runs=args.runs,
        case_ids=case_ids,
        temperature=args.temperature,
        condition=args.condition,
        seed=args.seed,
        out_dir=args.out_dir,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
