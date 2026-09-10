"""CLI für die Erhebung.

Schreibt Rohurteile als JSONL. Es findet keine Auswertung statt: Gewichte,
Matrizen und Konsistenzmaße gehören in einen späteren Schritt, der für
Fragebogen- und LLM-Daten identisch sein muss.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Iterable

from agent import AhpAgent, build_client
from prompt_builder import ROOT, PromptBuilder
from schema import CONDITION_HISTORY, CONDITION_STATELESS, Response

RESULTS_DIR = ROOT / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="runner.py",
        description="Erhebt AHP-Paarvergleiche eines LLM zu den konfigurierten Fällen.",
    )
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    # History ist die Hauptbedingung, stateless die Kontrolle.
    parser.add_argument(
        "--condition",
        choices=[CONDITION_HISTORY, CONDITION_STATELESS],
        default=CONDITION_HISTORY,
        help="history: alle Fälle in einer Konversation. stateless: ein Fall pro Aufruf.",
    )
    # Kein Default-Modell: ein geratenes Modell wäre in den Ergebnissen nicht
    # von einer bewussten Wahl zu unterscheiden.
    parser.add_argument("--model", help="Modell-ID des Anbieters (außer bei --dry-run Pflicht)")
    parser.add_argument("--runs", type=int, default=1, help="Wiederholungen je Fall (Default 1)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Fall-ID; mehrfach angebbar. Ohne Angabe werden alle Fälle erhoben.",
    )
    parser.add_argument("--out", type=Path, help="Zieldatei (Default results/<provider>_<model>_<condition>.jsonl)")
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed für die Fallreihenfolge. Nur in der History-Bedingung wirksam; "
        "ohne Angabe wird einer gezogen und in den Metadaten festgehalten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Gibt alle Prompts aus, ohne ein Modell aufzurufen.",
    )

    args = parser.parse_args(argv)
    if not args.dry_run and not args.model:
        parser.error("--model ist erforderlich (außer bei --dry-run)")
    if args.runs < 1:
        parser.error("--runs muss mindestens 1 sein")
    return args


def model_slug(model: str) -> str:
    """Modell-ID als Dateinamensbestandteil."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def default_out_path(provider: str, model: str, condition: str) -> Path:
    """Je Anbieter/Modell/Bedingung eine eigene Datei, damit nichts vermischt."""
    return RESULTS_DIR / f"{provider}_{model_slug(model)}_{condition}.jsonl"


def resolve_seed(condition: str, seed: int | None) -> int | None:
    """Ohne Vorgabe wird ein Seed gezogen und mitgeschrieben, damit die
    Reihenfolge auch nachträglich reproduzierbar bleibt. Ohne Verlauf gibt es
    nichts zu permutieren, dort bleibt der Seed leer."""
    if condition != CONDITION_HISTORY:
        return None
    return seed if seed is not None else random.randrange(2**32)


def case_sequences(
    case_ids: list[str], runs: int, condition: str, seed: int | None
) -> list[list[str]]:
    """Fallreihenfolge je Durchlauf.

    In der History-Bedingung hat die Reihenfolge Bedeutung, weil jeder Fall den
    Kontext des nächsten bildet; sie wird deshalb je Durchlauf neu permutiert.
    Ohne Verlauf sind die Aufrufe unabhängig, dort bleibt die konfigurierte
    Reihenfolge stehen.
    """
    if condition != CONDITION_HISTORY:
        return [list(case_ids) for _ in range(runs)]

    zufall = random.Random(seed)
    folgen = []
    for _ in range(runs):
        folge = list(case_ids)
        zufall.shuffle(folge)
        folgen.append(folge)
    return folgen


def warne_ohne_unabhaengigkeitshinweis(builder: PromptBuilder, condition: str) -> None:
    """Der Hinweis hat nur mit Verlauf einen Adressaten; fehlt er dort, ist die
    Bedingung nicht die, die erhoben werden soll."""
    if condition == CONDITION_HISTORY and not builder.independence_note():
        print(
            "Warnung: scenario.independence_note in config/cases.yaml ist leer. "
            "In der History-Bedingung fehlt damit der Unabhängigkeitshinweis aus "
            "der Fragebogeninstruktion.",
            file=sys.stderr,
        )


def resolve_cases(builder: PromptBuilder, selected: list[str] | None) -> list[str]:
    if not selected:
        return builder.case_ids()
    unknown = [cid for cid in selected if cid not in builder.case_ids()]
    if unknown:
        raise SystemExit(f"Unbekannte Fall-IDs: {unknown}. Bekannt: {builder.case_ids()}")
    return selected


def print_prompts(builder: PromptBuilder, case_ids: list[str]) -> None:
    print("=" * 72)
    print("SYSTEM-PROMPT (für alle Fälle identisch)")
    print("=" * 72)
    print(builder.build_system())
    for case_id in case_ids:
        print()
        print("=" * 72)
        print(f"USER-PROMPT: {case_id}")
        print("=" * 72)
        print(builder.build_user(case_id))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    builder = PromptBuilder.from_paths()
    case_ids = resolve_cases(builder, args.cases)

    # Auch im Trockenlauf warnen: dort wird der Prompt vor der Erhebung geprüft.
    warne_ohne_unabhaengigkeitshinweis(builder, args.condition)

    if args.dry_run:
        print_prompts(builder, case_ids)
        return 0

    out_path = args.out or default_out_path(args.provider, args.model, args.condition)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agent = AhpAgent(
        builder=builder,
        client=build_client(args.provider, args.model),
        temperature=args.temperature,
    )
    seed = resolve_seed(args.condition, args.seed)
    folgen = case_sequences(case_ids, args.runs, args.condition, seed)
    if seed is not None:
        print(f"Bedingung {args.condition}, Seed {seed}")

    total = valid = 0
    # Append-only: bereits erhobene Zeilen bleiben erhalten, auch wenn derselbe
    # Lauf später ergänzt wird.
    with out_path.open("a", encoding="utf-8") as handle:
        # Läufe außen, Fälle innen: ein Abbruch hinterlässt vollständige
        # Durchgänge über alle Fälle statt einen überrepräsentierten Fall.
        for run, folge in enumerate(folgen, start=1):
            antworten: Iterable[Response]
            if args.condition == CONDITION_HISTORY:
                print(f"Lauf {run}/{args.runs} Reihenfolge: {', '.join(folge)}", flush=True)
                antworten = agent.judge_sequence(folge, run=run, seed=seed)
            else:
                # Generator statt Liste: so wird nach jedem Aufruf sofort geschrieben.
                antworten = (agent.judge(case_id, run=run) for case_id in folge)

            for response in antworten:
                handle.write(response.to_json_line() + "\n")
                handle.flush()  # nach jeder Zeile, damit ein Abbruch nichts vernichtet
                total += 1
                valid += int(response.valid)
                status = "gültig" if response.valid else f"UNGÜLTIG: {'; '.join(response.errors)}"
                print(f"[{response.case_id} | Lauf {run}/{args.runs}] {status}", flush=True)

    print(f"\n{total} Antworten, davon {valid} gültig -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
