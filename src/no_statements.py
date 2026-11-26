"""Experiment ohne Aussagen: Welche Gewichte gibt das Modell allein zum Bauteil?

Das Modell bekommt dieselben Kriterien, dieselbe Skala, dasselbe Antwortformat
und dieselben Paarvergleiche wie in Schritt 5, aber keine Aussagen aus dem
Beschaffungsteam. Es sieht nur Fall und Bauteil (Ketten, Batteriepacks,
Konnektivitätsmodule). Die Gewichte zeigen also das Vorwissen des Modells
über das Bauteil. Der Vergleich mit den Läufen mit Aussagen zeigt, wie weit die
Aussagen die Gewichte von diesem Vorwissen wegbewegen.

    python3 src/no_statements.py --dry-run                     # Prompts ansehen
    python3 src/no_statements.py --model gpt-4o-2024-08-06 --runs 10
    python3 src/no_statements.py --report results/no_statements/*.json \\
        --reference results/term_swap/*_termswap-original.json

Ergebnisse landen in results/no_statements/, getrennt von den
Experimentläufen in results/, deren Prompt ein Test festhält.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

import ahp
import ledger
from agent import build_client
from experiment import run_experiment
from prompt_builder import PROMPTS_DIR, ConfigError, PromptBuilder, _read_text
from runner import RESULTS_DIR, experiment_builder, resolve_cases, warne_ohne_unabhaengigkeitshinweis
from schema import CONDITION_HISTORY, CONDITION_STATELESS
from weights import berechne

SYSTEM_TEMPLATE = PROMPTS_DIR / "system_no_statements.txt"
USER_TEMPLATE = PROMPTS_DIR / "user_no_statements.txt"
OUT_DIR = RESULTS_DIR / "no_statements"
SUFFIX = "_nostatements"
STATEMENTS_SOURCE = "none"
# Einstellungen, die zwischen der Bedingung und ihrer Referenz gleich sein sollen.
SETTINGS = ("condition", "temperature", "justification")


def no_statements_builder(basis: PromptBuilder | None = None) -> PromptBuilder:
    """Builder von Schritt 5 mit den Vorlagen ohne Aussagen.

    Die Fälle bleiben unverändert (die Validierung verlangt ihre Aussagen),
    nur die Vorlagen rendern sie nicht. Laut statt still: steht eine Aussage
    doch im Prompt, bricht der Aufbau ab.
    """
    basis = basis or experiment_builder()
    if basis.justification or basis.evidence:
        raise ConfigError("Ohne Aussagen gibt es keine Begründung und keine Belegstellen")
    builder = replace(
        basis,
        system_template=_read_text(SYSTEM_TEMPLATE),
        user_template=_read_text(USER_TEMPLATE),
    )
    for case in builder.cases:
        prompt = builder.build_user(case.id)
        if case.component not in prompt:
            raise ConfigError(f"Fall {case.id!r}: Bauteil {case.component!r} fehlt im Prompt")
        reste = [cid for cid, text in case.statements.items() if text[:40] in prompt]
        if reste:
            raise ConfigError(f"Fall {case.id!r}: Aussagen {reste} stehen im Prompt")
    return builder


def no_statements_meta() -> dict[str, Any]:
    return {
        "statements_source": STATEMENTS_SOURCE,
        "no_statements": {
            "system_template": SYSTEM_TEMPLATE.name,
            "system_sha256": ledger.sha256(SYSTEM_TEMPLATE),
            "user_template": USER_TEMPLATE.name,
            "user_sha256": ledger.sha256(USER_TEMPLATE),
        },
    }


# ------------------------------------------------------------- Auswertung


def _distanz(a: dict[str, float], b: dict[str, float], criteria: list[str]) -> float:
    """Euklidische Distanz zweier Gewichtsvektoren, wie in results_rq3.tex."""
    return math.sqrt(sum((a[c] - b[c]) ** 2 for c in criteria))


def _sammle(pfade: list[Path], criteria: list[str], config: dict[str, Any], *, ohne: bool):
    """Gültige Antworten mit Gewichten je (Modell, Fall), dazu die Einstellungen je Modell."""
    zellen: dict[tuple[str, str], list[dict[str, Any]]] = {}
    einstellungen: dict[str, set[tuple]] = {}
    for pfad in pfade:
        dokument = json.loads(Path(pfad).read_text(encoding="utf-8"))
        meta = dokument["meta"]
        if ohne and not meta.get("no_statements"):
            raise SystemExit(f"{Path(pfad).name}: kein meta.no_statements, kein Lauf von no_statements.py")
        if not ohne:
            if meta.get("no_statements"):
                raise SystemExit(f"{Path(pfad).name}: Lauf ohne Aussagen gehört nicht zu --reference")
            if meta.get("statements_source") != "cases.yaml":
                raise SystemExit(
                    f"{Path(pfad).name}: Referenz braucht die Aussagen aus cases.yaml "
                    f"(statements_source={meta.get('statements_source')!r})"
                )
        einstellungen.setdefault(meta["model"], set()).add(tuple(meta.get(k) for k in SETTINGS))
        for antwort in berechne(dokument, criteria, config)["responses"]:
            if antwort["valid"]:
                zellen.setdefault((meta["model"], antwort["case_id"]), []).append(antwort)
    return zellen, einstellungen


def _mittel(liste: list[dict[str, Any]], criteria: list[str]) -> dict[str, Any]:
    return {
        "n_valid": len(liste),
        "mean_weights": {c: statistics.fmean(a["weights"][c] for a in liste) for c in criteria},
        "sd_weights": {
            c: statistics.stdev(a["weights"][c] for a in liste) if len(liste) > 1 else 0.0 for c in criteria
        },
        "mean_cr": statistics.fmean(a["cr"] for a in liste),
    }


def report(paths: list[Path], reference: list[Path] | None = None,
           config_path: Path = ahp.AHP_CONFIG) -> dict[str, Any]:
    """Mittlere Gewichte ohne Aussagen je Modell und Fall. Mit Referenz zusätzlich
    die Gewichte mit Aussagen, die Verschiebung durch die Aussagen in
    Prozentpunkten (mit − ohne) und die euklidische Distanz der Mittelwerte."""
    config = ahp.load_config(config_path)
    criteria = [c.id for c in PromptBuilder.from_paths().criteria]
    ohne, einst_ohne = _sammle(paths, criteria, config, ohne=True)
    mit, einst_mit = _sammle(reference or [], criteria, config, ohne=False)

    ergebnis: dict[str, Any] = {"criteria": criteria, "cells": [], "models": [], "warnings": []}
    for modell in sorted(einst_ohne):
        if len(einst_ohne[modell]) > 1:
            ergebnis["warnings"].append(f"{modell}: Läufe ohne Aussagen mit unterschiedlichen Einstellungen")
        if modell in einst_mit and einst_mit[modell] != einst_ohne[modell]:
            ergebnis["warnings"].append(
                f"{modell}: Referenz mit anderen Einstellungen ({', '.join(SETTINGS)}): "
                f"{sorted(einst_mit[modell])} statt {sorted(einst_ohne[modell])}"
            )

    for (modell, case_id), liste in sorted(ohne.items()):
        zelle = {"model": modell, "case_id": case_id, **_mittel(liste, criteria), "reference": None}
        if (modell, case_id) in mit:
            ref = _mittel(mit[(modell, case_id)], criteria)
            zelle["reference"] = ref
            zelle["delta_pp"] = {c: 100 * (ref["mean_weights"][c] - zelle["mean_weights"][c]) for c in criteria}
            zelle["distance"] = _distanz(ref["mean_weights"], zelle["mean_weights"], criteria)
        ergebnis["cells"].append(zelle)

    # Unterscheidet das Modell die Bauteile schon ohne Aussagen? Mittlere
    # paarweise Distanz zwischen den Fällen, ohne und mit Aussagen.
    for modell in sorted(einst_ohne):
        eigene = {z["case_id"]: z for z in ergebnis["cells"] if z["model"] == modell}
        paare = list(combinations(sorted(eigene), 2))
        zwischen_ohne = [_distanz(eigene[a]["mean_weights"], eigene[b]["mean_weights"], criteria) for a, b in paare]
        zwischen_mit = [
            _distanz(eigene[a]["reference"]["mean_weights"], eigene[b]["reference"]["mean_weights"], criteria)
            for a, b in paare if eigene[a]["reference"] and eigene[b]["reference"]
        ]
        ergebnis["models"].append({
            "model": modell,
            "between_cases_without": statistics.fmean(zwischen_ohne) if zwischen_ohne else None,
            "between_cases_with": statistics.fmean(zwischen_mit) if zwischen_mit and len(zwischen_mit) == len(paare) else None,
        })
    return ergebnis


def print_report(ergebnis: dict[str, Any]) -> None:
    criteria = ergebnis["criteria"]
    kopf = (f"{'Modell':<28} {'Fall':<22} {'n':>4}  "
            + "  ".join(f"{c[:12]:>12}" for c in criteria) + "    CR Ø   Distanz")
    print(kopf)
    print("-" * len(kopf))
    for z in ergebnis["cells"]:
        werte = "  ".join(
            f"{z['mean_weights'][c]:>6.3f}" + (f" {z['delta_pp'][c]:+5.1f}" if z["reference"] else " " * 6)
            for c in criteria
        )
        distanz = f"{z['distance']:.3f}" if z["reference"] else "–"
        print(f"{z['model']:<28} {z['case_id']:<22} {z['n_valid']:>4}  {werte}  {z['mean_cr']:.3f}   {distanz:>7}")
    print("\nGewicht ohne Aussagen; Zahl dahinter: Verschiebung durch die Aussagen (mit − ohne) in Prozentpunkten.")
    print("\nMittlere Distanz zwischen den Fällen (unterscheidet das Modell die Bauteile?):")
    for m in ergebnis["models"]:
        ohne = f"{m['between_cases_without']:.3f}" if m["between_cases_without"] is not None else "–"
        mit = f"{m['between_cases_with']:.3f}" if m["between_cases_with"] is not None else "–"
        print(f"  {m['model']:<28} ohne Aussagen {ohne}   mit Aussagen {mit}")
    for warnung in ergebnis["warnings"]:
        print(f"WARNUNG: {warnung}")


# ------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="no_statements.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", choices=["openai", "anthropic", "mistral"], default="openai")
    parser.add_argument("--model", help="Modell-ID des Anbieters (außer bei --dry-run/--report Pflicht)")
    parser.add_argument("--condition", choices=[CONDITION_HISTORY, CONDITION_STATELESS], default=CONDITION_STATELESS)
    parser.add_argument("--seed", type=int, help="Seed der Fallreihenfolge (nur history)")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--case", action="append", dest="cases", help="Fall-ID; mehrfach angebbar")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="System- und User-Prompts ausgeben, kein Modellaufruf")
    parser.add_argument("--report", nargs="+", type=Path, metavar="JSON", help="Läufe ohne Aussagen auswerten")
    parser.add_argument("--reference", nargs="+", type=Path, metavar="JSON",
                        help="Läufe mit Aussagen aus cases.yaml zum Vergleich (z.B. *_termswap-original.json)")
    parser.add_argument("--report-json", type=Path, help="Auswertung zusätzlich als JSON schreiben")
    args = parser.parse_args(argv)
    if args.reference and not args.report:
        parser.error("--reference nur zusammen mit --report")
    if not (args.dry_run or args.report) and not args.model:
        parser.error("--model ist erforderlich (außer bei --dry-run/--report)")
    if args.runs < 1:
        parser.error("--runs muss mindestens 1 sein")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.report:
        ergebnis = report(args.report, args.reference)
        print_report(ergebnis)
        if args.report_json:
            args.report_json.write_text(json.dumps(ergebnis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0

    builder = no_statements_builder()
    case_ids = resolve_cases(builder, args.cases)
    warne_ohne_unabhaengigkeitshinweis(builder, args.condition)

    if args.dry_run:
        print("=" * 72 + "\nSYSTEM\n" + "=" * 72)
        print(builder.build_system())
        for case_id in case_ids:
            print("=" * 72 + f"\nUSER: {case_id}\n" + "=" * 72)
            print(builder.build_user(case_id))
        return 0

    client = build_client(args.provider, args.model)
    run_experiment(
        builder, client, runs=args.runs, case_ids=case_ids, temperature=args.temperature,
        condition=args.condition, seed=args.seed, out_dir=args.out_dir, quiet=args.quiet,
        suffix=SUFFIX, extra_meta=no_statements_meta(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
