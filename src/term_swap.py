"""Begriffstausch-Experiment: Wie stark hängen die Urteile am Bauteilnamen?

Die Aussagen eines Falls bleiben zeichengleich, nur die Bauteilbegriffe
(Ketten, Batteriepacks, Konnektivitätsmodule) werden gegen die eines anderen
Falls getauscht. Ändern sich die Gewichte, hat das Modell aus dem Begriff
selbst etwas abgeleitet (z.B. Wert oder Kritikalität des Bauteils), statt
allein die Aussagen zu gewichten.

Design (config/term_swap.yaml): lateinisches Quadrat aus dem Original und zwei
Rotationen. Jeder Aussagensatz läuft einmal unter jedem Bauteilnamen. Die
Fall-ID bleibt die des Inhalts; den angezeigten Begriff hält meta.term_swap fest.

    python3 src/term_swap.py --dry-run                         # Prompts + Ersetzungen ansehen
    python3 src/term_swap.py --model gpt-4o-2024-08-06 --runs 10
    python3 src/term_swap.py --model gpt-4o-2024-08-06 --runs 10 --variant rot1
    python3 src/term_swap.py --report results/term_swap/*.json

Ohne --variant laufen das Original und alle Rotationen mit identischen
Einstellungen nacheinander, damit der Vergleich nicht an Modellversion oder
Zeitpunkt hängt. Ergebnisse landen in results/term_swap/ (getrennt von den
Experimentläufen in results/, deren Prompt ein Test festhält).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import ahp
import ledger
from agent import build_client
from experiment import run_experiment
from prompt_builder import CONFIG_DIR, ConfigError, PromptBuilder, _load_yaml
from runner import RESULTS_DIR, experiment_builder, resolve_cases, warne_ohne_unabhaengigkeitshinweis
from schema import CONDITION_HISTORY, CONDITION_STATELESS, Case
from weights import berechne

TERM_SWAP_FILE = CONFIG_DIR / "term_swap.yaml"
OUT_DIR = RESULTS_DIR / "term_swap"
ORIGINAL = "original"
SUFFIX = "_termswap-{variant}"


@dataclass(frozen=True)
class TermSwap:
    components: dict[str, str]
    stems: dict[str, list[str]]
    variants: dict[str, dict[str, str]]
    replacements: dict[str, dict[str, list[dict[str, Any]]]]

    def mapping(self, variant: str) -> dict[str, str]:
        """Inhalt -> Begriff; das Original ist die Identität."""
        if variant == ORIGINAL:
            return {case_id: case_id for case_id in self.components}
        if variant not in self.variants:
            raise ConfigError(f"Unbekannte Variante {variant!r} (bekannt: {self.variant_names()})")
        return dict(self.variants[variant])

    def variant_names(self) -> list[str]:
        return [ORIGINAL, *self.variants]


def load_term_swap(path: Path | str = TERM_SWAP_FILE) -> TermSwap:
    roh = _load_yaml(Path(path))
    spec = TermSwap(
        components={str(k): str(v) for k, v in roh.get("components", {}).items()},
        stems={str(k): [str(s).casefold() for s in v] for k, v in roh.get("stems", {}).items()},
        variants={str(n): {str(k): str(v) for k, v in m.items()} for n, m in roh.get("variants", {}).items()},
        replacements={
            str(case_id): {str(cid): list(liste or []) for cid, liste in (je_kriterium or {}).items()}
            for case_id, je_kriterium in roh.get("replacements", {}).items()
        },
    )
    _validate_spec(spec)
    return spec


def _validate_spec(spec: TermSwap) -> None:
    probleme = []
    ids = set(spec.components)
    if set(spec.stems) != ids:
        probleme.append(f"stems deckt die Bauteile {sorted(ids)} nicht genau ab")
    if ORIGINAL in spec.variants:
        probleme.append(f"Variante {ORIGINAL!r} ist reserviert")
    for name, mapping in spec.variants.items():
        if set(mapping) != ids or sorted(mapping.values()) != sorted(ids):
            probleme.append(f"Variante {name!r} ist keine Permutation der Bauteile {sorted(ids)}")
    for case_id, je_kriterium in spec.replacements.items():
        if case_id not in ids:
            probleme.append(f"replacements: unbekanntes Bauteil {case_id!r}")
        for cid, liste in je_kriterium.items():
            for eintrag in liste:
                if not eintrag.get("from") or not isinstance(eintrag.get("to"), dict):
                    probleme.append(f"replacements.{case_id}.{cid}: Eintrag braucht 'from' und 'to'")
    if probleme:
        raise ConfigError("term_swap.yaml inkonsistent:\n- " + "\n- ".join(probleme))


def swap_case(case: Case, target: str, spec: TermSwap) -> Case:
    """Der Fall mit den Aussagen von `case` und dem Bauteilbegriff von `target`.

    Laut statt still: jedes 'from' muss genau einmal vorkommen, und danach darf
    kein Wortstamm des ursprünglichen Bauteils übrig sein.
    """
    if target == case.id:
        return case
    if case.id not in spec.components or target not in spec.components:
        raise ConfigError(f"Tausch {case.id!r} -> {target!r}: Bauteil fehlt in term_swap.yaml")
    alt_name, neu_name = spec.components[case.id], spec.components[target]
    if case.component != alt_name:
        raise ConfigError(f"Fall {case.id!r}: component {case.component!r} ≠ {alt_name!r} aus term_swap.yaml")
    if case.title.count(alt_name) != 1:
        raise ConfigError(f"Fall {case.id!r}: {alt_name!r} steht nicht genau einmal im Titel {case.title!r}")

    aussagen = dict(case.statements)
    for cid, liste in spec.replacements.get(case.id, {}).items():
        if cid not in aussagen:
            raise ConfigError(f"Fall {case.id!r}: Kriterium {cid!r} hat keine Aussage")
        for eintrag in liste:
            von, nach = eintrag["from"], eintrag["to"].get(target)
            if nach is None:
                raise ConfigError(f"Fall {case.id!r}/{cid}: keine Form für {target!r} zu {von!r}")
            anzahl = aussagen[cid].count(von)
            if anzahl != 1:
                raise ConfigError(f"Fall {case.id!r}/{cid}: {von!r} kommt {anzahl}-mal vor statt einmal")
            aussagen[cid] = aussagen[cid].replace(von, nach)

    titel = case.title.replace(alt_name, neu_name)
    reste = [
        f"{wo}: {stamm!r}"
        for wo, text in [("Titel", titel), *aussagen.items()]
        for stamm in spec.stems[case.id]
        if stamm in text.casefold()
    ]
    if reste:
        raise ConfigError(f"Fall {case.id!r} -> {target!r}: Begriff nicht ersetzt ({', '.join(reste)})")
    return replace(case, title=titel, component=neu_name, statements=aussagen)


def swap_builder(builder: PromptBuilder, variant: str, spec: TermSwap) -> PromptBuilder:
    mapping = spec.mapping(variant)
    fehlend = [case.id for case in builder.cases if case.id not in mapping]
    if fehlend:
        raise ConfigError(f"Variante {variant!r}: Fälle {fehlend} fehlen in term_swap.yaml")
    return replace(builder, cases=[swap_case(case, mapping[case.id], spec) for case in builder.cases])


def swap_meta(variant: str, spec: TermSwap, path: Path | str = TERM_SWAP_FILE) -> dict[str, Any]:
    mapping = spec.mapping(variant)
    return {
        "statements_source": "cases.yaml" if variant == ORIGINAL else "cases.yaml+term_swap.yaml",
        "term_swap": {
            "variant": variant,
            "labels": mapping,
            "components": {case_id: spec.components[label] for case_id, label in mapping.items()},
            "config_file": Path(path).name,
            "config_sha256": ledger.sha256(path),
        },
    }


# ------------------------------------------------------------- Auswertung


def report(paths: list[Path], config_path: Path = ahp.AHP_CONFIG) -> dict[str, Any]:
    """Mittlere Gewichte je Inhalt (Fall-ID) und Begriff, dazu die Abweichung
    vom Original (gleicher Inhalt, eigener Begriff) in Prozentpunkten."""
    config = ahp.load_config(config_path)
    criteria = [c.id for c in PromptBuilder.from_paths().criteria]
    zellen: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pfad in paths:
        dokument = json.loads(Path(pfad).read_text(encoding="utf-8"))
        tausch = dokument["meta"].get("term_swap")
        if not tausch:
            raise SystemExit(f"{Path(pfad).name}: kein meta.term_swap, kein Lauf von term_swap.py")
        for antwort in berechne(dokument, criteria, config)["responses"]:
            if antwort["valid"]:
                label = tausch["labels"][antwort["case_id"]]
                zellen.setdefault((antwort["case_id"], label), []).append(antwort)

    ergebnis: dict[str, Any] = {"criteria": criteria, "cells": []}
    for (inhalt, label), liste in sorted(zellen.items()):
        mittel = {c: sum(a["weights"][c] for a in liste) / len(liste) for c in criteria}
        ergebnis["cells"].append({
            "content": inhalt, "label": label, "n_valid": len(liste),
            "mean_weights": mittel, "mean_cr": sum(a["cr"] for a in liste) / len(liste),
        })
    original = {z["content"]: z["mean_weights"] for z in ergebnis["cells"] if z["content"] == z["label"]}
    for zelle in ergebnis["cells"]:
        basis = original.get(zelle["content"])
        zelle["delta_pp"] = (
            {c: 100 * (zelle["mean_weights"][c] - basis[c]) for c in criteria} if basis else None
        )
    return ergebnis


def print_report(ergebnis: dict[str, Any]) -> None:
    criteria = ergebnis["criteria"]
    kopf = f"{'Inhalt':<22} {'Begriff':<22} {'n':>3}  " + "  ".join(f"{c[:12]:>12}" for c in criteria) + "    CR Ø"
    print(kopf)
    print("-" * len(kopf))
    for z in ergebnis["cells"]:
        werte = "  ".join(
            f"{z['mean_weights'][c]:>6.3f}" + (f" {z['delta_pp'][c]:+5.1f}" if z["delta_pp"] and z["content"] != z["label"] else " " * 6)
            for c in criteria
        )
        print(f"{z['content']:<22} {z['label']:<22} {z['n_valid']:>3}  {werte}  {z['mean_cr']:.3f}")
    print("\nZahl hinter dem Gewicht: Abweichung vom Original desselben Inhalts in Prozentpunkten.")


# ------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="term_swap.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", choices=["openai", "anthropic", "mistral"], default="openai")
    parser.add_argument("--model", help="Modell-ID des Anbieters (außer bei --dry-run/--report Pflicht)")
    parser.add_argument("--variant", action="append", dest="variants",
                        help="original, rot1, rot2 …; mehrfach angebbar. Ohne Angabe: alle inkl. Original.")
    parser.add_argument("--condition", choices=[CONDITION_HISTORY, CONDITION_STATELESS], default=CONDITION_STATELESS)
    parser.add_argument("--seed", type=int, help="Seed der Fallreihenfolge (nur history), für alle Varianten gleich")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--case", action="append", dest="cases", help="Fall-ID (Inhalt); mehrfach angebbar")
    parser.add_argument("--justification", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Ersetzungen und User-Prompts ausgeben, kein Modellaufruf")
    parser.add_argument("--report", nargs="+", type=Path, metavar="JSON", help="Ergebnisdateien auswerten")
    parser.add_argument("--report-json", type=Path, help="Auswertung zusätzlich als JSON schreiben")
    args = parser.parse_args(argv)
    if not (args.dry_run or args.report) and not args.model:
        parser.error("--model ist erforderlich (außer bei --dry-run/--report)")
    if args.runs < 1:
        parser.error("--runs muss mindestens 1 sein")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.report:
        ergebnis = report(args.report)
        print_report(ergebnis)
        if args.report_json:
            args.report_json.write_text(json.dumps(ergebnis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0

    spec = load_term_swap()
    varianten = args.variants or spec.variant_names()
    unbekannt = [v for v in varianten if v not in spec.variant_names()]
    if unbekannt:
        raise SystemExit(f"Unbekannte Varianten {unbekannt}. Bekannt: {spec.variant_names()}")
    basis = experiment_builder(args.justification)
    case_ids = resolve_cases(basis, args.cases)
    warne_ohne_unabhaengigkeitshinweis(basis, args.condition)
    builders = {v: swap_builder(basis, v, spec) for v in varianten}  # prüft alle vor dem ersten Aufruf

    if args.dry_run:
        for variant, builder in builders.items():
            for case_id in case_ids:
                label = spec.mapping(variant)[case_id]
                print("=" * 72)
                print(f"{variant}: Aussagen {case_id} unter Begriff {spec.components[label]}")
                print("=" * 72)
                print(builder.build_user(case_id))
                print()
        return 0

    client = build_client(args.provider, args.model)
    for variant, builder in builders.items():
        if not args.quiet:
            print(f"== Variante {variant}: " + ", ".join(f"{k} -> {v}" for k, v in spec.mapping(variant).items()))
        run_experiment(
            builder, client, runs=args.runs, case_ids=case_ids, temperature=args.temperature,
            condition=args.condition, seed=args.seed, out_dir=args.out_dir, quiet=args.quiet,
            suffix=SUFFIX.format(variant=variant), extra_meta=swap_meta(variant, spec),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
