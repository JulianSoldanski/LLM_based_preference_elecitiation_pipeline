"""Gewichte und CR zu einer Ergebnisdatei von experiment.py (Schritt 6).

    python3 src/weights.py results/<lauf>.json

Schreibt <lauf>_weights.json daneben: je Antwort Gewichte, lambda_max, CI,
CR und ob CR unter der Schwelle liegt, dazu je Fall die Mittelwerte über die
gültigen Antworten. Ungültige Antworten bekommen keine Gewichte, nur einen
Vermerk. Die Datei wird schreibgeschützt; eine vorhandene wird nicht
überschrieben. Gerechnet wird nur mit src/ahp.py, ohne Modellaufruf.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import ahp
import ledger
from prompt_builder import PromptBuilder


def weights_pfad(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}_weights.json")


def berechne(dokument: dict[str, Any], criteria: list[str], config: dict[str, Any]) -> dict[str, Any]:
    """Reine Rechnung über alle Antworten einer Ergebnisdatei."""
    antworten, je_fall = [], {}
    for r in dokument["responses"]:
        kopf = {"run": r["run"], "case_id": r["case_id"], "valid": r["valid"]}
        if not r["valid"]:
            antworten.append({**kopf, "note": "ungültige Antwort, keine Gewichte"})
            continue
        ergebnis = ahp.compute(r["comparisons"], criteria)
        antworten.append({**kopf, **ahp.evaluation(ergebnis, config)})
        je_fall.setdefault(r["case_id"], []).append(ergebnis)

    zusammenfassung = {
        case_id: {
            "n_valid": len(liste),
            "mean_weights": {c: sum(e.weights[c] for e in liste) / len(liste) for c in criteria},
            "mean_cr": sum(e.cr for e in liste) / len(liste),
            "share_above_threshold": sum(not ahp.is_consistent(e, config) for e in liste) / len(liste),
        }
        for case_id, liste in je_fall.items()
    }
    return {"responses": antworten, "summary": zusammenfassung}


def schreibe(result_path: Path, config_path: Path = ahp.AHP_CONFIG) -> Path:
    result_path = Path(result_path)
    ziel = weights_pfad(result_path)
    if ziel.exists():
        raise SystemExit(f"{ziel.name} existiert bereits und wird nicht überschrieben.")
    dokument = json.loads(result_path.read_text(encoding="utf-8"))
    config = ahp.load_config(config_path)
    criteria = [c.id for c in PromptBuilder.from_paths().criteria]
    inhalt = {
        "meta": {
            "source_file": result_path.name,
            "source_sha256": ledger.sha256(result_path),
            "criteria": criteria,
            "method": ahp.METHOD,
            "random_index_source": config.get("random_index_source"),
            "cr_threshold": {str(n): ahp.cr_threshold(n, config) for n in (len(criteria),)},
            "config_file": Path(config_path).name,
            "config_sha256": ledger.sha256(config_path),
        },
        **berechne(dokument, criteria, config),
    }
    ziel.write_text(json.dumps(inhalt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ziel.chmod(0o444)
    return ziel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="weights.py", description="Gewichte und CR zu einer Ergebnisdatei.")
    parser.add_argument("result", type=Path, help="Ergebnisdatei von experiment.py")
    parser.add_argument("--config", type=Path, default=ahp.AHP_CONFIG, help="Default config/ahp.yaml")
    args = parser.parse_args(argv)
    ziel = schreibe(args.result, args.config)
    inhalt = json.loads(ziel.read_text(encoding="utf-8"))
    for case_id, s in inhalt["summary"].items():
        gewichte = " · ".join(f"{c} {w:.3f}" for c, w in s["mean_weights"].items())
        print(f"{case_id}: {gewichte} · CR Ø {s['mean_cr']:.3f} · über Schwelle {s['share_above_threshold']:.0%}")
    print(f"-> {ziel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
