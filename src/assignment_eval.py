"""Auswertung einer Zuordnung gegen expected_criteria.

Getrennt von assignment.py: dort wird zugeordnet, hier nur gezählt. Gelesen wird
die JSON-Datei eines Durchlaufs; die Ground Truth steht dort im Abschnitt
"ground_truth" und nicht in den Datensätzen.

    python3 src/assignment_eval.py results/assignment/<datei>.json

Je Lauf: Exact-Match-Accuracy (vorhergesagte Menge == erwartete Menge) und
Precision/Recall je Kriterium. Eine Passage mit Status "failed" zählt als leere
Vorhersage: für die Accuracy falsch, für den Recall ein Fehlen. Ist ein Nenner
null, steht die Kennzahl auf null (JSON) bzw. "-" (Tabelle).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from assignment import STATUS_OK


def evaluate(dokument: dict[str, Any]) -> list[dict[str, Any]]:
    truth = dokument.get("ground_truth") or {}
    codes = dokument["meta"]["criteria"]
    ergebnisse = []
    for run in sorted({r["run"] for r in dokument["records"]}):
        bewertet = [r for r in dokument["records"] if r["run"] == run and r["passage_id"] in truth]
        if not bewertet:
            continue
        treffer = 0
        zaehler = {code: Counter() for code in codes}
        for record in bewertet:
            vorhergesagt = set(record["criteria"] or [])
            erwartet = set(truth[record["passage_id"]])
            treffer += record["status"] == STATUS_OK and vorhergesagt == erwartet
            for code in codes:
                if code in vorhergesagt and code in erwartet:
                    zaehler[code]["tp"] += 1
                elif code in vorhergesagt:
                    zaehler[code]["fp"] += 1
                elif code in erwartet:
                    zaehler[code]["fn"] += 1
        ergebnisse.append(
            {
                "run": run,
                "passages": len(bewertet),
                "failed": sum(r["status"] != STATUS_OK for r in bewertet),
                "exact_match": treffer / len(bewertet),
                "criteria": {
                    code: {
                        "precision": _quote(z["tp"], z["tp"] + z["fp"]),
                        "recall": _quote(z["tp"], z["tp"] + z["fn"]),
                        "tp": z["tp"],
                        "fp": z["fp"],
                        "fn": z["fn"],
                    }
                    for code, z in zaehler.items()
                },
            }
        )
    return ergebnisse


def format_table(ergebnisse: list[dict[str, Any]]) -> str:
    zeilen = []
    for e in ergebnisse:
        zeilen.append(
            f"Lauf {e['run']}: Exact Match {e['exact_match']:.2f} "
            f"({e['passages']} Passagen, davon {e['failed']} failed)"
        )
        zeilen.append(f"  {'Kriterium':<10}{'Precision':>10}{'Recall':>8}{'TP':>4}{'FP':>4}{'FN':>4}")
        for code, m in e["criteria"].items():
            zeilen.append(
                f"  {code:<10}{_zahl(m['precision']):>10}{_zahl(m['recall']):>8}"
                f"{m['tp']:>4}{m['fp']:>4}{m['fn']:>4}"
            )
    return "\n".join(zeilen)


def evaluate_file(path: Path) -> tuple[Path | None, str]:
    """Schreibt <datei>_eval.json neben die Ergebnisdatei. Ohne Ground Truth: (None, "")."""
    dokument = json.loads(Path(path).read_text(encoding="utf-8"))
    ergebnisse = evaluate(dokument)
    if not ergebnisse:
        return None, ""
    out = Path(path).with_name(f"{Path(path).stem}_eval.json")
    out.write_text(
        json.dumps({"source": Path(path).name, "runs": ergebnisse}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out, format_table(ergebnisse)


def _quote(zaehler: int, nenner: int) -> float | None:
    return zaehler / nenner if nenner else None


def _zahl(wert: float | None) -> str:
    return "-" if wert is None else f"{wert:.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="assignment_eval.py", description=__doc__.splitlines()[0])
    parser.add_argument("result", type=Path, help="JSON-Datei aus assignment.py")
    args = parser.parse_args(argv)

    out, tabelle = evaluate_file(args.result)
    if out is None:
        print("Die Datei enthält keine Ground Truth (expected_criteria).", file=sys.stderr)
        return 1
    print(f"{tabelle}\n\nAuswertung: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
