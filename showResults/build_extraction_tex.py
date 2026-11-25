"""Erzeugt die Tabellen zu Schritt 1 (Aussagen suchen) und Schritt 3 (Zuordnung).

Liest die Wiederholungsläufe aus results/relevance_runs/ (relevance_experiment.py,
je Fall eine Datei) und einen Zuordnungslauf mit mehreren Wiederholungen aus
results/assignment/ (assignment.py --runs n) und schreibt daraus eine
\\input-fähige LaTeX-Datei mit zwei Tabellen. Nichts ist abgetippt; jede Zahl
im Text steht zusätzlich als \\resval-Makro.

    python3 showResults/build_extraction_tex.py
    python3 showResults/build_extraction_tex.py --relevance <a.json> <b.json> <c.json> --assignment <z.json>

Ohne Angabe: je Fall die neueste abgeschlossene Datei in results/relevance_runs/
und der neueste abgeschlossene Zuordnungslauf mit mehr als einer Wiederholung.

Tabelle 1 (Schritt 1) trennt die Fehlerarten: falsch negativ (übersehen,
kritisch, die Aussage erreicht Gate 1 nie) und falsch positiv (wird in Gate 1
gelöscht). Tabelle 2 (Schritt 3) zeigt je Aussage, wie oft sie in n Läufen dem
erwarteten Kriterium zugeordnet wurde und wohin sonst.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from relevance_experiment import ANDERES, GESUCHT, STATEMENT, auswerten  # noqa: E402

RELEVANCE_DIR = ROOT / "results" / "relevance_runs"
ASSIGNMENT_DIR = ROOT / "results" / "assignment"
OUT = HERE / "results_extraction.tex"

CASES = [("ketten", "Chains"), ("batteriepacks", "Battery packs"), ("konnektivitaetsmodule", "Connectivity modules")]
CRITERIA = [("DR", "Delivery reliability"), ("FR", "Flexibility"), ("CT", "Cost"), ("none", "none")]
# Passagentypen für die falsch-positiven Zeilen, ohne Unterscheidung des Dokuments,
# außer bei bewertenden Aussagen anderer Produkte.
TYPES = [
    (STATEMENT + ANDERES, "Statement on another product"),
    ("near_miss", "Near miss (same wording, other topic)"),
    ("neighbor", "Neighbour (procurement process)"),
    ("off_topic", "Off-topic"),
]
TEX_SPECIALS = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(text: str) -> str:
    return "".join(TEX_SPECIALS.get(z, z) for z in str(text))


def num(wert: float | None, stellen: int = 2) -> str:
    return "--" if wert is None else f"{wert:.{stellen}f}"


def pct(anteil: float | None, stellen: int = 0) -> str:
    return "--" if anteil is None else f"{anteil * 100:.{stellen}f}\\%"


# ---------------------------------------------------------------- laden


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def newest_relevance(ordner: Path = RELEVANCE_DIR) -> list[Path]:
    """Je Fall die neueste abgeschlossene Datei."""
    gefunden = {}
    for pfad in sorted(Path(ordner).glob("*.json")):
        meta = _json(pfad)["meta"]
        if meta.get("step") == "relevance_experiment" and meta.get("finished"):
            gefunden[meta["case_id"]] = pfad
    fehlt = [c for c, _ in CASES if c not in gefunden]
    if fehlt:
        raise SystemExit(f"Keine abgeschlossene Datei in {ordner} für: {', '.join(fehlt)}")
    return [gefunden[c] for c, _ in CASES]


def newest_assignment(ordner: Path = ASSIGNMENT_DIR) -> Path:
    kandidaten = []
    for pfad in sorted(Path(ordner).glob("*.json")):
        try:
            meta = _json(pfad).get("meta") or {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if (meta.get("runs") or 0) > 1 and meta.get("finished") and not meta.get("search_file"):
            kandidaten.append(pfad)
    if not kandidaten:
        raise SystemExit(f"Kein abgeschlossener Zuordnungslauf mit mehreren Wiederholungen in {ordner}")
    return kandidaten[-1]


# ------------------------------------------------------------ Kennzahlen


def relevance_stats(dokument: dict) -> dict:
    a = auswerten(dokument)
    n = a["runs_completed"]
    passagen = a["passages"]
    gesucht = [p for p in passagen if p["group"] == GESUCHT]
    typ_quote = {}
    for schluessel, _ in TYPES:
        if schluessel.endswith(ANDERES):
            mitglieder = [p for p in passagen if p["group"] == schluessel]
        else:
            mitglieder = [p for p in passagen if p["group"] in (schluessel, schluessel + ANDERES)]
        typ_quote[schluessel] = (sum(p["selected"] for p in mitglieder), len(mitglieder) * n)
    return {
        "meta": dokument["meta"],
        "runs": n,
        "passages": len(passagen),
        "sought": len(gesucht),
        "recall_mean": a["recall"]["mean"], "recall_min": a["recall"]["min"],
        "fn": a["false_negatives"]["total"], "fn_runs": a["false_negatives"]["runs_with_any"],
        "fn_slots": len(gesucht) * n,
        "missed_passages": a["false_negatives"]["passages_ever_missed"],
        "precision_mean": a["precision"]["mean"],
        "fp": a["false_positives"]["total"],
        "fp_per_run": a["false_positives"]["per_run"]["mean"],
        "fp_passages": len(a["false_positives"]["passages_ever_selected"]),
        "negatives": sum(p["should_select"] is False for p in passagen),
        "types": typ_quote,
        "f2_mean": a["f2"]["mean"],
        "stable": a["stable_passages"],
        "failed": sum(lauf["failed"] for lauf in a["runs"]),
    }


def pooled(stats: list[dict]) -> dict:
    """Über alle Fälle: Summen und Quoten über alle Läufe, Mittel der Lauf-Kennzahlen."""
    runs = sum(s["runs"] for s in stats)
    gewichtet = lambda feld: sum(s[feld] * s["runs"] for s in stats) / runs  # noqa: E731
    return {
        "runs": runs,
        "passages": stats[0]["passages"],
        "sought": sum(s["sought"] for s in stats),
        "recall_mean": gewichtet("recall_mean"), "recall_min": min(s["recall_min"] for s in stats),
        "fn": sum(s["fn"] for s in stats), "fn_runs": sum(s["fn_runs"] for s in stats),
        "fn_slots": sum(s["fn_slots"] for s in stats),
        "precision_mean": gewichtet("precision_mean"),
        "fp": sum(s["fp"] for s in stats),
        "fp_per_run": gewichtet("fp_per_run"),
        "fp_passages": sum(s["fp_passages"] for s in stats),
        "negatives": sum(s["negatives"] for s in stats),
        "types": {k: (sum(s["types"][k][0] for s in stats), sum(s["types"][k][1] for s in stats)) for k, _ in TYPES},
        "f2_mean": gewichtet("f2_mean"),
        "stable": sum(s["stable"] for s in stats),
        "stable_of": sum(s["passages"] for s in stats),
        "failed": sum(s["failed"] for s in stats),
    }


def assignment_stats(dokument: dict) -> dict:
    truth = dokument["ground_truth"]
    runs = dokument["meta"]["runs"]
    je_passage: dict[str, Counter] = {pid: Counter() for pid in truth}
    doc = {}
    for r in dokument["records"]:
        if r["passage_id"] not in truth:
            continue
        doc[r["passage_id"]] = r["doc_id"]
        vorhergesagt = "failed" if r["status"] != "ok" else ((r["criteria"] or ["none"])[0])
        je_passage[r["passage_id"]][vorhergesagt] += 1
    zeilen = []
    for pid in sorted(truth):
        erwartet = (truth[pid] or ["none"])[0]
        z = je_passage[pid]
        zeilen.append({"passage_id": pid, "doc_id": doc[pid], "expected": erwartet, "counts": z,
                       "correct": z[erwartet], "runs": sum(z.values())})
    # Precision/Recall je Kriterium über alle Läufe gepoolt.
    kriterien = {}
    for code, _ in CRITERIA[:3]:
        tp = sum(z["counts"][code] for z in zeilen if z["expected"] == code)
        fp = sum(z["counts"][code] for z in zeilen if z["expected"] != code)
        fn = sum(z["runs"] - z["counts"][code] for z in zeilen if z["expected"] == code)
        kriterien[code] = {"precision": tp / (tp + fp) if tp + fp else None,
                           "recall": tp / (tp + fn) if tp + fn else None}
    je_lauf = Counter()
    for r in dokument["records"]:
        if r["passage_id"] in truth and r["status"] == "ok" and set(r["criteria"] or []) == set(truth[r["passage_id"]]):
            je_lauf[r["run"]] += 1
    exact = [je_lauf[run] / len(truth) for run in range(1, runs + 1)]
    return {"meta": dokument["meta"], "runs": runs, "rows": zeilen, "criteria": kriterien,
            "exact_mean": sum(exact) / len(exact), "exact_min": min(exact), "exact_max": max(exact),
            "correct": sum(z["correct"] for z in zeilen), "total": sum(z["runs"] for z in zeilen),
            "failed": sum(z["counts"]["failed"] for z in zeilen),
            "stable": sum(z["correct"] in (0, z["runs"]) and max(z["counts"].values()) == z["runs"] for z in zeilen)}


# ---------------------------------------------------------------- LaTeX


def relevance_table(stats: list[dict], gesamt: dict) -> str:
    spalten = [*stats, gesamt]

    def zeile(titel: str, werte: list[str]) -> str:
        return f"    {titel} & " + " & ".join(werte) + r" \\"

    def anteil(wert: tuple[int, int]) -> str:
        return pct(wert[0] / wert[1] if wert[1] else None, 1)

    meta = stats[0]["meta"]
    kopf = " & ".join([*(name for _, name in CASES), "All cases"])
    z = [
        r"\begin{table}[htbp]",
        r"  \centering\footnotesize",
        "  \\caption{Step 1 (relevance screening): " + str(stats[0]["runs"]) + " repeated runs per case with "
        + tex(meta["model_requested"]) + f" at temperature {meta['temperature']:g} over all "
        + str(stats[0]["passages"]) + " passages of \\texttt{" + tex(meta["passages_file"]) + "}. "
        "False negatives (missed statements) are reported first because they are critical: a missed "
        "statement never reaches Gate~1. False positives are removed by the reviewer in Gate~1. "
        "Rates are pooled over all runs; recall, precision and $F_2$ are means of the per-run values.}",
        r"  \label{tab:relevance-screening}",
        r"  \begin{tabular}{@{}lrrrr@{}}",
        r"    \toprule",
        f"     & {kopf} \\\\",
        r"    \midrule",
        r"    \multicolumn{5}{@{}l}{\textit{False negatives (critical)}} \\",
        zeile("Sought statements", [str(s["sought"]) for s in spalten]),
        zeile("Recall, mean", [num(s["recall_mean"]) for s in spalten]),
        zeile("Recall, worst run", [num(s["recall_min"]) for s in spalten]),
        zeile("Missed statements", [f"{s['fn']} / {s['fn_slots']}" for s in spalten]),
        zeile("Runs with $\\geq 1$ miss", [f"{s['fn_runs']} / {s['runs']}" for s in spalten]),
        r"    \addlinespace",
        r"    \multicolumn{5}{@{}l}{\textit{False positives (removed in Gate~1)}} \\",
        zeile("Precision, mean", [num(s["precision_mean"]) for s in spalten]),
        zeile("Wrongly selected per run", [num(s["fp_per_run"], 1) for s in spalten]),
        zeile("Passages ever wrongly selected", [f"{s['fp_passages']} / {s['negatives']}" for s in spalten]),
        *(zeile(f"\\quad {name}", [anteil(s["types"][k]) for s in spalten]) for k, name in TYPES),
        r"    \addlinespace",
        r"    \multicolumn{5}{@{}l}{\textit{Overall}} \\",
        zeile("$F_2$, mean", [num(s["f2_mean"]) for s in spalten]),
        zeile("Same decision in all runs",
              [f"{s['stable']} / {s.get('stable_of', s['passages'])}" for s in spalten]),
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \par\smallskip\raggedright\scriptsize Rows under false positives give the share of runs in "
        r"which a passage of that type was selected. Failed responses count as not selected "
        f"({gesamt['failed']} in all runs).",
        r"\end{table}",
    ]
    return "\n".join(z)


def assignment_table(a: dict) -> str:
    meta = a["meta"]
    codes = [c for c, _ in CRITERIA]
    kopf = " & ".join(f"\\multicolumn{{1}}{{c}}{{{c}}}" for c in codes)
    z = [
        r"\begin{table}[htbp]",
        r"  \centering\footnotesize",
        "  \\caption{Step 3 (assignment): " + str(a["runs"]) + " repeated runs with "
        + tex(meta["model_requested"]) + f" at temperature {meta['temperature']:g} and prompt "
        + "\\texttt{" + tex(meta["prompt_version"]) + "} over the "
        + str(len(a["rows"])) + " statements of \\texttt{" + tex(meta["passages_file"]) + "}. "
        "For each statement: the expected criterion, how often it was assigned correctly, and the "
        "distribution of all assignments (DR delivery reliability, FR flexibility, CT cost).}",
        r"  \label{tab:assignment-runs}",
        r"  \begin{tabular}{@{}llcr" + "r" * len(codes) + "@{}}",
        r"    \toprule",
        f"    Case & Statement & Expected & Correct & {kopf} \\\\",
        r"    \midrule",
    ]
    namen = dict(CASES)
    vorher = None
    for zeile in a["rows"]:
        if vorher and zeile["doc_id"] != vorher:
            z.append(r"    \addlinespace")
        fall = namen.get(zeile["doc_id"], tex(zeile["doc_id"])) if zeile["doc_id"] != vorher else ""
        vorher = zeile["doc_id"]
        zellen = [f"\\textbf{{{zeile['counts'][c]}}}" if c == zeile["expected"] else
                  (str(zeile["counts"][c]) if zeile["counts"][c] else "--") for c in codes]
        z.append(f"    {fall} & {tex(zeile['passage_id'])} & {zeile['expected']} & "
                 f"{zeile['correct']} / {zeile['runs']} & " + " & ".join(zellen) + r" \\")
    k = a["criteria"]
    z += [
        r"    \midrule",
        f"    \\multicolumn{{3}}{{@{{}}l}}{{All statements}} & {a['correct']} / {a['total']} & "
        f"\\multicolumn{{{len(codes)}}}{{l}}{{exact match per run: mean {num(a['exact_mean'])}, "
        f"min {num(a['exact_min'])}}} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \par\smallskip\raggedright\scriptsize Pooled over all runs: "
        + "; ".join(f"{c} precision {num(k[c]['precision'])}, recall {num(k[c]['recall'])}" for c, _ in CRITERIA[:3])
        + f". Failed responses: {a['failed']}.",
        r"\end{table}",
    ]
    return "\n".join(z)


def macros(stats: list[dict], gesamt: dict, a: dict) -> list[str]:
    werte = {}
    for (fall, _), s in zip(CASES, stats):
        werte[f"relevance-recall-mean.{fall}"] = num(s["recall_mean"])
        werte[f"relevance-fn.{fall}"] = str(s["fn"])
        werte[f"relevance-precision-mean.{fall}"] = num(s["precision_mean"])
    werte.update({
        "relevance-runs": str(stats[0]["runs"]),
        "relevance-recall-mean.all": num(gesamt["recall_mean"]),
        "relevance-recall-min.all": num(gesamt["recall_min"]),
        "relevance-fn.all": str(gesamt["fn"]),
        "relevance-fn-slots.all": str(gesamt["fn_slots"]),
        "relevance-fn-runs.all": str(gesamt["fn_runs"]),
        "relevance-precision-mean.all": num(gesamt["precision_mean"]),
        "relevance-fp-per-run.all": num(gesamt["fp_per_run"], 1),
        "relevance-f2-mean.all": num(gesamt["f2_mean"]),
        "assignment-runs": str(a["runs"]),
        "assignment-correct": str(a["correct"]),
        "assignment-total": str(a["total"]),
        "assignment-accuracy": pct(a["correct"] / a["total"], 1),
        "assignment-exact-mean": num(a["exact_mean"]),
    })
    return [f"\\defresval{{{k}}}{{{v}}}" for k, v in sorted(werte.items())]


def build(relevance_paths: list[Path], assignment_path: Path) -> str:
    stats = [relevance_stats(_json(p)) for p in relevance_paths]
    gesamt = pooled(stats)
    a = assignment_stats(_json(assignment_path))
    quellen = ", ".join(p.name for p in [*relevance_paths, assignment_path])
    return "\n".join([
        f"% Erzeugt von build_extraction_tex.py aus {quellen}. Nicht von Hand aendern.",
        r"% Benoetigt im Praeambel-Teil: \usepackage{booktabs}.",
        r"\makeatletter",
        r"\providecommand{\defresval}[2]{\expandafter\gdef\csname res@#1\endcsname{#2}}",
        r"\providecommand{\resval}[1]{%",
        r"  \ifcsname res@#1\endcsname\csname res@#1\endcsname\else\textbf{??#1}\fi}",
        r"\makeatother",
        "",
        *macros(stats, gesamt, a),
        "",
        relevance_table(stats, gesamt),
        "",
        assignment_table(a),
        "",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_extraction_tex.py", description=__doc__.splitlines()[0])
    parser.add_argument("--relevance", type=Path, nargs=3, metavar="JSON",
                        help="Je Fall eine Datei aus relevance_experiment.py (Reihenfolge egal)")
    parser.add_argument("--assignment", type=Path, help="Zuordnungslauf mit --runs > 1")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    relevance = args.relevance or newest_relevance()
    if args.relevance:
        nach_fall = {_json(p)["meta"]["case_id"]: p for p in args.relevance}
        relevance = [nach_fall[c] for c, _ in CASES]
    assignment = args.assignment or newest_assignment()
    args.out.write_text(build(relevance, assignment), encoding="utf-8")
    print(f"{args.out}\n  Schritt 1: {', '.join(p.name for p in relevance)}\n  Schritt 3: {assignment.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
