"""Erzeugt den Ergebnisteil zum Begriffstausch-Experiment als LaTeX-Datei.

Liest die Läufe von src/term_swap.py aus results/term_swap/ (je Modell das
Original und die Rotationen) und die früheren Experimentläufe aus results/
(Referenz für die Wiederholbarkeit) und schreibt eine \\input-fähige
Unterabschnitt-Datei. Nichts ist abgetippt; jede Zahl im Text steht als
\\resval-Makro (Präfix "ts-"), jede Tabelle wird aus den Daten erzeugt.

    python3 showResults/build_termswap_tex.py [--out showResults/results_termswap.tex]

build_results_tex.py setzt denselben Abschnitt über section() ans Ende von
results_rq3.tex; diese eigenständige Datei braucht man nur, wenn man ihn
getrennt einbinden will. Gewichte und CR kommen aus src/weights.py und src/ahp.py,
also aus demselben Verfahren wie Schritt 6. Abstand ist wie in results_rq3.tex
die euklidische Distanz zweier Gewichtsvektoren.

Der Text ist wie in build_results_tex.py rein deskriptiv.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
import ahp  # noqa: E402
from term_swap import ORIGINAL, OUT_DIR as TERM_SWAP_DIR  # noqa: E402
from weights import berechne  # noqa: E402

RESULTS_DIR = ROOT / "results"
OUT = HERE / "results_termswap.tex"

CRIT = ["liefertreue", "flexibilitaet", "kosten"]
CRIT_LABEL = {"liefertreue": "Delivery reliability", "flexibilitaet": "Flexibility", "kosten": "Cost"}
CASES = ["ketten", "batteriepacks", "konnektivitaetsmodule"]
CASE_LABEL = {"ketten": "Chains", "batteriepacks": "Battery packs", "konnektivitaetsmodule": "Connectivity modules"}
CASE_LOWER = {"ketten": "chains", "batteriepacks": "battery packs", "konnektivitaetsmodule": "connectivity modules"}
# Reihenfolge der Modelle wie in results_rq3.tex.
MODELS = ["gpt-5.4-mini-2026-03-17", "claude-haiku-4-5-20251001", "mistral-small-2603"]
TABLE_DISTANCE = "tab:termswap-distance"
TABLE_WEIGHTS = "tab:termswap-weights"
TABLE_SPREAD = "tab:weights-spread"  # Gewichtstabelle in results_rq3.tex
# Richtungswechsel nur bei klarer Mehrheit in beiden Bedingungen; ein 53:47 im
# Original ist keine Richtung, die wechseln könnte.
CLEAR_MAJORITY = 0.75
# Darunter: Verschiebung des Anteils einer Antwort um mindestens so viel.
SHARE_SHIFT = 0.25
TEX_SPECIALS = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(text: str) -> str:
    return "".join(TEX_SPECIALS.get(z, z) for z in str(text))


def slug(text: str) -> str:
    return "".join(z if z.isalnum() or z == "-" else "-" for z in text.lower())


def num(wert: float, stellen: int = 3) -> str:
    return f"{wert:.{stellen}f}"


def signed_pp(wert: float) -> str:
    return "$0.0$" if abs(wert) < 0.05 else f"${wert:+.1f}$"


def pct(anteil: float) -> str:
    return f"{anteil * 100:.0f}\\%"


def aufzaehlen(teile: list[str]) -> str:
    if len(teile) <= 2:
        return " and ".join(teile)
    return ", ".join(teile[:-1]) + " and " + teile[-1]


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- laden


def find_term_swap(ordner: Path, modelle: list[str]) -> dict[str, dict[str, Path]]:
    """Je Modell und Variante die neueste abgeschlossene Datei."""
    gefunden: dict[str, dict[str, Path]] = {}
    for pfad in sorted(Path(ordner).glob("*_termswap-*.json")):
        meta = _json(pfad)["meta"]
        if meta.get("finished") and meta.get("term_swap"):
            gefunden.setdefault(meta["model"], {})[meta["term_swap"]["variant"]] = pfad
    fehlt = [m for m in modelle if ORIGINAL not in gefunden.get(m, {}) or len(gefunden[m]) < 2]
    if fehlt:
        raise SystemExit(f"Kein vollständiger Begriffstausch in {ordner} für: {', '.join(fehlt)}")
    return {m: gefunden[m] for m in modelle}


def find_earlier(ordner: Path, original: Path) -> Path | None:
    """Neuester früherer Experimentlauf desselben Modells mit zeichengleichen
    Prompts und Einstellungen; ohne Treffer keine Referenz."""
    neu = _json(original)
    kandidaten = []
    for pfad in sorted(Path(ordner).glob("*.json")):
        try:
            alt = _json(pfad)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        meta = alt.get("meta") or {}
        if (meta.get("model") == neu["meta"]["model"] and meta.get("condition") == neu["meta"]["condition"]
                and meta.get("temperature") == neu["meta"]["temperature"] and meta.get("finished")
                and (meta.get("runs") or 0) > 1 and not meta.get("gate1") and not meta.get("justification")
                and alt.get("prompts") == neu["prompts"]):
            kandidaten.append(pfad)
    return kandidaten[-1] if kandidaten else None


# ------------------------------------------------------------ Kennzahlen


def cell_stats(dokument: dict, case_id: str, config: dict) -> dict:
    """Mittlere Gewichte, CR und Mehrheitsurteil je Paar für einen Fall einer Datei."""
    responses = [r for r in dokument["responses"] if r["case_id"] == case_id]
    gueltig = [r for r in responses if r["valid"]]
    berechnet = [a for a in berechne({"responses": gueltig}, CRIT, config)["responses"]]
    n = len(berechnet)
    if not n:
        raise SystemExit(f"Keine gültige Antwort für {case_id}")
    urteile: dict[tuple[str, str], Counter] = {}
    for r in gueltig:
        for c in r["comparisons"]:
            urteile.setdefault((c["a"], c["b"]), Counter())[c["preference"]] += 1
    return {
        "n": n,
        "invalid": len(responses) - n,
        "weights": {c: sum(a["weights"][c] for a in berechnet) / n for c in CRIT},
        "cr": sum(a["cr"] for a in berechnet) / n,
        "majority": {paar: z.most_common(1)[0] for paar, z in urteile.items()},
        "shares": {paar: {p: k / n for p, k in z.items()} for paar, z in urteile.items()},
    }


def distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt(sum((a[c] - b[c]) ** 2 for c in CRIT))


def collect(dateien: dict[str, dict[str, Path]], frueher: dict[str, Path | None]) -> dict:
    config = ahp.load_config()
    daten: dict = {"models": {}, "runs": None, "temperature": None, "condition": None, "invalid": 0, "total": 0}
    for modell, varianten in dateien.items():
        zellen: dict[tuple[str, str], dict] = {}
        for variante, pfad in varianten.items():
            dokument = _json(pfad)
            meta = dokument["meta"]
            for feld in ("runs", "temperature", "condition"):
                if daten[feld] is None:
                    daten[feld] = meta[feld]
                elif daten[feld] != meta[feld]:
                    raise SystemExit(f"{pfad.name}: {feld}={meta[feld]} weicht von {daten[feld]} ab")
            daten["total"] += dokument["summary"]["total"]
            daten["invalid"] += dokument["summary"]["invalid"]
            for case_id in meta["cases"]:
                label = meta["term_swap"]["labels"][case_id]
                zellen[(case_id, label)] = cell_stats(dokument, case_id, config)
        for (inhalt, label), z in zellen.items():
            basis = zellen[(inhalt, inhalt)]
            z["distance"] = distance(z["weights"], basis["weights"])
            z["delta_pp"] = {c: 100 * (z["weights"][c] - basis["weights"][c]) for c in CRIT}
            z["flipped"], z["shifted"] = [], []
            for paar, (pref, anzahl) in z["majority"].items():
                alt_pref, alt_anzahl = basis["majority"][paar]
                if (pref != alt_pref and anzahl / z["n"] >= CLEAR_MAJORITY
                        and alt_anzahl / basis["n"] >= CLEAR_MAJORITY):
                    z["flipped"].append(paar)
                    continue
                antworten = set(z["shares"][paar]) | set(basis["shares"][paar])
                if max(abs(z["shares"][paar].get(p, 0) - basis["shares"][paar].get(p, 0)) for p in antworten) >= SHARE_SHIFT:
                    z["shifted"].append(paar)
        referenz = {}
        if frueher.get(modell):
            alt = _json(frueher[modell])
            for case_id in CASES:
                referenz[case_id] = distance(cell_stats(alt, case_id, config)["weights"],
                                             zellen[(case_id, case_id)]["weights"])
        daten["models"][modell] = {"cells": zellen, "reference": referenz, "earlier": frueher.get(modell)}
    daten["variants"] = sorted({v for d in dateien.values() for v in d})
    return daten


# ---------------------------------------------------------------- LaTeX


def macros(daten: dict) -> dict[str, str]:
    werte = {
        "ts-runs": str(daten["runs"]),
        "ts-temperature": f"{daten['temperature']:g}",
        "ts-condition": daten["condition"],
        "ts-total": str(daten["total"]),
        "ts-invalid": str(daten["invalid"]),
        "ts-n-models": str(len(daten["models"])),
        "ts-clear-majority": pct(CLEAR_MAJORITY),
        "ts-share-shift": f"{SHARE_SHIFT * 100:.0f}",
    }
    alle_d, alle_ref = [], []
    for modell, m in daten["models"].items():
        k = slug(modell)
        getauscht = [z["distance"] for (i, l), z in m["cells"].items() if i != l]
        werte[f"ts-dist-max.{k}"] = num(max(getauscht))
        werte[f"ts-dist-min.{k}"] = num(min(getauscht))
        alle_d += getauscht
        if m["reference"]:
            werte[f"ts-ref-max.{k}"] = num(max(m["reference"].values()))
            alle_ref += list(m["reference"].values())
        for (inhalt, label), z in m["cells"].items():
            schluessel = f"{k}.{inhalt}.{label}"
            werte[f"ts-dist.{schluessel}"] = num(z["distance"])
            for c in CRIT:
                werte[f"ts-w.{schluessel}.{c}"] = num(z["weights"][c])
            for (a, b), anteile in z["shares"].items():
                for pref, anteil in anteile.items():
                    werte[f"ts-share.{schluessel}.{a}-{b}.{pref}"] = pct(anteil)
        for case_id, d in m["reference"].items():
            werte[f"ts-ref.{k}.{case_id}"] = num(d)
    werte["ts-dist-max.models"] = num(max(alle_d))
    if alle_ref:
        werte["ts-ref-max.models"] = num(max(alle_ref))
        werte["ts-ref-min.models"] = num(min(alle_ref))
    return werte


def distance_table(daten: dict) -> str:
    kopf = " & ".join(["Earlier run"] + [f"as {CASE_LABEL[c]}" for c in CASES])
    z = [
        r"\begin{table}[htbp]",
        r"  \centering\footnotesize",
        "  \\caption{Influence of the component names: distance between the mean weight vector of a set of "
        "statements under an exchanged component name and its mean weight vector under the original name, "
        f"per model ({daten['runs']} runs per combination). \\emph{{Earlier run}} gives, as a reference, the "
        "distance between the rerun of the original in this experiment and the earlier run in "
        f"Table~\\ref{{{TABLE_SPREAD}}}, with identical prompts and settings. "
        "A bold value marks a combination in which at least one pairwise comparison changes direction: at least "
        f"{CLEAR_MAJORITY * 100:.0f}\\% of the runs prefer one criterion under the original name and at least "
        f"{CLEAR_MAJORITY * 100:.0f}\\% the other criterion under the exchanged name.}}",
        f"  \\label{{{TABLE_DISTANCE}}}",
        r"  \begin{tabular}{@{}lrrrr@{}}",
        r"    \toprule",
        f"    Statements of & {kopf} \\\\",
        r"    \midrule",
    ]
    for index, (modell, m) in enumerate(daten["models"].items()):
        if index:
            z.append(r"    \addlinespace")
        z.append(f"    \\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{tex(modell)}}}}} \\\\")
        for inhalt in CASES:
            zellen = [num(m["reference"][inhalt]) if inhalt in m["reference"] else "--"]
            for label in CASES:
                zelle = m["cells"].get((inhalt, label))
                if label == inhalt or zelle is None:
                    zellen.append("--")
                else:
                    wert = num(zelle["distance"])
                    zellen.append(f"\\textbf{{{wert}}}" if zelle["flipped"] else wert)
            z.append(f"    {CASE_LABEL[inhalt]} & " + " & ".join(zellen) + r" \\")
    z += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(z)


def weights_table(daten: dict) -> str:
    z = [
        r"\begin{table}[htbp]",
        r"  \centering\footnotesize",
        "  \\caption{Mean weight per criterion for each set of statements under each component name, with the "
        "difference to the original name in percentage points in parentheses, the mean consistency ratio (CR) and "
        "the number of valid runs. The first row of each block is the original name.}",
        f"  \\label{{{TABLE_WEIGHTS}}}",
        r"  \begin{tabular}{@{}llrrrrr@{}}",
        r"    \toprule",
        "    Statements of & Name & " + " & ".join(CRIT_LABEL[c] for c in CRIT) + r" & CR & $n$ \\",
        r"    \midrule",
    ]
    for index, (modell, m) in enumerate(daten["models"].items()):
        if index:
            z.append(r"    \addlinespace")
        z.append(f"    \\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{tex(modell)}}}}} \\\\")
        for inhalt in CASES:
            reihenfolge = [inhalt] + [l for l in CASES if l != inhalt]
            for erste, label in zip([True, False, False], reihenfolge):
                zelle = m["cells"].get((inhalt, label))
                if zelle is None:
                    continue
                gewichte = [
                    num(zelle["weights"][c]) + ("" if label == inhalt else f" ({signed_pp(zelle['delta_pp'][c])})")
                    for c in CRIT
                ]
                z.append(f"    {CASE_LABEL[inhalt] if erste else ''} & {CASE_LABEL[label]} & "
                         + " & ".join(gewichte) + f" & {num(zelle['cr'])} & {zelle['n']} \\\\")
    z += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(z)


def prose(daten: dict) -> str:
    modelle = list(daten["models"])
    keys = {m: slug(m) for m in modelle}
    absaetze = []

    absaetze.append(
        "To test whether the component names themselves influence the judgements, the statements of each case "
        "were presented under the names of the other two components. Only the component names and the words that "
        "depend on them grammatically were exchanged; the statements were otherwise identical, character by "
        "character. Together with the original assignment, two rotations present each set of statements once "
        "under each of the three names (a Latin square). Each of the \\resval{ts-n-models} models rated every "
        "combination in \\resval{ts-runs} runs (\\resval{ts-condition}, temperature \\resval{ts-temperature}). "
        "The original assignment was rerun in the same session, so that all combinations were collected under "
        "the same conditions. Of the \\resval{ts-total} responses, \\resval{ts-invalid} were invalid and are not "
        "included."
    )

    saetze = [
        f"Table~\\ref{{{TABLE_DISTANCE}}} gives the distance between the mean weight vector of a set of statements "
        f"under an exchanged name and under its original name, and Table~\\ref{{{TABLE_WEIGHTS}}} the underlying "
        "mean weights."
    ]
    if any(m["reference"] for m in daten["models"].values()):
        saetze.append(
            "As a reference, the rerun of the original lies at a distance of \\resval{ts-ref-min.models} to "
            f"\\resval{{ts-ref-max.models}} from the earlier run in Table~\\ref{{{TABLE_SPREAD}}}, although prompts, "
            "model versions and settings were identical."
        )
    teile = [f"\\resval{{ts-dist-min.{keys[m]}}} to \\resval{{ts-dist-max.{keys[m]}}} for {tex(m)}" for m in modelle]
    saetze.append("Under an exchanged name, the distance ranges from " + aufzaehlen(teile) + ".")
    absaetze.append(" ".join(saetze))

    def name(p: str) -> str:
        return "equal importance" if p == "gleich" else CRIT_LABEL[p].lower()

    def aufteilung(k: str, inhalt: str, label: str, paar: tuple[str, str], anteile: dict[str, float]) -> str:
        a, b = paar
        teile = sorted(anteile.items(), key=lambda t: -t[1])
        return aufzaehlen([f"\\resval{{ts-share.{k}.{inhalt}.{label}.{a}-{b}.{p}}} {name(p)}" for p, _ in teile])

    def vergleich(paar: tuple[str, str]) -> str:
        return f"{CRIT_LABEL[paar[0]].lower()} vs.\\ {CRIT_LABEL[paar[1]].lower()}"

    n_kombi = sum(1 for m in daten["models"].values() for (i, l) in m["cells"] if i != l)
    saetze = []
    # Richtungswechsel: klare Mehrheit in beiden Bedingungen, entgegengesetzt.
    for modell in modelle:
        k = keys[modell]
        zellen = daten["models"][modell]["cells"]
        for (inhalt, label), z in zellen.items():
            basis = zellen[(inhalt, inhalt)]
            for paar in z["flipped"]:
                a, b = paar
                alt_pref, neu_pref = basis["majority"][paar][0], z["majority"][paar][0]
                saetze.append(
                    f"For {tex(modell)}, the comparison {vergleich(paar)} changes direction when the statements of "
                    f"the {CASE_LOWER[inhalt]} case appear under the name \\emph{{{CASE_LOWER[label]}}}: under the "
                    f"original name, \\resval{{ts-share.{k}.{inhalt}.{inhalt}.{a}-{b}.{alt_pref}}} of the runs prefer "
                    f"{name(alt_pref)}, under the exchanged name \\resval{{ts-share.{k}.{inhalt}.{label}.{a}-{b}.{neu_pref}}} "
                    f"prefer {name(neu_pref)}. The weights move from "
                    + aufzaehlen([f"\\resval{{ts-w.{k}.{inhalt}.{inhalt}.{c}}}" for c in CRIT])
                    + " to "
                    + aufzaehlen([f"\\resval{{ts-w.{k}.{inhalt}.{label}.{c}}}" for c in CRIT])
                    + f" ({', '.join(CRIT_LABEL[c].lower() for c in CRIT)}), a distance of "
                    f"\\resval{{ts-dist.{k}.{inhalt}.{label}}}."
                )
    n_wechsel = sum(len(z["flipped"]) > 0 for m in daten["models"].values() for (i, l), z in m["cells"].items() if i != l)
    if n_wechsel:
        saetze.append(f"No other of the {n_kombi} combinations with an exchanged name contains such a change of direction, "
                      "with a clear majority of \\resval{ts-clear-majority} of the runs on both sides.")
    else:
        saetze.append(f"None of the {n_kombi} combinations with an exchanged name contains a comparison that changes "
                      "direction with a clear majority of \\resval{ts-clear-majority} of the runs on both sides.")

    # Verschiebungen der Antwortanteile ohne klaren Richtungswechsel, je Modell, Fall und Paar gebündelt.
    gruppen: dict[tuple[str, str, tuple[str, str]], list[str]] = {}
    for modell in modelle:
        for (inhalt, label), z in daten["models"][modell]["cells"].items():
            for paar in z["shifted"]:
                gruppen.setdefault((modell, inhalt, paar), []).append(label)
    for (modell, inhalt, paar), labels in gruppen.items():
        k = keys[modell]
        zellen = daten["models"][modell]["cells"]
        neu = [
            f"under the name \\emph{{{CASE_LOWER[l]}}} "
            + aufteilung(k, inhalt, l, paar, zellen[(inhalt, l)]["shares"][paar])
            for l in sorted(labels, key=CASES.index)
        ]
        saetze.append(
            f"Without such a change of direction, the shares of the answers shift by at least "
            f"\\resval{{ts-share-shift}} percentage points for {tex(modell)} in the comparison {vergleich(paar)} of the "
            f"{CASE_LOWER[inhalt]} case: under the original name, the runs prefer "
            + aufteilung(k, inhalt, inhalt, paar, zellen[(inhalt, inhalt)]["shares"][paar])
            + ", " + aufzaehlen(neu) + "."
        )
    if not gruppen:
        saetze.append("In no other comparison does the share of an answer shift by \\resval{ts-share-shift} "
                      "percentage points or more.")
    absaetze.append(" ".join(saetze))

    return "\n\n".join(absaetze)


HEADING = "Influence of the component names"


def load(term_swap_dir: Path = TERM_SWAP_DIR, results_dir: Path = RESULTS_DIR
         ) -> tuple[dict[str, dict[str, Path]], dict[str, Path | None]]:
    """Begriffstausch-Läufe je Modell und die früheren Läufe als Referenz."""
    dateien = find_term_swap(term_swap_dir, MODELS)
    return dateien, {m: find_earlier(results_dir, dateien[m][ORIGINAL]) for m in MODELS}


def section(dateien: dict[str, dict[str, Path]], frueher: dict[str, Path | None], heading: str
            ) -> tuple[list[str], str, list[str]]:
    """(\\defresval-Zeilen, Abschnitt ab der Überschrift, Quelldateien).
    Auch von build_results_tex.py genutzt, das den Abschnitt in results_rq3.tex setzt."""
    daten = collect(dateien, frueher)
    quellen = [p.name for d in dateien.values() for p in d.values()] + [p.name for p in frueher.values() if p]
    absaetze = [textwrap.fill(a, width=84, break_long_words=False, break_on_hyphens=False)
                for a in prose(daten).split("\n\n")]
    body = "\n\n".join([heading, *absaetze, distance_table(daten), weights_table(daten)])
    return [f"\\defresval{{{k}}}{{{v}}}" for k, v in sorted(macros(daten).items())], body, quellen


def build(dateien: dict[str, dict[str, Path]], frueher: dict[str, Path | None]) -> str:
    zeilen, body, quellen = section(dateien, frueher, f"\\subsection*{{{HEADING}}}")
    return "\n".join([
        f"% Erzeugt von build_termswap_tex.py aus {', '.join(quellen)}. Nicht von Hand aendern.",
        r"% Benoetigt im Praeambel-Teil: \usepackage{booktabs}. Gedacht zum Einbinden hinter results_rq3.tex",
        r"% (verweist auf tab:weights-spread).",
        r"\makeatletter",
        r"\providecommand{\defresval}[2]{\expandafter\gdef\csname res@#1\endcsname{#2}}",
        r"\providecommand{\resval}[1]{%",
        r"  \ifcsname res@#1\endcsname\csname res@#1\endcsname\else\textbf{??#1}\fi}",
        r"\makeatother",
        "",
        *zeilen,
        "",
        body,
        "",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_termswap_tex.py", description=__doc__.splitlines()[0])
    parser.add_argument("--term-swap-dir", type=Path, default=TERM_SWAP_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Frühere Experimentläufe als Referenz (Default results/)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    dateien, frueher = load(args.term_swap_dir, args.results_dir)
    args.out.write_text(build(dateien, frueher), encoding="utf-8")
    print(args.out)
    for m in MODELS:
        print(f"  {m}: {', '.join(p.name for p in dateien[m].values())}")
        print(f"    Referenz: {frueher[m].name if frueher[m] else '(keine)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
