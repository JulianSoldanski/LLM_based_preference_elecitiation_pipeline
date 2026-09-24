"""Erzeugt den Ergebnisteil (RQ3 und Gewichtungsteil RQ4) als LaTeX-Datei.

Gerechnet wird in `auswertung.ipynb`: dort entstehen Gewichte, CR und die
Abbildungen. Dieses Skript liest nur das exportierte `runs.csv` und schreibt
daraus eine \\input-faehige Section. Jede Zahl im Text steht als Makro, jede
Tabelle und jede Abbildung wird aus den Daten erzeugt; nichts ist abgetippt.

    python3 build_results_tex.py [--runs runs.csv] [--out results_rq3.tex]

Der Text ist bewusst rein deskriptiv: er nennt die gemessenen Werte und
vergleicht sie, bewertet sie aber nicht.
"""

from __future__ import annotations

import argparse
import itertools
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Feste Reihenfolge der Kriterien; muss zu den Spalten in runs.csv passen.
CRIT = ["liefertreue", "flexibilitaet", "kosten"]
CRIT_LABEL = {
    "liefertreue": "Delivery reliability",
    "flexibilitaet": "Flexibility",
    "kosten": "Cost",
}
CASE_LABEL = {
    "ketten": "Chains",
    "batteriepacks": "Battery packs",
    "konnektivitaetsmodule": "Connectivity modules",
}
# Quelle der Probandendaten in runs.csv (Spalte "quelle"). HUMAN_KEY ist der
# feste Bestandteil der Makroschluessel: er haengt nicht am Anzeigenamen, damit
# eine Umbenennung im Notebook die Verweise im Fliesstext nicht bricht.
HUMAN_SOURCE = "Participants"
HUMAN_LABEL = "Participants"
HUMAN_KEY = "human"
LEGACY_HUMAN_SOURCES = ["Probanden"]

# Ueberschrift des Abschnitts; die Nummer davor vergibt LaTeX.
TITLE = "RQ3: Consistency and Stability Results"

# Rechenverfahren aus dem Notebook; steht hier nur fuer den Methodensatz im
# Text und muss mit AHP_METHOD in auswertung.ipynb uebereinstimmen.
WEIGHT_METHOD = "row geometric mean"
RANDOM_INDEX = "0.58"
CR_THRESHOLD = 0.1

# Abbildungen aus dem Notebook. Fehlende Dateien werden uebersprungen.
FIGURES = {
    "weights": ("01_gewichte", "Weights per run and per participant"),
    "consistency": ("02_konsistenz", "Consistency ratio per case and source"),
    "rankings": ("03_rangfolgen", "Criterion rankings per case and source"),
}


# --------------------------------------------------------------------------
# LaTeX-Hilfen
# --------------------------------------------------------------------------

TEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex(text: str) -> str:
    """Maskiert LaTeX-Sonderzeichen in Bezeichnern aus den Daten."""
    return "".join(TEX_SPECIALS.get(z, z) for z in str(text))


def slug(text: str) -> str:
    """Schluessel fuer die Wertmakros: Kleinbuchstaben, Ziffern, Bindestrich."""
    gesaeubert = "".join(z.lower() if z.isalnum() else "-" for z in str(text))
    return "-".join(teil for teil in gesaeubert.split("-") if teil)


def num(wert: float, stellen: int = 3) -> str:
    return "--" if pd.isna(wert) else f"{wert:.{stellen}f}"


def pct(anteil: float, stellen: int = 0) -> str:
    return "--" if pd.isna(anteil) else f"{anteil * 100:.{stellen}f}\\%"


class Values:
    """Sammelt die Zahlen und schreibt sie als \\defresval-Zeilen."""

    def __init__(self) -> None:
        self._werte: dict[str, str] = {}

    def set(self, schluessel: str, wert: str) -> str:
        if schluessel in self._werte and self._werte[schluessel] != wert:
            raise ValueError(f"Schluessel doppelt mit anderem Wert: {schluessel}")
        self._werte[schluessel] = wert
        return wert

    def lines(self) -> list[str]:
        return [f"\\defresval{{{k}}}{{{v}}}" for k, v in sorted(self._werte.items())]

    def __len__(self) -> int:
        return len(self._werte)


MACRO_PREAMBLE = r"""% Zugriff auf die Werte: \resval{schluessel}. Fehlt ein Schluessel, steht
% ??schluessel im Satz, statt dass die Zahl stillschweigend verschwindet.
\makeatletter
\providecommand{\defresval}[2]{\expandafter\gdef\csname res@#1\endcsname{#2}}
\providecommand{\resval}[1]{%
  \ifcsname res@#1\endcsname\csname res@#1\endcsname\else\textbf{??#1}\fi}
\makeatother"""


def table(caption: str, label: str, spalten: str, kopf: list[str],
          gruppen: list[tuple[str | None, list[list[str]]]]) -> str:
    """Booktabs-Tabelle.

    `gruppen` sind Bloecke aus Titel und Zeilen; der Titel wird als eigene
    Zeile ueber den Block gesetzt, statt in jeder Zeile zu stehen. Das spart
    eine Spalte, und die Tabellen bleiben bei langen Modellnamen im Satzspiegel.
    """
    zeilen = [
        r"\begin{table}[htbp]",
        r"  \centering\footnotesize",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{spalten}}}",
        r"    \toprule",
        "    " + " & ".join(kopf) + r" \\",
        r"    \midrule",
    ]
    for i, (titel, block) in enumerate(gruppen):
        if i:
            zeilen.append(r"    \addlinespace")
        if titel:
            zeilen.append(f"    \\multicolumn{{{len(kopf)}}}{{@{{}}l}}"
                          f"{{\\textit{{{titel}}}}} \\\\")
        zeilen += ["    " + " & ".join(zelle) + r" \\" for zelle in block]
    zeilen += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(zeilen)


def join_path(*teile: str) -> str:
    """Pfad mit Schraegstrich, leere Bestandteile fallen weg."""
    return "/".join(t.strip("/") for t in teile if t and t.strip("/"))


def graphics_path(tex_dir: str, figure_dir: str) -> str:
    """Abbildungspfad, wie er in der LaTeX-Datei stehen muss.

    LaTeX loest relative Pfade gegenueber der Hauptdatei auf, nicht gegenueber
    der per \\input eingebundenen Datei. Liegt die Section in einem Unterordner,
    gehoert dieser deshalb mit in den Pfad.
    """
    return join_path(tex_dir, figure_dir)


def figure(name: str, caption: str, label: str, figure_dir: str, breite: str) -> str:
    return "\n".join([
        r"\begin{figure}[htbp]",
        r"  \centering",
        f"  \\includegraphics[width={breite}\\linewidth]{{{figure_dir}/{name}}}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"\end{figure}",
    ])


# --------------------------------------------------------------------------
# Kennzahlen
# --------------------------------------------------------------------------

def distances(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Euklidische Abstaende zwischen Gewichtsvektoren.

    Ohne `b` alle Paare innerhalb von `a` (jedes Paar einmal), mit `b` alle
    Paare zwischen den beiden Mengen.
    """
    if b is None:
        paare = list(itertools.combinations(range(len(a)), 2))
        if not paare:
            return np.empty(0)
        i, j = np.array(paare).T
        return np.linalg.norm(a[i] - a[j], axis=1)
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).ravel()


def load_ground_truth(path: Path, cases: list[str]) -> dict:
    """Liest die Gewichte, aus denen die Faelle erzeugt wurden.

    Fehlt die Datei, entfaellt der Abschnitt; das Skript laeuft weiter. Fehlt
    ein Fall oder ein Kriterium, ist das ein Fehler: ein still ausgelassener
    Vergleich waere schlimmer als ein Abbruch.
    """
    import yaml

    if not path.is_file():
        print(f"Ohne Ground-Truth-Abschnitt, Datei fehlt: {path}")
        return {}

    roh = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    eintraege = roh.get("cases") or {}
    gt: dict[str, dict] = {}
    for case in cases:
        werte = eintraege.get(case)
        if werte is None:
            raise SystemExit(f"{path}: Fall {case!r} fehlt")
        fehlend = [c for c in CRIT if c not in werte]
        if fehlend:
            raise SystemExit(f"{path}: {case}: Kriterien fehlen: {fehlend}")
        vektor = np.array([float(werte[c]) for c in CRIT])
        summe = vektor.sum()
        if abs(summe - 1.0) > 0.01:
            print(f"Warnung: {case}: Gewichte summieren sich auf {summe:.4f}, nicht auf 1")
        gt[case] = {"weights": vektor, "strategy": werte.get("strategy", "")}
    return {"cases": gt, "source": roh.get("source", "")}


def load_runs(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Liest runs.csv und haelt die Reihenfolge aus der Datei fest."""
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    df["quelle"] = df["quelle"].replace(dict.fromkeys(LEGACY_HUMAN_SOURCES, HUMAN_SOURCE))
    fehlend = {"quelle", "case", "run", "CR", *CRIT} - set(df.columns)
    if fehlend:
        raise SystemExit(f"{path}: Spalten fehlen: {', '.join(sorted(fehlend))}")

    sources = list(dict.fromkeys(df["quelle"]))
    cases = [c for c in CASE_LABEL if c in set(df["case"])]
    unbekannt = set(df["case"]) - set(CASE_LABEL)
    if unbekannt:
        raise SystemExit(f"{path}: Faelle ohne Label in CASE_LABEL: {sorted(unbekannt)}")
    return df, sources, cases


def key_of(quelle: str) -> str:
    """Schluesselbestandteil einer Quelle fuer die Wertmakros."""
    return HUMAN_KEY if quelle == HUMAN_SOURCE else slug(quelle)


def source_labels(sources: list[str]) -> tuple[dict[str, str], str | None]:
    """Kurze Bezeichner fuer die Tabellen.

    Die Quellen aus dem Notebook heissen "modell (bedingung)". Laeuft alles in
    derselben Bedingung, steht sie einmal im Text statt in jeder Tabellenzeile.
    """
    modelle = [q for q in sources if q != HUMAN_SOURCE]
    bedingungen = {q.rsplit(" (", 1)[1][:-1] for q in modelle
                   if q.endswith(")") and " (" in q}
    gemeinsam = bedingungen.pop() if len(bedingungen) == 1 else None

    labels = {HUMAN_SOURCE: HUMAN_LABEL}
    for q in modelle:
        labels[q] = q.rsplit(" (", 1)[0] if gemeinsam else q
    return labels, gemeinsam


def collect(df: pd.DataFrame, sources: list[str], cases: list[str],
            vals: Values, gt: dict | None = None) -> dict:
    """Rechnet alle Kennzahlen und legt sie zugleich als Makros ab."""
    llm_sources = [q for q in sources if q != HUMAN_SOURCE]
    hat_probanden = HUMAN_SOURCE in sources
    labels, bedingung = source_labels(sources)

    if bedingung:
        vals.set("condition", tex(bedingung))
    vals.set("models", ", ".join(tex(labels[q]) for q in llm_sources))
    vals.set("n-sources", str(len(sources)))
    vals.set("n-models", str(len(llm_sources)))
    vals.set("n-cases", str(len(cases)))
    vals.set("n-criteria", str(len(CRIT)))
    vals.set("cr-threshold", num(CR_THRESHOLD, 2))
    vals.set("weight-method", WEIGHT_METHOD)
    vals.set("random-index", RANDOM_INDEX)

    kennzahlen: dict[tuple[str, str], dict] = {}
    for quelle in sources:
        for case in cases:
            g = df[(df["quelle"] == quelle) & (df["case"] == case)]
            if g.empty:
                continue
            w = g[CRIT].to_numpy()
            cr = g["CR"].to_numpy()
            d = distances(w)
            eintrag = {
                "n": len(g),
                "cr_mean": cr.mean(),
                "cr_median": float(np.median(cr)),
                "cr_max": cr.max(),
                "cr_share": float((cr > CR_THRESHOLD).mean()),
                "mean": w.mean(axis=0),
                "sd": w.std(axis=0, ddof=1) if len(g) > 1 else np.full(len(CRIT), np.nan),
                "patterns": g[CRIT].round(6).apply(tuple, axis=1).nunique(),
                "dist_median": float(np.median(d)) if d.size else np.nan,
                "weights": w,
            }
            eintrag["sd_mean"] = float(np.mean(eintrag["sd"]))
            kennzahlen[(quelle, case)] = eintrag

            p = f"{key_of(quelle)}.{slug(case)}"
            vals.set(f"n.{p}", str(eintrag["n"]))
            vals.set(f"cr-mean.{p}", num(eintrag["cr_mean"]))
            vals.set(f"cr-median.{p}", num(eintrag["cr_median"]))
            vals.set(f"cr-max.{p}", num(eintrag["cr_max"]))
            vals.set(f"cr-share.{p}", pct(eintrag["cr_share"]))
            vals.set(f"sd-mean.{p}", num(eintrag["sd_mean"]))
            vals.set(f"patterns.{p}", str(eintrag["patterns"]))
            vals.set(f"dist-median.{p}", num(eintrag["dist_median"]))
            for i, c in enumerate(CRIT):
                vals.set(f"w-mean.{p}.{slug(c)}", num(eintrag["mean"][i]))
                vals.set(f"w-sd.{p}.{slug(c)}", num(eintrag["sd"][i]))

    # Ueber alle Faelle zusammengefasst, je Quelle: das sind die Zahlen, die im
    # Flughtext stehen, ohne dass ein einzelner Fall herausgegriffen wird.
    for quelle in sources:
        g = df[df["quelle"] == quelle]
        if g.empty:
            continue
        s = key_of(quelle)
        vals.set(f"n.{s}.total", str(len(g)))
        vals.set(f"runs.{s}", str(g["run"].nunique()))
        vals.set(f"cr-mean.{s}.pooled", num(g["CR"].mean()))
        vals.set(f"cr-share.{s}.pooled", pct((g["CR"] > CR_THRESHOLD).mean()))
        je_fall = [kennzahlen[(quelle, c)] for c in cases if (quelle, c) in kennzahlen]
        for feld, name, fmt in (
            ("cr_mean", "cr-mean", num), ("cr_share", "cr-share", pct),
            ("sd_mean", "sd-mean", num), ("dist_median", "dist-median", num),
        ):
            werte = [e[feld] for e in je_fall]
            vals.set(f"{name}.{s}.min", fmt(min(werte)))
            vals.set(f"{name}.{s}.max", fmt(max(werte)))

    # Spannweite ueber die Modelle: erlaubt Saetze, die unabhaengig davon
    # gelten, welche Modelle in FILES stehen.
    for feld, name, fmt in (
        ("cr_mean", "cr-mean", num), ("cr_share", "cr-share", pct),
        ("sd_mean", "sd-mean", num), ("dist_median", "dist-median", num),
    ):
        werte = [kennzahlen[(q, c)][feld] for q in llm_sources for c in cases
                 if (q, c) in kennzahlen]
        if werte:
            vals.set(f"{name}.models.min", fmt(min(werte)))
            vals.set(f"{name}.models.max", fmt(max(werte)))

    # Abstand zur Ground-Truth, mit demselben Mass wie alle anderen Vergleiche.
    wahrheit: dict[tuple[str, str], dict] = {}
    gt_cases = (gt or {}).get("cases", {})
    for case, eintrag in gt_cases.items():
        ziel = eintrag["weights"]
        for i, c in enumerate(CRIT):
            vals.set(f"gt.{slug(case)}.{slug(c)}", num(ziel[i], 4))
        if eintrag["strategy"]:
            vals.set(f"gt-strategy.{slug(case)}", tex(eintrag["strategy"]))
        for quelle in sources:
            e = kennzahlen.get((quelle, case))
            if e is None:
                continue
            d = np.linalg.norm(e["weights"] - ziel, axis=1)
            treffer = {
                "d_means": float(np.linalg.norm(e["mean"] - ziel)),
                "median": float(np.median(d)),
                "q25": float(np.quantile(d, 0.25)),
                "q75": float(np.quantile(d, 0.75)),
            }
            wahrheit[(quelle, case)] = treffer
            pq = f"{key_of(quelle)}.{slug(case)}"
            vals.set(f"gt-dmeans.{pq}", num(treffer["d_means"]))
            vals.set(f"gt-median.{pq}", num(treffer["median"]))
            vals.set(f"gt-q25.{pq}", num(treffer["q25"]))
            vals.set(f"gt-q75.{pq}", num(treffer["q75"]))

    if wahrheit:
        modellwerte = [e["d_means"] for (q, _), e in wahrheit.items() if q != HUMAN_SOURCE]
        menschwerte = [e["d_means"] for (q, _), e in wahrheit.items() if q == HUMAN_SOURCE]
        if modellwerte:
            vals.set("gt-dmeans.models.min", num(min(modellwerte)))
            vals.set("gt-dmeans.models.max", num(max(modellwerte)))
        if menschwerte:
            vals.set("gt-dmeans.human.min", num(min(menschwerte)))
            vals.set("gt-dmeans.human.max", num(max(menschwerte)))

    anker: dict[tuple[str, str], dict] = {}
    referenz: dict[str, dict] = {}
    if hat_probanden:
        for case in cases:
            human = kennzahlen.get((HUMAN_SOURCE, case))
            if human is None:
                continue
            d_hh = distances(human["weights"])
            referenz[case] = {
                "n": human["n"],
                "median": float(np.median(d_hh)),
                "q25": float(np.quantile(d_hh, 0.25)),
                "q75": float(np.quantile(d_hh, 0.75)),
            }
            p = f"{slug(case)}"
            vals.set(f"hh-median.{p}", num(referenz[case]["median"]))
            vals.set(f"hh-q25.{p}", num(referenz[case]["q25"]))
            vals.set(f"hh-q75.{p}", num(referenz[case]["q75"]))

            for quelle in llm_sources:
                modell = kennzahlen.get((quelle, case))
                if modell is None:
                    continue
                d_lh = distances(modell["weights"], human["weights"])
                eintrag = {
                    "d_means": float(np.linalg.norm(modell["mean"] - human["mean"])),
                    "median": float(np.median(d_lh)),
                    "q25": float(np.quantile(d_lh, 0.25)),
                    "q75": float(np.quantile(d_lh, 0.75)),
                    "diff": modell["mean"] - human["mean"],
                }
                anker[(quelle, case)] = eintrag
                pq = f"{key_of(quelle)}.{slug(case)}"
                vals.set(f"lh-dmeans.{pq}", num(eintrag["d_means"]))
                vals.set(f"lh-median.{pq}", num(eintrag["median"]))
                vals.set(f"lh-q25.{pq}", num(eintrag["q25"]))
                vals.set(f"lh-q75.{pq}", num(eintrag["q75"]))
                for i, c in enumerate(CRIT):
                    vals.set(f"lh-diff.{pq}.{slug(c)}", num(eintrag["diff"][i]))

        if anker:
            vals.set("lh-dmeans.models.min", num(min(e["d_means"] for e in anker.values())))
            vals.set("lh-dmeans.models.max", num(max(e["d_means"] for e in anker.values())))
            vals.set("lh-median.models.min", num(min(e["median"] for e in anker.values())))
            vals.set("lh-median.models.max", num(max(e["median"] for e in anker.values())))
            vals.set("hh-median.cases.min", num(min(r["median"] for r in referenz.values())))
            vals.set("hh-median.cases.max", num(max(r["median"] for r in referenz.values())))

    return {
        "kennzahlen": kennzahlen,
        "anker": anker,
        "referenz": referenz,
        "llm_sources": llm_sources,
        "hat_probanden": hat_probanden,
        "labels": labels,
        "bedingung": bedingung,
        "wahrheit": wahrheit,
        "gt": gt_cases,
    }


# --------------------------------------------------------------------------
# Abschnitte
# --------------------------------------------------------------------------

def basis_section(df: pd.DataFrame, sources: list[str], cases: list[str],
                  daten: dict) -> str:
    kennzahlen, labels = daten["kennzahlen"], daten["labels"]
    block = [
        [tex(labels[q]),
         str(df[df["quelle"] == q]["run"].nunique()),
         *(str(kennzahlen[(q, c)]["n"]) if (q, c) in kennzahlen else "--" for c in cases)]
        for q in sources
    ]
    return table(
        caption="Data basis: runs or participants per source and case. "
                "Only responses that passed schema validation enter the analysis.",
        label="tab:weights-basis",
        spalten="l" + "r" * (len(cases) + 1),
        kopf=["Source", "Runs / participants", *(tex(CASE_LABEL[c]) for c in cases)],
        gruppen=[(None, block)],
    )


def consistency_section(sources: list[str], cases: list[str], daten: dict) -> str:
    kennzahlen, labels = daten["kennzahlen"], daten["labels"]
    gruppen = []
    for case in cases:
        block = []
        for q in sources:
            e = kennzahlen.get((q, case))
            if e is None:
                continue
            block.append([
                tex(labels[q]),
                str(e["n"]),
                num(e["cr_mean"]),
                num(e["cr_median"]),
                num(e["cr_max"]),
                pct(e["cr_share"]),
            ])
        if block:
            gruppen.append((tex(CASE_LABEL[case]), block))
    return table(
        caption=f"Consistency ratio (CR) per case and source. The last column is the "
                f"share of matrices with $CR > {num(CR_THRESHOLD, 2)}$.",
        label="tab:weights-cr",
        spalten="lrrrrr",
        kopf=["Source", "$n$", "Mean", "Median", "Max.",
              f"$CR > {num(CR_THRESHOLD, 2)}$"],
        gruppen=gruppen,
    )


def variance_section(sources: list[str], cases: list[str], daten: dict) -> str:
    kennzahlen, labels = daten["kennzahlen"], daten["labels"]
    gruppen = []
    for case in cases:
        block = []
        for q in sources:
            e = kennzahlen.get((q, case))
            if e is None:
                continue
            block.append([
                tex(labels[q]),
                *(f"{num(e['mean'][i], 3)} ({num(e['sd'][i], 3)})" for i in range(len(CRIT))),
                str(e["patterns"]),
                num(e["dist_median"]),
            ])
        if block:
            gruppen.append((tex(CASE_LABEL[case]), block))
    return table(
        caption="Mean weight per criterion with the standard deviation across runs or "
                "participants in parentheses, the number of distinct weight vectors, and "
                "the median pairwise distance within the source.",
        label="tab:weights-spread",
        spalten="l" + "r" * (len(CRIT) + 2),
        kopf=["Source", *(tex(CRIT_LABEL[c]) for c in CRIT), "Distinct", "Median dist."],
        gruppen=gruppen,
    )


def ground_truth_tables(sources: list[str], cases: list[str], daten: dict) -> tuple[str, str]:
    """Zwei Tabellen: die Zielgewichte selbst und die Abstaende dorthin."""
    gt, wahrheit, labels = daten["gt"], daten["wahrheit"], daten["labels"]

    block = []
    for case in cases:
        if case not in gt:
            continue
        zeile = [tex(CASE_LABEL[case])]
        if any(e["strategy"] for e in gt.values()):
            zeile.append(tex(gt[case]["strategy"]))
        zeile += [num(gt[case]["weights"][i], 4) for i in range(len(CRIT))]
        block.append(zeile)
    kopf = ["Case"] + (["Strategy"] if any(e["strategy"] for e in gt.values()) else [])
    gewichte = table(
        caption="Ground-truth weights the cases were generated from.",
        label="tab:weights-gt",
        spalten="l" * len(kopf) + "r" * len(CRIT),
        kopf=kopf + [tex(CRIT_LABEL[c]) for c in CRIT],
        gruppen=[(None, block)],
    )

    gruppen = []
    for case in cases:
        rows = []
        for q in sources:
            e = wahrheit.get((q, case))
            if e is None:
                continue
            rows.append([
                tex(labels[q]),
                num(e["d_means"]),
                f"{num(e['median'])} [{num(e['q25'])}, {num(e['q75'])}]",
            ])
        if rows:
            gruppen.append((tex(CASE_LABEL[case]), rows))
    abstaende = table(
        caption="Distance to the ground truth. Column two is the distance between the "
                "mean weight vector of the source and the ground-truth vector of that "
                "case. Column three is the median distance of the single weight vectors "
                "to the same target, with the interquartile range in brackets.",
        label="tab:weights-gt-distance",
        spalten="lrr",
        kopf=["Source", "Distance of means", "Single vectors, median [IQR]"],
        gruppen=gruppen,
    )
    return gewichte, abstaende


def anchor_section(cases: list[str], daten: dict) -> str:
    anker, referenz, labels = daten["anker"], daten["referenz"], daten["labels"]
    gruppen = []
    for case in cases:
        if case not in referenz:
            continue
        block = []
        for q in daten["llm_sources"]:
            e = anker.get((q, case))
            if e is None:
                continue
            block.append([
                tex(labels[q]),
                num(e["d_means"]),
                f"{num(e['median'])} [{num(e['q25'])}, {num(e['q75'])}]",
            ])
        r = referenz[case]
        block.append([
            f"{tex(HUMAN_LABEL)} (reference)",
            "--",
            f"{num(r['median'])} [{num(r['q25'])}, {num(r['q75'])}]",
        ])
        gruppen.append((tex(CASE_LABEL[case]), block))
    return table(
        caption="Human anchor. Column two is the distance between the mean weight "
                "vector of the source and the mean weight vector of the participants. "
                "Column three is the median pairwise distance between single weight "
                "vectors with the interquartile range in brackets: model runs against "
                "participants, and, in the reference row, participants against each other.",
        label="tab:weights-anchor",
        spalten="lrr",
        kopf=["Source", "Distance of means", "Pairwise distance, median [IQR]"],
        gruppen=gruppen,
    )


def prose(figure_dir: str, vorhandene: dict[str, str], daten: dict,
          sources: list[str], cases: list[str], df: pd.DataFrame,
          level: str = "section", title: str = TITLE,
          starred_subs: bool = True) -> str:
    """Der Fliesstext. Alle Zahlen kommen aus \\resval, keine steht hier fest."""
    teile: list[str] = []

    teile.append("\\section{" + tex(title) + "}\n\\label{sec:results-weights}\n\n"
                 + r"""This section addresses RQ3 and the weighting part of RQ4. It reports the raw
consistency ratio of the LLM-derived pairwise matrices, the run-to-run variance
of the resulting weights, and the exploratory human anchor: the LLM-derived
weights compared against the human-derived weights, interpreted relative to the
human--human spread. The section is descriptive and confines itself to the
measured values on exactly these measures.""")

    teile.append(r"""\subsection{Data basis and measures}
\label{sec:results-weights-basis}

Each run and each participant supplies \resval{n-criteria} pairwise comparisons
per case, from which a reciprocal $\resval{n-criteria} \times \resval{n-criteria}$
matrix is formed. Weights are derived per matrix with the \resval{weight-method};
the consistency ratio follows Saaty with a random index of \resval{random-index}
for $n = \resval{n-criteria}$. Responses that failed schema validation carry no
weights and are excluded, so the counts in Table~\ref{tab:weights-basis} are the
retained ones. The comparison of two weight vectors throughout this section is
their Euclidean distance, which ranges from $0$ for identical weights to
$\sqrt{2} \approx 1.414$ for the two most dissimilar vectors on the simplex.
Table~\ref{tab:weights-basis} lists the \resval{n-sources} sources and
\resval{n-cases} cases that enter the analysis.""")
    if daten["bedingung"]:
        teile.append(r"""The \resval{n-models} models (\resval{models}) were run in the
\texttt{\resval{condition}} condition, in which all cases of one run are
processed in a single conversation. Table~\ref{tab:weights-basis} and all
following tables therefore name the model alone.""")
    teile.append(basis_section(df, sources, cases, daten))

    teile.append(r"""\subsection{Consistency of the pairwise matrices}
\label{sec:results-weights-cr}

Table~\ref{tab:weights-cr} reports the consistency ratio per case and source, and
Figure~\ref{fig:weights-cr} shows the single values on a logarithmic axis.
Across cases and models, the mean consistency ratio ranges from
\resval{cr-mean.models.min} to \resval{cr-mean.models.max}, and the share of
matrices above $CR > \resval{cr-threshold}$ ranges from
\resval{cr-share.models.min} to \resval{cr-share.models.max}. For the
participants the mean consistency ratio ranges from
\resval{cr-mean.human.min} to \resval{cr-mean.human.max}, with
\resval{cr-share.human.min} to \resval{cr-share.human.max} of the
matrices above the same threshold.""")
    teile.append(consistency_section(sources, cases, daten))
    if "consistency" in vorhandene:
        teile.append(figure(vorhandene["consistency"],
                            "Consistency ratio per case and source. Each point is one run "
                            f"or one participant; the dashed line marks $CR = {num(CR_THRESHOLD, 2)}$.",
                            "fig:weights-cr", figure_dir, "0.8"))

    teile.append(r"""\subsection{Run-to-run variance of the weights}
\label{sec:results-weights-spread}

Table~\ref{tab:weights-spread} reports, per case and source, the mean weight of
each criterion with its standard deviation across runs or participants, the
number of distinct weight vectors, and the median pairwise distance within the
source. Figure~\ref{fig:weights-spread} shows the underlying single values.
Averaged over the \resval{n-criteria} criteria, the standard deviation of the
model weights ranges from \resval{sd-mean.models.min} to
\resval{sd-mean.models.max} across cases and models, against
\resval{sd-mean.human.min} to \resval{sd-mean.human.max} for the
participants. The median distance between two weight vectors of the same source
ranges from \resval{dist-median.models.min} to \resval{dist-median.models.max}
for the models and from \resval{dist-median.human.min} to
\resval{dist-median.human.max} for the participants.""")
    teile.append(variance_section(sources, cases, daten))
    if "weights" in vorhandene:
        teile.append(figure(vorhandene["weights"],
                            "Weights per run and per participant. Each point is one run or "
                            "one participant, the horizontal line is the mean of the source.",
                            "fig:weights-spread", figure_dir, "1.0"))

    if daten["wahrheit"]:
        teile.append(r"""\subsection{Distance to the ground truth}
\label{sec:results-weights-gt}

Each case was generated from a fixed weight vector. Table~\ref{tab:weights-gt}
reproduces these vectors, Table~\ref{tab:weights-gt-distance} reports how far
the elicited weights are from them, measured with the same Euclidean distance.
For the models, the distance between the mean weight vector and the
ground-truth vector of the case ranges from \resval{gt-dmeans.models.min} to
\resval{gt-dmeans.models.max}; for the participants it ranges from
\resval{gt-dmeans.human.min} to \resval{gt-dmeans.human.max}.""")
        gewichte, abstaende = ground_truth_tables(sources, cases, daten)
        teile.append(gewichte)
        teile.append(abstaende)

    if daten["hat_probanden"] and daten["anker"]:
        teile.append(r"""\subsection{Human anchor}
\label{sec:results-weights-anchor}

The anchor is exploratory: it places the model weights next to the weights of
\resval{runs.human} participants (\resval{n.human.total} judgements
across \resval{n-cases} cases) and reads the difference against the spread among
the participants themselves.
Table~\ref{tab:weights-anchor} reports both distances per case. The distance
between the mean weight vector of a model and the mean weight vector of the
participants ranges from \resval{lh-dmeans.models.min} to
\resval{lh-dmeans.models.max}. At the level of single vectors, the median
distance between a model run and a participant ranges from
\resval{lh-median.models.min} to \resval{lh-median.models.max}, while the median
distance between two participants ranges from \resval{hh-median.cases.min} to
\resval{hh-median.cases.max}. Figure~\ref{fig:weights-rankings} shows the same
material as rankings: the share of runs or participants per resulting order of
the \resval{n-criteria} criteria.""")
        teile.append(anchor_section(cases, daten))
        if "rankings" in vorhandene:
            teile.append(figure(vorhandene["rankings"],
                                "Criterion rankings per case and source, as a share of the runs "
                                "or participants of that source.",
                                "fig:weights-rankings", figure_dir, "0.85"))

    # Ueberschriften erst hier auf die gewuenschte Ebene setzen, damit der Text
    # oben lesbar bleibt. Andere \section-Befehle kommen darin nicht vor.
    text = "\n\n".join(teile)
    unter = {"chapter": "section", "section": "subsection",
             "subsection": "subsubsection"}[level]

    if starred_subs:
        # Sternform: ohne Nummer und ohne Eintrag im Inhaltsverzeichnis. Die
        # zugehoerigen \label fallen mit weg, weil sie sonst auf die Nummer des
        # umgebenden Abschnitts zeigen wuerden statt auf die Ueberschrift.
        text = re.sub(r"\\subsection\{([^}]*)\}\n\\label\{[^}]*\}\n",
                      lambda m: "\\UNTER*{" + m.group(1) + "}\n", text)

    # Ueber einen Platzhalter, sonst trifft die zweite Ersetzung das Ergebnis
    # der ersten (\subsection -> \section -> \chapter).
    text = text.replace("\\subsection{", "\\UNTER{").replace("\\section{", f"\\{level}{{")
    return text.replace("\\UNTER", f"\\{unter}")


# --------------------------------------------------------------------------

def zip_for_overleaf(out: Path, figure_dir: str, namen: dict[str, str],
                     figure_ext: str, ziel: Path, tex_dir: str) -> None:
    r"""Packt Section und Abbildungen so, wie sie im Projekt liegen sollen.

    Das Archiv bildet genau die Struktur ab, die `graphics_path()` in die
    \includegraphics-Befehle schreibt: Section in `tex_dir`, Abbildungen in
    `tex_dir/figure_dir`.
    """
    with zipfile.ZipFile(ziel, "w", zipfile.ZIP_DEFLATED) as archiv:
        archiv.write(out, join_path(tex_dir, out.name))
        for name in namen.values():
            datei = out.parent / figure_dir / f"{name}.{figure_ext}"
            archiv.write(datei, join_path(tex_dir, figure_dir, datei.name))
    print(f"Archiv fuer Overleaf -> {ziel}")


def build(runs: Path, out: Path, figure_dir: str, figure_ext: str,
          level: str, tex_dir: str, zip_ziel: Path | None, title: str,
          starred_subs: bool, ground_truth: Path) -> None:
    df, sources, cases = load_runs(runs)
    gt = load_ground_truth(ground_truth, cases)
    vals = Values()
    daten = collect(df, sources, cases, vals, gt)

    vorhandene = {}
    for schluessel, (name, _) in FIGURES.items():
        pfad = (out.parent / figure_dir / f"{name}.{figure_ext}")
        if pfad.is_file():
            vorhandene[schluessel] = name
        else:
            print(f"Abbildung fehlt, wird ausgelassen: {pfad}")

    kopf = [
        f"% Erzeugt von {Path(__file__).name} aus {runs.name} "
        f"({len(df)} Zeilen, {len(sources)} Quellen). Nicht von Hand aendern.",
        "% Benoetigt im Praeambel-Teil: \\usepackage{graphicx} und \\usepackage{booktabs}.",
        f"% Erwartet diese Datei im Projekt unter {join_path(tex_dir, out.name)} "
        f"und die Abbildungen unter {join_path(tex_dir, figure_dir)}/, weil LaTeX die "
        "Abbildungspfade",
        "% gegenueber der Hauptdatei aufloest, nicht gegenueber dieser Datei.",
        "",
        MACRO_PREAMBLE,
        "",
        f"% {len(vals)} Werte aus den Daten.",
        *vals.lines(),
        "",
        prose(graphics_path(tex_dir, figure_dir), vorhandene, daten,
              sources, cases, df, level, title, starred_subs),
        "",
    ]
    out.write_text("\n".join(kopf), encoding="utf-8")
    print(f"{len(vals)} Werte, {len(vorhandene)} Abbildungen -> {out}")

    if zip_ziel:
        zip_for_overleaf(out, figure_dir, vorhandene, figure_ext, zip_ziel, tex_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_results_tex.py",
        description="Schreibt den Ergebnisteil zu RQ3/RQ4 als LaTeX-Datei.",
    )
    parser.add_argument("--runs", type=Path, default=HERE / "runs.csv",
                        help="Export aus auswertung.ipynb (Default runs.csv)")
    parser.add_argument("--out", type=Path, default=HERE / "results_rq3.tex",
                        help="Zieldatei (Default results_rq3.tex)")
    parser.add_argument("--figure-dir", default="figures",
                        help="Ordner der Abbildungen neben der Zieldatei (Default figures)")
    parser.add_argument("--tex-dir", default="results",
                        help="Ordner, in dem diese Datei im LaTeX-Projekt liegt, relativ "
                             "zur Hauptdatei. Steht in den Abbildungspfaden und im Archiv. "
                             "Leer, wenn sie neben der Hauptdatei liegt (Default results)")
    parser.add_argument("--figure-ext", default="pdf",
                        help="Dateiendung der Abbildungen (Default pdf)")
    parser.add_argument("--zip", type=Path, dest="zip_ziel", nargs="?",
                        const=HERE / "results_rq3_overleaf.zip",
                        help="Zusaetzlich ein Archiv zum Hochladen in Overleaf schreiben "
                             "(ohne Pfadangabe results_rq3_overleaf.zip)")
    parser.add_argument("--ground-truth", type=Path, default=HERE / "ground_truth.yaml",
                        help="Zielgewichte der Fallgenerierung (Default ground_truth.yaml). "
                             "Fehlt die Datei, entfaellt der Abschnitt dazu.")
    parser.add_argument("--title", default=TITLE,
                        help=f"Ueberschrift des Abschnitts (Default: {TITLE!r})")
    parser.add_argument("--numbered-subs", action="store_true",
                        help="Unterueberschriften nummeriert und im Inhaltsverzeichnis. "
                             "Ohne die Option stehen sie in der Sternform.")
    parser.add_argument("--level", choices=["chapter", "section", "subsection"],
                        default="section",
                        help="Ebene der obersten Ueberschrift (Default section)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if not args.runs.is_file():
        raise SystemExit(f"{args.runs} fehlt. Zuerst auswertung.ipynb laufen lassen.")
    build(args.runs, args.out, args.figure_dir, args.figure_ext, args.level,
          args.tex_dir, args.zip_ziel, args.title, not args.numbered_subs,
          args.ground_truth)
