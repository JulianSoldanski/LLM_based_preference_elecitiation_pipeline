"""Erzeugt den Ergebnisteil (RQ3 und Gewichtungsteil RQ4) als LaTeX-Datei.

Gerechnet wird in `auswertung.ipynb`: dort entstehen Gewichte, CR und die
Abbildungen. Dieses Skript liest nur das exportierte `runs.csv` und schreibt
daraus eine \\input-faehige Section. Jede Zahl im Text steht als Makro, jede
Tabelle und jede Abbildung wird aus den Daten erzeugt; nichts ist abgetippt.

    python3 build_results_tex.py [--runs runs.csv] [--out results_rq3.tex]

Der Text ist bewusst rein deskriptiv: er nennt die gemessenen Werte und
vergleicht sie, bewertet sie aber nicht.

Aufbau des Abschnitts: Stichprobe und Modelllaeufe; RQ3 mit der Stabilitaet der
Gewichte und der Konsistenz der Matrizen; Gewichtungsteil von RQ4 mit dem
Abstand zu den Zielgewichten und zu den Probanden. Drei Tabellen, zwei
Abbildungen.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from clean_interviews import (ERSTE_TEILNAHME, MIN_SEKUNDEN,  # noqa: E402
                              fehlende_vergleiche, zu_schnelle_seiten)

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

# Ueberschrift des Abschnitts; die Nummer davor vergibt LaTeX. Der Abschnitt
# deckt RQ3 und den Gewichtungsteil von RQ4 ab, daher kein "RQ3" im Titel.
TITLE = "Weights: Stability, Consistency and Agreement"

# Rechenverfahren aus dem Notebook; steht hier nur fuer den Methodensatz im
# Text und muss mit AHP_METHOD in auswertung.ipynb uebereinstimmen.
WEIGHT_METHOD = "row geometric mean"
RANDOM_INDEX = "0.58"
CR_THRESHOLD = 0.1
# "Most runs" im Text heisst: mehr als dieser Anteil der Laeufe.
MAJORITY = 0.5
# Temperatur aller Modellaufrufe; steht hier nur fuer den Satz im Text und muss
# mit den API-Aufrufen uebereinstimmen.
TEMPERATURE = "0"
# Erklaerung der Modellbedingung fuer den Satz zu den Modelllaeufen. Eine
# Bedingung ohne Eintrag wird nur genannt, nicht erklaert.
CONDITION_TEXT = {
    "stateless": "each case was processed in a separate conversation, "
                 "without the preceding cases",
}
# Bezeichnung der Zielgewichte aus ground_truth.yaml in Text und Tabellen. Sie
# sind das, was die Generierung treffen sollte, keine gepruefte Wahrheit. Die
# Saetze sind so gebaut, dass Einzahl ("attempted ground truth") und Mehrzahl
# ("assumed target weights") beide passen.
TARGET_TERM = "assumed target weights"
TARGET_LABEL = TARGET_TERM[:1].upper() + TARGET_TERM[1:]
# Labels im Kapitel, auf die der Text verweist: Aufbau der Befragung und
# Ablauf der Modellaufrufe.
SURVEY_SECTION = "sec:survey-design"
LLM_SECTION = "subsec:parallel-llm"

# SoSci-Rohexport fuer die Stichprobenbeschreibung, mit allen Eintraegen. Die
# Auslese wendet die Regeln aus src/clean_interviews.py an. Fehlt die Datei,
# entfaellt der Abschnitt dazu.
SURVEY = HERE.parent / "results" / "InterviewResults" / (
    "data_ahpsupplierselectionjs23_2026-09-19_12-33.csv")
CONSENT_REFUSED = "1"                                        # SC01: Nein, nicht teilnehmen
# Antwortcodes aus dem SoSci-Fragebogen, uebersetzt nach
# results/InterviewResults/values_ahpsupplierselectionjs23_2026-09-24_16-57.csv.
# Ein Code ohne Eintrag bricht den Lauf ab.
EXPERIENCE_LABEL = {1: "Yes", 2: "No"}                       # SD20 Einkaufserfahrung
PROGRAMME_LABEL = {                                          # SD21 Studiengang
    1: "Business informatics",                               # Wirtschaftsinformatik
    2: "Business and economics",                             # Wirtschaftswissenschaften
    3: "Industrial engineering and management",              # Wirtschaftsingenieurwesen
    4: "Other",
}
OTHER_CODE = 4                                               # "Andere" mit Freitext SD21_04
# Freitext zu "Andere" (SD21_04), kleingeschrieben -> Studiengang. Traegt der
# Studiengang denselben Namen wie ein Code oben, werden beide zusammengezaehlt.
OTHER_PROGRAMME = {"bwl": "Business and economics"}
# Was danach noch unter "Andere" steht, zaehlt als dieser Studiengang. Das
# Skript nennt jeden so zugeordneten Freitext, damit die Zuordnung pruefbar
# bleibt.
OTHER_FALLBACK = "Industrial engineering and management"
# Verweildauer der drei Fallseiten (SoSci-Variablen, randomisierte Reihenfolge).
CASE_PAGES = ["TIME004", "TIME005", "TIME006"]

# Abbildungen aus dem Notebook. Fehlende Dateien werden uebersprungen.
# 02_konsistenz wird nicht mehr eingebunden: Tabelle tab:weights-cr traegt die
# Aussage, und bei den Modellen liegen die Punkte eines Falls fast alle
# aufeinander.
FIGURES = {
    "weights": ("01_gewichte", "Weights per run and per participant"),
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


def aufzaehlen(teile: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    if len(teile) < 2:
        return "".join(teile)
    return ", ".join(teile[:-1]) + " and " + teile[-1]


def von_bis(schluessel: str, werte: list) -> str:
    """'between \\resval{x.min} and \\resval{x.max}'; bei gleichen Werten nur eine Zahl."""
    if min(werte) == max(werte):
        return f"\\resval{{{schluessel}.max}}"
    return f"between \\resval{{{schluessel}.min}} and \\resval{{{schluessel}.max}}"


def absatz(saetze: list[str]) -> str:
    """Setzt erzeugte Saetze zu einem Absatz zusammen und bricht ihn fuer die
    .tex-Datei um. Am gesetzten Text aendert der Umbruch nichts."""
    text = " ".join(" ".join(s.split()) for s in saetze if s)
    return textwrap.fill(text, width=84, break_long_words=False, break_on_hyphens=False)


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


def load_survey(path: Path) -> pd.DataFrame | None:
    """Liest den SoSci-Rohexport als Text; die zweite Zeile sind Beschriftungen.

    Als Text, weil die Regeln aus clean_interviews.py auf Zeichenketten pruefen.
    """
    if not path.is_file():
        print(f"Ohne Stichprobenabschnitt, Datei fehlt: {path}")
        return None
    roh = pd.read_csv(path, sep=";", skiprows=[1], encoding="utf-8-sig",
                      dtype=str, keep_default_na=False)
    return roh[roh["CASE"].str.strip().str.isdigit()]


def reihenfolge(werte: pd.Series, namen) -> list[tuple[str, int]]:
    """Haeufigkeiten in der Reihenfolge des Fragebogens, nicht nach Groesse."""
    anzahl = werte.value_counts()
    return [(name, int(anzahl.get(name, 0))) for name in dict.fromkeys(namen)]


def zahl(spalte: pd.Series) -> pd.Series:
    return pd.to_numeric(spalte.str.strip(), errors="coerce")


def auslese(roh: pd.DataFrame) -> tuple[list[tuple[str, int]], dict[str, pd.DataFrame]]:
    """Weg von allen Eintraegen zu den ausgewerteten Teilnahmen.

    Dieselben Regeln wie clean_interviews.py, in derselben Reihenfolge; ein
    Ausschluss wird nur beim ersten zutreffenden Grund gezaehlt.
    """
    ab = roh[zahl(roh["CASE"]) >= ERSTE_TEILNAHME]
    verweigert = ab["SC01"].str.strip() == CONSENT_REFUSED
    abgeschlossen = ab[~verweigert & (ab["FINISHED"].str.strip() == "1")]
    zeilen = abgeschlossen.to_dict("records")
    fehlend = [bool(fehlende_vergleiche(z)) for z in zeilen]
    zu_schnell = [not f and bool(zu_schnelle_seiten(z)) for z, f in zip(zeilen, fehlend)]
    behalten = abgeschlossen[[not f and not z for f, z in zip(fehlend, zu_schnell)]]
    weg = [
        ("started", len(ab)),
        ("refused", int(verweigert.sum())),
        ("aborted", len(ab) - int(verweigert.sum()) - len(abgeschlossen)),
        ("completed", len(abgeschlossen)),
        ("missing", sum(fehlend)),
        ("fast", sum(zu_schnell)),
        ("retained", len(behalten)),
    ]
    return weg, {"completed": abgeschlossen, "retained": behalten}


def gruppe(survey: pd.DataFrame, name: str, vals: Values) -> dict:
    """Erfahrung, Studiengang und Bearbeitungszeit einer Gruppe, zugleich als Makros."""
    n = len(survey)

    def verteilung(codes: pd.Series, namen: dict[int, str], variable: str) -> pd.Series:
        codes = zahl(codes)
        unbekannt = set(codes.dropna().astype(int)) - set(namen)
        if unbekannt or codes.isna().any():
            raise SystemExit(f"{variable}: fehlend oder ohne Bedeutung: {sorted(unbekannt)}")
        return codes.astype(int).map(namen)

    erfahrung = reihenfolge(verteilung(survey["SD20"], EXPERIENCE_LABEL, "SD20"),
                            EXPERIENCE_LABEL.values())
    for label, anzahl in erfahrung:
        vals.set(f"survey-{name}-experience.{slug(label)}", str(anzahl))
        vals.set(f"survey-{name}-experience.{slug(label)}.share", pct(anzahl / n))

    # Studiengang: bekannter Freitext zuerst, der Rest von "Andere" danach als
    # OTHER_FALLBACK. "Andere" kommt damit im Text nicht mehr vor.
    anders = survey["SD21_04"].str.strip().str.lower()
    studiengang = verteilung(survey["SD21"], PROGRAMME_LABEL, "SD21")
    studiengang = studiengang.where(~anders.isin(OTHER_PROGRAMME), anders.map(OTHER_PROGRAMME))
    rest = studiengang == PROGRAMME_LABEL[OTHER_CODE]
    for freitext in anders[rest]:
        print(f"{name}: SD21 'Andere' mit Freitext {freitext!r} -> {OTHER_FALLBACK}")
    studiengang = studiengang.where(~rest, OTHER_FALLBACK)
    programme = reihenfolge(studiengang, [
        *(v for k, v in PROGRAMME_LABEL.items() if k != OTHER_CODE),
        *OTHER_PROGRAMME.values(), OTHER_FALLBACK])
    for label, anzahl in programme:
        vals.set(f"survey-{name}-programme.{slug(label)}", str(anzahl))
        vals.set(f"survey-{name}-programme.{slug(label)}.share", pct(anzahl / n))

    # Bearbeitungszeit: nur die drei Fallseiten zusammen, in Minuten. Median und
    # Quartile, weil einzelne Seitenzeiten stark nach rechts ausreissen.
    minuten = sum(zahl(survey[spalte]) for spalte in CASE_PAGES) / 60
    q1, median, q3 = minuten.quantile([0.25, 0.5, 0.75])
    for k, v in (("median", median), ("q1", q1), ("q3", q3)):
        vals.set(f"survey-time.{name}.{k}", num(v, 1))

    return {"n": n, "erfahrung": erfahrung, "programme": programme}


def sample(roh: pd.DataFrame, vals: Values, human_runs: set[int]) -> dict:
    """Weg der Auslese und beide Gruppen: alle Abgeschlossenen, die Ausgewerteten."""
    weg, gruppen = auslese(roh)
    for schritt, anzahl in weg:
        vals.set(f"survey-n-{schritt}", str(anzahl))
    vals.set("survey-min-seconds", str(MIN_SEKUNDEN))

    # Die ausgewerteten Teilnahmen muessen genau die Probanden in runs.csv sein,
    # sonst beschreibt dieser Abschnitt eine andere Stichprobe als der Rest.
    behalten = set(zahl(gruppen["retained"]["CASE"]).astype(int))
    if human_runs and behalten != human_runs:
        raise SystemExit("Stichprobe passt nicht zu runs.csv: nur im Export "
                         f"{sorted(behalten - human_runs)}, nur in runs.csv "
                         f"{sorted(human_runs - behalten)}. clean_interviews.py, "
                         "transform_interviews.py und auswertung.ipynb neu laufen lassen.")
    return {"weg": dict(weg), **{name: gruppe(g, name, vals) for name, g in gruppen.items()}}


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
    vals.set("temperature", TEMPERATURE)

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
    # Fliesstext stehen, ohne dass ein einzelner Fall herausgegriffen wird.
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
            ("patterns", "patterns", str),
        ):
            werte = [e[feld] for e in je_fall]
            vals.set(f"{name}.{s}.min", fmt(min(werte)))
            vals.set(f"{name}.{s}.max", fmt(max(werte)))

    # Laeufe und gueltige Antworten je Modell und Fall, fuer den Satz zu den
    # Modelllaeufen. Sind alle gleich, nennt der Text nur eine Zahl.
    runs_je_modell = [df[df["quelle"] == q]["run"].nunique() for q in llm_sources]
    n_je_modell_fall = [kennzahlen[(q, c)]["n"] for q in llm_sources for c in cases
                        if (q, c) in kennzahlen]
    if runs_je_modell:
        vals.set("runs.models.min", str(min(runs_je_modell)))
        vals.set("runs.models.max", str(max(runs_je_modell)))
    if n_je_modell_fall:
        vals.set("n-valid.models.min", str(min(n_je_modell_fall)))
        vals.set("n-valid.models.max", str(max(n_je_modell_fall)))

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
    einig_anders: list[str] = []
    for case, eintrag in gt_cases.items():
        ziel = eintrag["weights"]
        for i, c in enumerate(CRIT):
            vals.set(f"gt.{slug(case)}.{slug(c)}", num(ziel[i], 4))
        if eintrag["strategy"]:
            vals.set(f"gt-strategy.{slug(case)}", tex(eintrag["strategy"]))

        # Wie viele Quellen setzen das Kriterium mit dem hoechsten Zielgewicht
        # im Mittel auch auf Rang eins?
        oben = int(np.argmax(ziel))
        oben_je_quelle = [int(np.argmax(kennzahlen[(q, case)]["mean"]))
                          for q in sources if (q, case) in kennzahlen]
        vals.set(f"gt-top.{slug(case)}", CRIT_LABEL[CRIT[oben]].lower())
        vals.set(f"gt-top-hits.{slug(case)}", str(sum(o == oben for o in oben_je_quelle)))
        # Setzen alle Quellen dasselbe andere Kriterium auf Rang eins, nennt
        # der Text es.
        if len(set(oben_je_quelle)) == 1 and oben_je_quelle[0] != oben:
            vals.set(f"gt-top-instead.{slug(case)}",
                     CRIT_LABEL[CRIT[oben_je_quelle[0]]].lower())
            einig_anders.append(case)

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

    naeher = None
    if wahrheit:
        for wer, ist_mensch in (("models", False), ("human", True)):
            eintraege = [e for (q, _), e in wahrheit.items() if (q == HUMAN_SOURCE) == ist_mensch]
            if not eintraege:
                continue
            for feld, name in (("d_means", "gt-dmeans"), ("median", "gt-median")):
                vals.set(f"{name}.{wer}.min", num(min(e[feld] for e in eintraege)))
                vals.set(f"{name}.{wer}.max", num(max(e[feld] for e in eintraege)))

        # In wie vielen Faellen liegt der Probandenmittelwert naeher an den
        # Zielgewichten als der Mittelwert jedes Modells?
        n_naeher = n_verglichen = 0
        for case in gt_cases:
            mensch = wahrheit.get((HUMAN_SOURCE, case))
            modelle = [wahrheit[(q, case)]["d_means"] for q in llm_sources
                       if (q, case) in wahrheit]
            if mensch is None or not modelle:
                continue
            n_verglichen += 1
            n_naeher += mensch["d_means"] < min(modelle)
        if n_verglichen:
            vals.set("gt-human-closer", str(n_naeher))
            vals.set("gt-n-compared", str(n_verglichen))
            naeher = (n_naeher, n_verglichen)

    anker: dict[tuple[str, str], dict] = {}
    referenz: dict[str, dict] = {}
    lh_unter: dict[str, int] = {}
    lh_ueber: dict[str, int] = {}
    lh_vorne = 0
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

            # Referenz fuer den Abstand Modellmittel -> Probandenmittel: Abstand
            # eines Probanden zum Mittel der uebrigen Probanden (leave one out).
            # Der Paarvergleich Proband-Proband bevorzugt dagegen jeden Vektor,
            # der nahe der Mitte liegt.
            w_h = human["weights"]
            if len(w_h) > 1:
                uebrige = (w_h.sum(axis=0) - w_h) / (len(w_h) - 1)
                d_hc = np.linalg.norm(w_h - uebrige, axis=1)
                referenz[case]["hc_median"] = float(np.median(d_hc))
                vals.set(f"hc-median.{p}", num(referenz[case]["hc_median"]))

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
            hc = [r["hc_median"] for r in referenz.values() if "hc_median" in r]
            if hc:
                vals.set("hc-median.cases.min", num(min(hc)))
                vals.set("hc-median.cases.max", num(max(hc)))

            # Richtung der Unterschiede je Kriterium, ueber alle Kombinationen
            # aus Modell und Fall gezaehlt.
            vals.set("lh-n-combinations", str(len(anker)))
            for i, c in enumerate(CRIT):
                lh_unter[c] = int(sum(e["diff"][i] < 0 for e in anker.values()))
                lh_ueber[c] = int(sum(e["diff"][i] > 0 for e in anker.values()))
                vals.set(f"lh-below.{slug(c)}", str(lh_unter[c]))
                vals.set(f"lh-above.{slug(c)}", str(lh_ueber[c]))
            lh_vorne = int(sum(
                e["diff"][int(np.argmax(kennzahlen[(HUMAN_SOURCE, case)]["mean"]))] > 0
                for (_, case), e in anker.items()))
            vals.set("lh-lead-above", str(lh_vorne))

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
        "gt_einig_anders": einig_anders,
        "gt_naeher": naeher,
        "hc_da": any("hc_median" in r for r in referenz.values()),
        "lh_unter": lh_unter,
        "lh_ueber": lh_ueber,
        "lh_vorne": lh_vorne,
        "n_gleich": len(set(n_je_modell_fall)) <= 1,
    }


# --------------------------------------------------------------------------
# Tabellen
# --------------------------------------------------------------------------

def variance_section(sources: list[str], cases: list[str], daten: dict) -> str:
    kennzahlen, labels, gt = daten["kennzahlen"], daten["labels"], daten["gt"]
    gruppen = []
    for case in cases:
        block = []
        # Zielgewichte als erste Zeile, damit jede Quelle direkt daran gemessen
        # werden kann. Streuung und Abstand gibt es fuer sie nicht.
        if case in gt:
            block.append([tex(TARGET_LABEL),
                          *(num(gt[case]["weights"][i], 3) for i in range(len(CRIT))),
                          "--", "--"])
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
    zielzeile = (f" The first row of each case gives the {tex(TARGET_TERM)}." if gt else "")
    return table(
        caption="Mean weight per criterion with the standard deviation across runs or "
                "participants in parentheses, the number of distinct weight vectors, and "
                "the median pairwise distance within the source." + zielzeile,
        label="tab:weights-spread",
        spalten="l" + "r" * (len(CRIT) + 2),
        kopf=["Source", *(tex(CRIT_LABEL[c]) for c in CRIT), "Distinct", "Median dist."],
        gruppen=gruppen,
    )


def consistency_section(sources: list[str], cases: list[str], daten: dict) -> str:
    """CR je Fall und Quelle. Ohne n und Distinct: n steht bei den Modelllaeufen,
    Distinct in tab:weights-spread. Ohne Median: bei den Modellen ist er fast
    immer gleich dem Mittelwert."""
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
                num(e["cr_mean"]),
                num(e["cr_max"]),
                pct(e["cr_share"]),
            ])
        if block:
            gruppen.append((tex(CASE_LABEL[case]), block))
    return table(
        caption="Consistency ratio (CR) per case and source: mean and maximum over the "
                "runs or participants, and the share of matrices with "
                f"$CR > {num(CR_THRESHOLD, 2)}$.",
        label="tab:weights-cr",
        spalten="lrrr",
        kopf=["Source", "Mean", "Max.", f"$CR > {num(CR_THRESHOLD, 2)}$"],
        gruppen=gruppen,
    )


def agreement_section(sources: list[str], cases: list[str], daten: dict) -> str:
    """Eine Tabelle fuer RQ4: Abstand zu den Zielgewichten und zu den Probanden.

    Ersetzt die frueheren Tabellen tab:weights-gt-distance, tab:weights-anchor
    und tab:weights-diff. Die Probandenzeile steht wie in den anderen Tabellen
    oben; ihre letzte Spalte ist die Referenz: Probanden untereinander.
    """
    wahrheit, anker, referenz = daten["wahrheit"], daten["anker"], daten["referenz"]
    labels = daten["labels"]
    mit_ziel, mit_anker = bool(wahrheit), bool(anker)
    if not (mit_ziel or mit_anker):
        return ""

    def spanne(e: dict) -> str:
        return f"{num(e['median'])} [{num(e['q25'])}, {num(e['q75'])}]"

    gruppen = []
    for case in cases:
        block = []
        for q in sources:
            if (q, case) not in daten["kennzahlen"]:
                continue
            zeile = [tex(labels[q])]
            if mit_ziel:
                e = wahrheit.get((q, case))
                zeile.append(num(e["d_means"]) if e else "--")
            if mit_anker:
                if q == HUMAN_SOURCE:
                    r = referenz.get(case)
                    zeile += ["--", spanne(r) if r else "--"]
                else:
                    e = anker.get((q, case))
                    zeile += [num(e["d_means"]), spanne(e)] if e else ["--", "--"]
            block.append(zeile)
        if block:
            gruppen.append((tex(CASE_LABEL[case]), block))

    kopf, ziele, erklaerung = ["Source"], [], []
    if mit_ziel:
        kopf.append("To target")
        ziele.append(f"the {tex(TARGET_TERM)}")
        erklaerung.append("\\emph{To target} is the distance between the mean weight vector "
                          f"of the source and the {tex(TARGET_TERM)}.")
    if mit_anker:
        kopf += ["To participant mean", "To single participants, median [IQR]"]
        ziele.append("the participants")
        erklaerung.append("\\emph{To participant mean} is the distance between the mean "
                          "weight vector of the model and the mean weight vector of the "
                          "participants. \\emph{To single participants} is the median "
                          "distance between single weight vectors, with the interquartile "
                          "range in brackets: model runs against participants and, in the "
                          "participant row, participants against each other.")
    return table(
        caption="Distance of the weights to " + aufzaehlen(ziele) + ". " + " ".join(erklaerung),
        label="tab:weights-agreement",
        spalten="l" + "r" * (len(kopf) - 1),
        kopf=kopf,
        gruppen=gruppen,
    )


# --------------------------------------------------------------------------
# Fliesstext
# --------------------------------------------------------------------------

def prose(figure_dir: str, vorhandene: dict[str, str], daten: dict,
          sources: list[str], cases: list[str], df: pd.DataFrame,
          level: str = "section", title: str = TITLE,
          starred_subs: bool = True) -> str:
    """Der Fliesstext. Alle Zahlen kommen aus \\resval, keine steht hier fest.

    Reihenfolge: Stichprobe, Modelllaeufe, RQ3 (Stabilitaet, Konsistenz),
    RQ4 (Zielgewichte, Probanden). Saetze, die ein Muster beschreiben ("in
    allen Laeufen", "in den meisten Laeufen"), erscheinen nur, wenn die Daten
    das Muster tatsaechlich zeigen.
    """
    k, labels, llm = daten["kennzahlen"], daten["labels"], daten["llm_sources"]
    begriff = tex(TARGET_TERM)
    mit_probanden = any((HUMAN_SOURCE, c) in k for c in cases)

    def fall(case: str) -> str:
        return tex(CASE_LABEL[case].lower())

    def name(quelle: str) -> str:
        return tex(labels[quelle])

    def krit(i: int) -> str:
        return tex(CRIT_LABEL[CRIT[i]].lower())

    teile: list[str] = []

    # 0. Einleitung
    vergleiche = ([f"the {begriff}"] if daten["wahrheit"] else []) + (
        ["the weights of the participants"] if daten["anker"] else [])
    teile.append("\\section{" + tex(title) + "}\n\\label{sec:results-weights}\n\n" + absatz([
        "This section addresses RQ3 and the weighting part of RQ4. It first describes the "
        "two sources of judgements, the participants and the models. It then answers RQ3: "
        "how stable the weights are across runs and how consistent the pairwise matrices are.",
        ("Finally, it answers the weighting part of RQ4 by comparing the weights with "
         + aufzaehlen(vergleiche) + ".") if vergleiche else "",
        "The section is descriptive and confines itself to the measured values.",
    ]))

    # 1. Befragung
    if daten.get("stichprobe"):
        stichprobe = daten["stichprobe"]
        ausschluss = (r"\resval{survey-n-fast} spent less than \resval{survey-min-seconds}\,s"
                      " on at least one case page")
        if stichprobe["weg"]["missing"]:
            ausschluss = r"\resval{survey-n-missing} lacked a pairwise comparison and " + ausschluss
        studium = [
            f"\\resval{{survey-retained-programme.{slug(label)}}} "
            f"(\\resval{{survey-retained-programme.{slug(label)}.share}})"
            + (" study " if i == 0 else " ") + tex(label[:1].lower() + label[1:])
            for i, (label, _) in enumerate(
                p for p in stichprobe["retained"]["programme"] if p[1])]
        teile.append("\\subsection{Participant sample}\n"
                     "\\label{sec:results-weights-sample}\n\n" + absatz([
            r"The participants rated the \resval{n-cases} cases in the online survey described"
            f" in Section~\\ref{{{SURVEY_SECTION}}}.",
            r"Of \resval{survey-n-started} recorded questionnaires, \resval{survey-n-refused}"
            r" refused consent and \resval{survey-n-aborted} were aborted before the last"
            r" page, which leaves \resval{survey-n-completed} completed questionnaires.",
            "Of these, " + ausschluss + r" and were excluded, so \resval{survey-n-retained}"
            " participants enter the analysis.",
            ("Among the retained participants, " + aufzaehlen(studium) + ".") if studium else "",
            r"Prior work experience in procurement was reported by"
            r" \resval{survey-retained-experience.yes}"
            r" (\resval{survey-retained-experience.yes.share}).",
            r"The retained participants spent a median of"
            r" \resval{survey-time.retained.median}\,min on the three case pages together"
            r" (interquartile range"
            r" \resval{survey-time.retained.q1}--\resval{survey-time.retained.q3}\,min).",
        ]))

    # 2. Modelllaeufe
    if llm:
        n_valid = (r"\resval{n-valid.models.min}" if daten["n_gleich"]
                   else r"between \resval{n-valid.models.min} and \resval{n-valid.models.max}")
        bedingung = ""
        if daten["bedingung"]:
            erklaerung = CONDITION_TEXT.get(daten["bedingung"])
            bedingung = (r"They ran in the \texttt{\resval{condition}} condition"
                         + (": " + erklaerung if erklaerung else "") + ".")
        teile.append("\\subsection{Model runs}\n"
                     "\\label{sec:results-weights-models}\n\n" + absatz([
            r"The \resval{n-models} models (\resval{models}) rated the same \resval{n-cases}"
            f" cases, following the procedure in Section~\\ref{{{LLM_SECTION}}}.",
            r"All calls used a temperature of \resval{temperature}.",
            bedingung,
            "The analysis uses " + n_valid + " valid runs per model and case.",
        ]))

    # 3. RQ3: Stabilitaet der Gewichte
    teile.append("\\subsection{Stability of the weights}\n"
                 "\\label{sec:results-weights-spread}\n\n" + absatz([
        r"Each participant and each model run supplies \resval{n-criteria} pairwise"
        r" comparisons per case. They form a reciprocal"
        r" $\resval{n-criteria} \times \resval{n-criteria}$ matrix, from which the weights"
        r" are derived with the \resval{weight-method}.",
        r"Two weight vectors are compared by their Euclidean distance. It ranges from $0$"
        r" for identical weights to $\sqrt{2} \approx 1.414$ for the two most dissimilar"
        r" vectors on the simplex.",
    ]) + "\n\n" + absatz([
        r"Table~\ref{tab:weights-spread} reports, per case and source, the mean weight of"
        r" each criterion with its standard deviation across runs or participants, the"
        r" number of distinct weight vectors, and the median pairwise distance within the"
        r" source.",
        r"Figure~\ref{fig:weights-spread} shows the underlying single values."
        if "weights" in vorhandene else "",
        r"In the text, the standard deviation is averaged over the \resval{n-criteria}"
        r" criteria.",
    ]))

    saetze = []
    for q in llm:
        je_fall = [k[(q, c)] for c in cases if (q, c) in k]
        if not je_fall:
            continue
        s = key_of(q)
        muster = [e["patterns"] for e in je_fall]
        if max(muster) == 1:
            saetze.append(f"For {name(q)}, all runs of a case give the same weight vector.")
        else:
            saetze.append(
                f"For {name(q)}, the runs of a case give {von_bis(f'patterns.{s}', muster)}"
                f" distinct weight vectors, with a standard deviation of at most"
                f" \\resval{{sd-mean.{s}.max}}.")
    # Groesste Streuung unter den Modellen: welcher Fall, welche Kriterien.
    streuend = [(q, c) for q in llm for c in cases if (q, c) in k and k[(q, c)]["patterns"] > 1]
    if streuend:
        q, c = max(streuend, key=lambda t: k[t]["sd_mean"])
        e, p = k[(q, c)], f"{key_of(q)}.{slug(c)}"
        i1, i2 = (int(i) for i in np.argsort(-e["sd"])[:2])
        saetze.append(
            f"The largest spread among the models occurs for {name(q)} on the {fall(c)}"
            f" (\\resval{{sd-mean.{p}}}). There, the standard deviation is largest for"
            f" {krit(i1)} (\\resval{{w-sd.{p}.{slug(CRIT[i1])}}}) and {krit(i2)}"
            f" (\\resval{{w-sd.{p}.{slug(CRIT[i2])}}}), and the median distance between two"
            f" runs is \\resval{{dist-median.{p}}}.")
    if saetze:
        teile.append(absatz(saetze))

    if mit_probanden:
        muster = [k[(HUMAN_SOURCE, c)]["patterns"] for c in cases if (HUMAN_SOURCE, c) in k]
        h = HUMAN_KEY
        teile.append(absatz([
            f"The \\resval{{runs.{h}}} participants give {von_bis(f'patterns.{h}', muster)}"
            " distinct weight vectors per case.",
            f"Their standard deviation ranges from \\resval{{sd-mean.{h}.min}} to"
            f" \\resval{{sd-mean.{h}.max}}, and the median distance between two participants"
            f" from \\resval{{dist-median.{h}.min}} to \\resval{{dist-median.{h}.max}}"
            + (r", against at most \resval{dist-median.models.max} between two runs of the"
               " same model." if llm else "."),
        ]))

    teile.append(variance_section(sources, cases, daten))
    if "weights" in vorhandene:
        teile.append(figure(vorhandene["weights"],
                            "Weights per run and per participant. Each point is one run or "
                            "one participant, the horizontal line is the mean of the source.",
                            "fig:weights-spread", figure_dir, "1.0"))

    # 4. RQ3: Konsistenz der Matrizen
    saetze = [
        r"The consistency ratio (CR) follows Saaty, with a random index of"
        r" \resval{random-index} for $n = \resval{n-criteria}$.",
        r"Table~\ref{tab:weights-cr} reports, per case and source, the mean and the maximum"
        r" CR and the share of matrices with $CR > \resval{cr-threshold}$.",
    ]
    if mit_probanden:
        h = HUMAN_KEY
        saetze.append(
            f"For the participants, the mean CR ranges from \\resval{{cr-mean.{h}.min}} to"
            f" \\resval{{cr-mean.{h}.max}}, and \\resval{{cr-share.{h}.min}} to"
            f" \\resval{{cr-share.{h}.max}} of the matrices lie above the threshold.")
    if llm:
        saetze.append(
            r"For the models, the mean CR ranges from \resval{cr-mean.models.min} to"
            r" \resval{cr-mean.models.max}, and the share of matrices above the threshold"
            r" from \resval{cr-share.models.min} to \resval{cr-share.models.max}.")
        anteile = {q: [k[(q, c)]["cr_share"] for c in cases if (q, c) in k] for q in llm}
        anteile = [a for a in anteile.values() if a]
        if anteile and all(min(a) < MAJORITY < max(a) for a in anteile):
            saetze.append("Each model stays below the threshold in most runs of at least one"
                          " case and lies above it in most runs of at least one other case.")
        ueber = [c for c in cases
                 if all((q, c) in k and k[(q, c)]["cr_share"] > MAJORITY for q in llm)]
        if ueber:
            saetze.append("For " + aufzaehlen([f"the {fall(c)}" for c in ueber])
                          + r", all \resval{n-models} models lie above the threshold in most"
                            " runs.")
        # Stabil, aber inkonsistent: ein einziger Gewichtsvektor, und jeder Lauf
        # liegt ueber der Schwelle.
        stabil = []
        for q in llm:
            for c in cases:
                e = k.get((q, c))
                if e is None or e["patterns"] != 1 or e["cr_share"] < 1:
                    continue
                p = f"{key_of(q)}.{slug(c)}"
                wert = ("of" if np.isclose(e["cr_mean"], e["cr_max"]) else "up to")
                stabil.append(f"{name(q)} on the {fall(c)} (CR {wert} \\resval{{cr-max.{p}}})")
        if stabil:
            saetze.append("A stable result is not always a consistent one. For "
                          + aufzaehlen(stabil) + ", all runs give the same weight vector,"
                          " and every run lies above the threshold.")
    teile.append("\\subsection{Consistency of the pairwise matrices}\n"
                 "\\label{sec:results-weights-cr}\n\n" + absatz(saetze))
    teile.append(consistency_section(sources, cases, daten))

    # 5. RQ4: Zielgewichte und Probanden
    if daten["wahrheit"] or daten["anker"]:
        ziele = ([f"the {begriff}"] if daten["wahrheit"] else []) + (
            ["the participants"] if daten["anker"] else [])
        absaetze = []

        if daten["wahrheit"]:
            rang_eins = aufzaehlen([
                f"\\resval{{gt-top-hits.{slug(c)}}} of \\resval{{n-sources}} sources for the"
                f" {fall(c)} (\\resval{{gt-top.{slug(c)}}})"
                for c in cases if c in daten["gt"]])
            saetze = [
                f"Each case was generated from a fixed weight vector, its {begriff}, that is,"
                " the weights its statements were meant to express.",
                f"Table~\\ref{{tab:weights-spread}} shows the {begriff} in the first row of"
                " each case, and Table~\\ref{tab:weights-agreement} reports the distances to"
                " them.",
                f"The criterion that ranks first in the {begriff} is also ranked first by the"
                f" mean weights of {rang_eins}.",
            ]
            for c in daten["gt_einig_anders"]:
                saetze.append(f"For the {fall(c)}, all \\resval{{n-sources}} sources rank"
                              f" \\resval{{gt-top-instead.{slug(c)}}} first instead.")
            if daten["gt_naeher"]:
                n_naeher, n_verglichen = daten["gt_naeher"]
                menge = (r"In all \resval{gt-n-compared} cases" if n_naeher == n_verglichen
                         else r"In \resval{gt-human-closer} of the \resval{gt-n-compared} cases")
                saetze.append(menge + ", the mean weight vector of the participants is closer"
                              f" to the {begriff} than the mean weight vector of every model.")
                saetze.append(
                    f"For single weight vectors, the median distance to the {begriff} ranges"
                    r" from \resval{gt-median.human.min} to \resval{gt-median.human.max} for"
                    r" the participants and from \resval{gt-median.models.min} to"
                    r" \resval{gt-median.models.max} for the model runs.")
            absaetze.append(absatz(saetze))

        if daten["anker"]:
            saetze = [
                r"The comparison with the participants is exploratory: it places the model"
                r" weights next to the weights of \resval{runs.human} participants.",
                r"The distance between the mean weight vector of a model and the mean weight"
                r" vector of the participants ranges from \resval{lh-dmeans.models.min} to"
                r" \resval{lh-dmeans.models.max}.",
            ]
            if daten["hc_da"]:
                saetze.append(
                    r"For reference, the median distance between a single participant and the"
                    r" mean weight vector of the other participants ranges from"
                    r" \resval{hc-median.cases.min} to \resval{hc-median.cases.max}.")
            saetze.append(
                r"At the level of single vectors, the median distance between a model run and"
                r" a participant ranges from \resval{lh-median.models.min} to"
                r" \resval{lh-median.models.max}, and between two participants from"
                r" \resval{hh-median.cases.min} to \resval{hh-median.cases.max}.")
            saetze.append(
                "A vector near the centre of the participants tends to be closer to a single"
                " participant than two participants are to each other, so this comparison"
                " favours central vectors.")
            absaetze.append(absatz(saetze))

            # Unterschiede je Kriterium: nur noch als Satz, die Werte ergeben sich
            # aus tab:weights-spread.
            n_komb = len(daten["anker"])
            vorne = (r"all \resval{lh-n-combinations}" if daten["lh_vorne"] == n_komb
                     else r"\resval{lh-lead-above} of the \resval{lh-n-combinations}")
            saetze = [
                r"Split by criterion (Table~\ref{tab:weights-spread}), the models give a"
                r" higher weight than the participants to the criterion that the participants"
                f" rank first in {vorne} combinations of model and case.",
            ]
            for c in CRIT:
                for zaehler, richtung in ((daten["lh_unter"], "lower"),
                                          (daten["lh_ueber"], "higher")):
                    if zaehler.get(c) == n_komb:
                        saetze.append(f"{tex(CRIT_LABEL[c])} receives a {richtung} weight from"
                                      " the models than from the participants in every"
                                      " combination.")
            if "rankings" in vorhandene:
                saetze.append(
                    r"Figure~\ref{fig:weights-rankings} shows the same material as rankings:"
                    r" the share of runs or participants per resulting order of the"
                    r" \resval{n-criteria} criteria.")
            absaetze.append(absatz(saetze))

        teile.append("\\subsection{Agreement with " + aufzaehlen(ziele) + "}\n"
                     "\\label{sec:results-weights-agreement}\n\n" + "\n\n".join(absaetze))
        teile.append(agreement_section(sources, cases, daten))
        if daten["anker"] and "rankings" in vorhandene:
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
          starred_subs: bool, ground_truth: Path, survey: Path = SURVEY) -> None:
    df, sources, cases = load_runs(runs)
    gt = load_ground_truth(ground_truth, cases)
    vals = Values()
    daten = collect(df, sources, cases, vals, gt)
    befragung = load_survey(survey)
    probanden = set(df.loc[df["quelle"] == HUMAN_SOURCE, "run"].astype(int))
    daten["stichprobe"] = sample(befragung, vals, probanden) if befragung is not None else None

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
    parser.add_argument("--survey", type=Path, default=SURVEY,
                        help="SoSci-Rohexport fuer Stichprobe, Auslese und Bearbeitungszeiten. "
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
          args.ground_truth, args.survey)