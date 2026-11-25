"""Wiederholungsexperiment zu Schritt 1 "Aussagen suchen".

Führt die Suche aus relevance.py n-mal über dieselbe Passagendatei aus, mit
demselben Prompt und denselben Einstellungen, und zählt je Passage, wie oft sie
ausgewählt wurde. So zeigt sich, welche Passagen zuverlässig gefunden werden
(gesucht), welche fälschlich hineinrutschen (und wie oft) und wie stabil die
Entscheidung über die Läufe ist.

    python3 src/relevance_experiment.py --dry-run --case batteriepacks --runs 20 --passages data/passages_hidden.csv
    python3 src/relevance_experiment.py --model gpt-4o-2024-08-06 --case batteriepacks --runs 20 --passages data/passages_hidden.csv
    python3 src/relevance_experiment.py --report results/relevance_runs/<datei>.json

Getrennt von relevance.py: Das Ergebnis ist eine Auswertung, keine Eingabe für
die Zuordnung, und landet in results/relevance_runs/. Prompt, Schema und die
Wiederholung bei ungültiger Antwort sind dieselben wie in relevance.py.

Gruppen der Auswertung (aus Ground Truth und der optionalen CSV-Spalte kind):
  gesucht                  Aussage zu diesem Produkt (soll rein)
  <kind>                   z. B. near_miss, neighbor, off_topic (soll nicht rein)
  <kind> (anderes Produkt) dasselbe, aber aus dem Dokument eines anderen Produkts
Eine gescheiterte Passage (failed) zählt als nicht ausgewählt.

Fehlerarten werden getrennt ausgewiesen, weil sie verschieden schwer wiegen:
  falsch negativ (übersehen)   kritisch: die Aussage erreicht Gate 1 nie und
                               fällt dort höchstens als "Kriterium ohne
                               Evidenz" auf
  falsch positiv (zu viel)     unkritisch: der Prüfer löscht sie in Gate 1
Als Gesamtmaß dient deshalb F2 (Recall viermal so stark gewichtet), nicht F1.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from agent import LLMClient, build_client
from assignment import MAX_ATTEMPTS, PASSAGES_FILE, STATUS_FAILED, sha256
from experiment import ZEITFORMAT_DATEI, _schreibe_json, setup_logger
from relevance import PROMPT_FILE, evaluate, ground_truth, prepare, screen
from runner import RESULTS_DIR, model_slug

OUT_DIR = RESULTS_DIR / "relevance_runs"
GESUCHT = "gesucht"
STATEMENT = "statement"
ANDERES = " (anderes Produkt)"
# So viele Passagen in Folge, die nur an Client-Fehlern scheitern (Netz, Kontingent),
# brechen das Experiment ab, statt tausende leere Aufrufe zu zählen.
MAX_CLIENT_FEHLER_IN_FOLGE = 5
F_BETA = 2.0  # Recall zählt mehr als Precision (falsch negativ ist schlimmer)


# ------------------------------------------------------------- Gruppen


def kinds(passages_path: Path) -> dict[str, str]:
    """Optionale Spalte kind der Passagendatei (z. B. statement, near_miss)."""
    with Path(passages_path).open(encoding="utf-8-sig", newline="") as handle:
        return {
            (row.get("passage_id") or "").strip(): (row.get("kind") or "").strip()
            for row in csv.DictReader(handle)
        }


def gruppe(passage: dict[str, Any], case_id: str, kind: str) -> str | None:
    """Auswertungsgruppe einer Passage; None ohne Ground Truth."""
    if passage["expected_criteria"] is None:
        return None
    eigen = passage["doc_id"] == case_id
    if passage["expected_criteria"]:
        return GESUCHT if eigen else STATEMENT + ANDERES
    name = kind if kind and kind != STATEMENT else "sonstige"
    return name if eigen else name + ANDERES


# ---------------------------------------------------------- Auswertung


def auswerten(dokument: dict[str, Any]) -> dict[str, Any]:
    """Je Lauf Precision/Recall, je Passage die Auswahlhäufigkeit, je Gruppe die Quote.

    Zählt nur abgeschlossene Läufe (alle Passagen beantwortet oder failed)."""
    passagen = dokument["passages"]
    n_passagen = len(passagen)
    truth = dokument.get("ground_truth") or {}
    je_lauf: dict[int, list[dict[str, Any]]] = {}
    for r in dokument["records"]:
        je_lauf.setdefault(r["run"], []).append(r)
    fertig = sorted(run for run, lauf in je_lauf.items() if len(lauf) == n_passagen)

    laeufe = []
    for run in fertig:
        e = evaluate({"records": je_lauf[run], "ground_truth": truth}) or {}
        laeufe.append({
            "run": run,
            "found": sum(r["relevant"] is True for r in je_lauf[run]),
            "failed": sum(r["status"] == STATUS_FAILED for r in je_lauf[run]),
            **{k: e.get(k) for k in ("precision", "recall", "tp", "fp", "fn", "tn", "missed")},
            "f2": _f_beta(e.get("tp"), e.get("fp"), e.get("fn"), F_BETA) if e else None,
        })

    gewaehlt = Counter(r["passage_id"] for run in fertig for r in je_lauf[run] if r["relevant"] is True)
    gescheitert = Counter(r["passage_id"] for run in fertig for r in je_lauf[run] if r["status"] == STATUS_FAILED)
    n = len(fertig)
    je_passage = []
    for p in passagen:
        k = gewaehlt[p["passage_id"]]
        je_passage.append({
            "passage_id": p["passage_id"],
            "doc_id": p["doc_id"],
            "group": p["group"],
            "should_select": truth.get(p["passage_id"]),
            "selected": k,
            "failed": gescheitert[p["passage_id"]],
            "rate": k / n if n else None,
            "stable": n > 0 and k in (0, n),
        })

    gruppen = []
    for name in _gruppen_reihenfolge({p["group"] for p in je_passage}):
        mitglieder = [p for p in je_passage if p["group"] == name]
        auswahlen = sum(p["selected"] for p in mitglieder)
        gruppen.append({
            "group": name,
            "should_select": mitglieder[0]["should_select"],
            "passages": len(mitglieder),
            "selections": auswahlen,
            "rate": auswahlen / (len(mitglieder) * n) if n else None,
            "never": sum(p["selected"] == 0 for p in mitglieder),
            "always": sum(n > 0 and p["selected"] == n for p in mitglieder),
        })

    def kennzahl(feld: str) -> dict[str, float] | None:
        werte = [lauf[feld] for lauf in laeufe if lauf[feld] is not None]
        return {"mean": mean(werte), "min": min(werte), "max": max(werte)} if werte else None

    return {
        "runs_completed": n,
        # Falsch negativ (übersehen) wiegt schwerer als falsch positiv: eine
        # übersehene Aussage erreicht Gate 1 nie, eine zu viel ausgewählte wird
        # dort gelöscht. Deshalb getrennt ausgewiesen und F2 statt F1.
        "false_negatives": {
            "total": sum(lauf["fn"] or 0 for lauf in laeufe),
            "runs_with_any": sum(bool(lauf["fn"]) for lauf in laeufe),
            "per_run": kennzahl("fn"),
            "passages_ever_missed": sorted({pid for lauf in laeufe for pid in lauf["missed"] or []}),
        },
        "false_positives": {
            "total": sum(lauf["fp"] or 0 for lauf in laeufe),
            "runs_with_any": sum(bool(lauf["fp"]) for lauf in laeufe),
            "per_run": kennzahl("fp"),
            "passages_ever_selected": sorted(p["passage_id"] for p in je_passage
                                             if p["should_select"] is False and p["selected"]),
        },
        "precision": kennzahl("precision"),
        "recall": kennzahl("recall"),
        "f2": kennzahl("f2"),
        "found_per_run": kennzahl("found"),
        "stable_passages": sum(p["stable"] for p in je_passage),
        "identical_runs": n > 0 and len({tuple(r["relevant"] for r in je_lauf[run]) for run in fertig}) == 1,
        "groups": gruppen,
        "passages": je_passage,
        "runs": laeufe,
    }


def _f_beta(tp: int | None, fp: int | None, fn: int | None, beta: float) -> float | None:
    """F-beta; beta = 2 gewichtet den Recall viermal so stark wie die Precision."""
    if tp is None:
        return None
    nenner = (1 + beta**2) * tp + beta**2 * (fn or 0) + (fp or 0)
    return (1 + beta**2) * tp / nenner if nenner else None


def _gruppen_reihenfolge(namen: set[str | None]) -> list[str | None]:
    """gesucht zuerst, dann die eigenen Gruppen, dann die anderer Produkte."""
    return sorted(namen, key=lambda g: (g != GESUCHT, g is None, (g or "").endswith(ANDERES), g or ""))


def format_report(dokument: dict[str, Any]) -> str:
    """Bericht in zwei getrennten Blöcken: falsch negativ (kritisch) zuerst,
    falsch positiv (unkritisch) danach."""
    meta, a = dokument["meta"], dokument["evaluation"]
    n = a["runs_completed"]
    texte = {p["passage_id"]: p["text"] for p in dokument["passages"]}
    zeilen = [
        f"Aussagen suchen, {n} von {meta['runs']} Läufen | Produkt {meta['product']} | "
        f"{len(dokument['passages'])} Passagen ({meta['passages_file']}) | "
        f"{meta['model_requested']}, Temperatur {meta['temperature']}",
    ]
    if meta.get("aborted"):
        zeilen.append(f"ABGEBROCHEN: {meta['aborted']}")
    if not n:
        return "\n".join(zeilen + ["Kein Lauf abgeschlossen."])
    passagen = a["passages"]
    fn, fp = a["false_negatives"], a["false_positives"]

    def kennzahl(name: str, k: dict[str, float] | None, format_: str = ".2f") -> str:
        if not k:
            return f"  {name:<25}-"
        return (f"  {name:<25}Mittel {k['mean']:{format_}}  "
                f"(min {k['min']:{format_}}, max {k['max']:{format_}})")

    def zeile(p: dict[str, Any], text: str) -> str:
        fehl = f", davon {p['failed']}× failed" if p["failed"] else ""
        return f"  {p['passage_id']:<6}{text}{fehl}  [{p['group']}] {_kurz(texte[p['passage_id']])}"

    gesucht = [p for p in passagen if p["should_select"] is True]
    zeilen += [
        "",
        "FALSCH NEGATIV – übersehen (KRITISCH: die Aussage erreicht Gate 1 nie)",
        f"  {'Übersehen gesamt':<25}{fn['total']} in {n} Läufen; "
        f"Läufe mit mindestens einer übersehenen Aussage: {fn['runs_with_any']} von {n}",
        kennzahl("Übersehen je Lauf", fn["per_run"]),
        kennzahl("Recall", a["recall"]),
        "  Gesuchte Passagen (gefunden / übersehen):",
    ]
    if not gesucht:
        zeilen.append("  keine")
    for p in sorted(gesucht, key=lambda p: (p["selected"], p["passage_id"])):
        verfehlt = n - p["selected"]
        markierung = "  <-- !" if verfehlt else ""
        zeilen.append(zeile(p, f"{p['selected']:>3}/{n} gefunden, {verfehlt:>2}× übersehen{markierung}"))

    falsch = sorted((p for p in passagen if p["should_select"] is False and p["selected"]),
                    key=lambda p: (-p["selected"], p["passage_id"]))
    zeilen += [
        "",
        "FALSCH POSITIV – zu viel ausgewählt (unkritisch: wird in Gate 1 gelöscht)",
        f"  {'Zu viel gesamt':<25}{fp['total']} in {n} Läufen; "
        f"Läufe mit mindestens einer: {fp['runs_with_any']} von {n}",
        kennzahl("Zu viel je Lauf", fp["per_run"]),
        kennzahl("Precision", a["precision"]),
        f"  Passagen, mindestens einmal fälschlich ausgewählt ({len(falsch)}):",
    ]
    if not falsch:
        zeilen.append("  keine")
    zeilen += [zeile(p, f"{p['selected']:>3}/{n} ausgewählt") for p in falsch]
    ohne = [p for p in passagen if p["should_select"] is None and p["selected"]]
    if ohne:
        zeilen += ["", "  Ausgewählt, ohne Ground Truth:"] + [zeile(p, f"{p['selected']:>3}/{n} ausgewählt") for p in ohne]

    zeilen += [
        "",
        "GESAMT",
        kennzahl("F2 (Recall > Precision)", a["f2"]),
        kennzahl("Ausgewählt je Lauf", a["found_per_run"]),
        f"  {'Stabil':<25}{a['stable_passages']} von {len(passagen)} Passagen immer gleich entschieden"
        + (" (alle Läufe identisch)" if a["identical_runs"] else ""),
        "",
        f"  {'Gruppe':<32}{'Fehlerart':<11}{'Pass.':>6}{'Fehlerquote':>13}{'nie gewählt':>13}{'immer':>7}",
    ]
    for g in a["groups"]:
        art = {True: "FN (krit.)", False: "FP", None: "-"}[g["should_select"]]
        quote = g["rate"] if g["rate"] is None or g["should_select"] is not True else 1 - g["rate"]
        zeilen.append(f"  {g['group'] or 'ohne Ground Truth':<32}{art:<11}{g['passages']:>6}"
                      f"{_zahl(quote):>13}{g['never']:>13}{g['always']:>7}")
    zeilen.append("  Fehlerquote: bei gesucht Anteil übersehen, sonst Anteil fälschlich ausgewählt.")
    return "\n".join(zeilen)


def _kurz(text: str, laenge: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= laenge else text[: laenge - 1] + "…"


def _zahl(wert: float | None) -> str:
    return "-" if wert is None else f"{wert:.2f}"


# ------------------------------------------------------------ Durchlauf


def run_experiment(
    client: LLMClient,
    *,
    case_id: str,
    runs: int,
    passages_path: Path = PASSAGES_FILE,
    prompt_path: Path = PROMPT_FILE,
    temperature: float = 0.0,
    out_dir: Path = OUT_DIR,
    quiet: bool = False,
) -> tuple[Path, Path]:
    """n Suchläufe über alle Passagen. Gibt (JSON, Log) zurück; die JSON wird nach
    jeder Passage geschrieben, ein Abbruch vernichtet also nichts."""
    if runs < 1:
        raise ValueError("runs muss mindestens 1 sein")
    eingabe = prepare(passages_path, case_id, prompt_path)
    art = kinds(passages_path)
    start = datetime.now(timezone.utc)
    run_id = (f"{start.astimezone().strftime(ZEITFORMAT_DATEI)}_{client.provider}_"
              f"{model_slug(client.model_id)}_{case_id}_n{runs}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, log_path = out_dir / f"{run_id}.json", out_dir / f"{run_id}.log"

    meta = {
        "step": "relevance_experiment",
        "started": start.isoformat(timespec="seconds"),
        "finished": None,
        "aborted": None,
        "provider": client.provider,
        "model_requested": client.model_id,
        "temperature": temperature,
        "runs": runs,
        "case_id": case_id,
        "product": eingabe["product"],
        "prompt_version": prompt_path.stem,
        "prompt_sha256": sha256(prompt_path),
        "passages_file": Path(passages_path).name,
        "passages_sha256": sha256(passages_path),
        "max_attempts": MAX_ATTEMPTS,
        "output_schema": eingabe["schema"],
    }
    dokument: dict[str, Any] = {
        "meta": meta,
        "prompts": {"system": eingabe["system"], "user": eingabe["user"]},
        "passages": [
            {
                "passage_id": p["passage_id"],
                "doc_id": p["doc_id"],
                "position": p["position"],
                "kind": art.get(p["passage_id"]) or None,
                "group": gruppe(p, case_id, art.get(p["passage_id"], "")),
                "text": p["text"],
            }
            for p in eingabe["passages"]
        ],
        "ground_truth": ground_truth(eingabe["passages"], case_id),
        "records": [],
    }

    logger = setup_logger(log_path, quiet)
    client_fehler_in_folge = 0
    try:
        logger.info("Experiment Aussagen suchen %s | provider=%s model=%s temperature=%s runs=%d produkt=%s "
                    "passagen=%d (%s) | %d Aufrufe ohne Wiederholungen",
                    meta["started"], client.provider, client.model_id, temperature, runs, eingabe["product"],
                    len(eingabe["passages"]), meta["passages_file"], runs * len(eingabe["passages"]))
        logger.debug("---- SYSTEM-PROMPT ----\n%s", eingabe["system"])
        for run in range(1, runs + 1):
            for passage in eingabe["passages"]:
                pid = passage["passage_id"]
                ergebnis = screen(client, eingabe["system"], eingabe["user"][pid], pid, temperature)
                letzter = ergebnis["calls"][-1]
                record = {
                    "run": run,
                    "passage_id": pid,
                    "relevant": ergebnis["relevant"],
                    "justification": ergebnis["justification"],
                    "status": ergebnis["status"],
                    "attempts": len(ergebnis["calls"]),
                    "model_returned": letzter["model_returned"],
                    "errors": letzter["errors"],
                }
                if ergebnis["status"] == STATUS_FAILED:
                    record["calls"] = ergebnis["calls"]  # Rohantworten nur, wo es nötig ist
                dokument["records"].append(record)
                _schreibe_json(json_path, dokument)
                logger.debug("[Lauf %d/%d | %s] %s", run, runs, pid,
                             {True: "relevant", False: "nicht relevant", None: "FAILED"}[record["relevant"]])

                nur_client = ergebnis["status"] == STATUS_FAILED and all(
                    f.startswith("Client-Fehler") for c in ergebnis["calls"] for f in c["errors"])
                client_fehler_in_folge = client_fehler_in_folge + 1 if nur_client else 0
                if client_fehler_in_folge >= MAX_CLIENT_FEHLER_IN_FOLGE:
                    meta["aborted"] = (f"Abbruch in Lauf {run}: {client_fehler_in_folge} Passagen in Folge "
                                       f"mit Client-Fehler, zuletzt {letzter['errors'][-1]}")
                    logger.error(meta["aborted"])
                    break
            else:
                gefunden = sum(r["relevant"] is True for r in dokument["records"] if r["run"] == run)
                logger.info("Lauf %d/%d: %d ausgewählt", run, runs, gefunden)
                continue
            break

        if not meta["aborted"]:
            meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        dokument["evaluation"] = auswerten(dokument)
        _schreibe_json(json_path, dokument)
        logger.info("\n%s", format_report(dokument))
        logger.info("Ergebnisse: %s", json_path)
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
    return json_path, log_path


def report_file(path: Path) -> str:
    """Bericht zu einer vorhandenen Datei, auch zu einer abgebrochenen."""
    dokument = json.loads(Path(path).read_text(encoding="utf-8"))
    if dokument.get("meta", {}).get("step") != "relevance_experiment":
        raise ValueError(f"{path}: keine Datei von relevance_experiment.py")
    dokument["evaluation"] = auswerten(dokument)
    return format_report(dokument)


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="relevance_experiment.py", description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=["openai", "anthropic", "mistral"], default="openai")
    parser.add_argument("--model", help="Modell-ID (außer bei --dry-run und --report Pflicht)")
    parser.add_argument("--case", help="Fall (doc_id); das Produkt kommt aus cases.yaml")
    parser.add_argument("--runs", type=int, default=20, help="Wiederholungen über alle Passagen (Standard 20)")
    parser.add_argument("--passages", type=Path, default=PASSAGES_FILE, help="CSV mit den Passagen")
    parser.add_argument("--prompt", type=Path, default=PROMPT_FILE, help="Versionierte Prompt-Vorlage")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Nur Umfang anzeigen, kein Modellaufruf")
    parser.add_argument("--report", type=Path, help="Bericht zu einer vorhandenen Ergebnisdatei ausgeben")
    args = parser.parse_args(argv)

    if args.report:
        print(report_file(args.report))
        return 0
    if not args.case:
        parser.error("--case ist erforderlich")
    if args.runs < 1:
        parser.error("--runs muss mindestens 1 sein")
    if args.dry_run:
        eingabe = prepare(args.passages, args.case, args.prompt)
        art = kinds(args.passages)
        gruppen = Counter(gruppe(p, args.case, art.get(p["passage_id"], "")) for p in eingabe["passages"])
        print(f"Produkt {eingabe['product']}, {len(eingabe['passages'])} Passagen, {args.runs} Läufe "
              f"= {args.runs * len(eingabe['passages'])} Aufrufe (ohne Wiederholungen)")
        for name in _gruppen_reihenfolge(set(gruppen)):
            print(f"  {name or 'ohne Ground Truth':<34}{gruppen[name]:>4}")
        return 0
    if not args.model:
        parser.error("--model ist erforderlich (außer bei --dry-run und --report)")

    json_path, _ = run_experiment(build_client(args.provider, args.model), case_id=args.case, runs=args.runs,
                                  passages_path=args.passages, prompt_path=args.prompt,
                                  temperature=args.temperature, out_dir=args.out_dir, quiet=args.quiet)
    if not args.quiet:
        print(f"\nBericht erneut: python3 src/relevance_experiment.py --report {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
