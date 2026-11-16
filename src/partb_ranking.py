"""Teil B: Kennzahlen je Kriterium und Ranking der Lieferanten (TOPSIS).

    python3 src/partb_ranking.py --component batteriepacks             # nur Kennzahlen
    python3 src/partb_ranking.py results/artifact/<lauf>_r1_<fall>_weights_v1.json
    python3 src/partb_ranking.py --options …                            # mit Pufferlager (Spec 08)

Eine Kennzahl je Kriterium (drei Kriterien ohne Subkriterien), gerechnet auf
den Rohdaten aus data/partb/ über die ganze Historie:

* Liefertreue (SCOR RL.1.2 Perfect Supplier Order): Anteil der Lieferungen,
  die pünktlich, vollständig, mit korrekten Dokumenten und ohne Befund kommen
  (RL.2.5 bis RL.2.8). Nutzen: höher ist besser.
* Flexibilität (SCOR Agility, Adaptability): Anteil der Änderungsanfragen, die
  vollständig (100 %) und bis zum gewünschten Termin umgesetzt wurden. Nutzen.
* Kosten (SCOR Costs): Gesamtkosten je geliefertem Stück aus Preis laut
  Bestellung, Fracht, internen Prozesskosten (Bestellung, Wareneingang,
  Rechnungsprüfung je Kanal, Dokumentenkorrektur) und Bearbeitungskosten der
  Reklamationen. Kosten: niedriger ist besser.

Aggregiert wird mit TOPSIS und Vektornormierung. Min-Max-Normierung läuft als
Sensitivitätsprüfung mit. Es gibt nur ein Ranking, keine Mengenaufteilung.

Mit dem Schalter options (Spec 08) kommt je Zeile aus buffer_options.csv der
Lieferant ein zweites Mal als <supplier_id>+LAGER in die Matrix: dieselben
Gewichte, andere Fakten (Regel in kennzahlen_option). Ohne Schalter bleibt
das Ranking unverändert.

Die Gewichte kommen ausschließlich aus einer fixen Gewichtsversion
(weight_version.load_version, Schritt 8). Das Ergebnis wird eingefroren und
mit Hash im Ereignislog des Laufs festgehalten.

Nur Standardbibliothek.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import ledger
import partb_data
import weight_version
from gate1 import GateError

# Kriterium -> Richtung. Reihenfolge = Spalten der Entscheidungsmatrix.
KRITERIEN = {"liefertreue": "benefit", "flexibilitaet": "benefit", "kosten": "cost"}
KENNZAHLEN = {
    "liefertreue": "Perfect Supplier Order (SCOR RL.1.2): Anteil Lieferungen pünktlich, vollständig, "
                   "Dokumente korrekt, ohne Befund",
    "flexibilitaet": "Adaptability (SCOR Agility): Anteil Änderungsanfragen vollständig und bis zum "
                     "gewünschten Termin umgesetzt",
    "kosten": "Gesamtkosten je geliefertem Stück (SCOR Costs): Preis, Fracht, Prozesskosten, Reklamationen",
}
NORMIERUNGEN = ("vector", "minmax")
EVENT = "SupplierRankingComputed"
OPTION_SUFFIX = "+LAGER"  # ID der Lager-Option: <supplier_id>+LAGER


class RankingError(GateError):
    """Gewichte, Kriterien oder Daten passen nicht zum Ranking."""


# ------------------------------------------------------------- Kennzahlen


def _satz(rates: list[dict[str, Any]], aktivitaet: str, kanal: str) -> float:
    for r in rates:
        if r["activity"] == aktivitaet and r["channel"] in (kanal, "any"):
            return float(r["cost_eur"])
    raise RankingError(f"Kein Prozesskostensatz für {aktivitaet} ({kanal})")


def kostenaufstellung(lieferant: dict[str, Any], rates: list[dict[str, Any]]) -> dict[str, float]:
    """Summen je Kostenart über die Historie und Kosten je geliefertem Stück."""
    orders, kanal = lieferant["orders"], lieferant["order_channel"]
    stueck = sum(o["qty_received"] for o in orders)
    if not stueck:
        raise RankingError(f"{lieferant['supplier_id']}: keine gelieferten Stück, Kosten je Stück undefiniert")
    je_bestellung = _satz(rates, "po_processing", kanal) + _satz(rates, "goods_receipt", kanal) \
        + _satz(rates, "invoice_check", kanal)
    summen = {
        "material": sum(o["qty_received"] * o["unit_price_eur"] for o in orders),
        "freight": sum(o["freight_eur"] for o in orders),
        "process": len(orders) * je_bestellung
                   + sum(not o["docs"] for o in orders) * _satz(rates, "doc_correction", kanal),
        "claims": sum(c["handling_cost_eur"] for c in lieferant["claims"]),
    }
    return {**summen, "total": sum(summen.values()), "units": stueck, "per_unit": sum(summen.values()) / stueck}


def _perfekt(o: dict[str, Any]) -> bool:
    return o["on_time"] and o["in_full"] and o["docs"] and o["condition"]


def _umgesetzt(c: dict[str, Any]) -> bool:
    return c["fulfilled_pct"] == 100 and c["fulfilled_date"] is not None and c["fulfilled_date"] <= c["requested_by_date"]


def kennzahlen(lieferant: dict[str, Any], rates: list[dict[str, Any]]) -> dict[str, Any]:
    """Die drei Kennzahlen eines Lieferanten, dazu Zähler und Nenner."""
    orders, changes = lieferant["orders"], lieferant["changes"]
    if not orders:
        raise RankingError(f"{lieferant['supplier_id']}: keine Bestellungen")
    perfekt = sum(map(_perfekt, orders))
    umgesetzt = sum(map(_umgesetzt, changes))
    kosten = kostenaufstellung(lieferant, rates)
    return {
        "liefertreue": perfekt / len(orders),
        # Ohne Änderungsanfragen ist Flexibilität nicht beobachtet: kein Wert statt 0.
        "flexibilitaet": umgesetzt / len(changes) if changes else None,
        "kosten": kosten["per_unit"],
        "details": {
            "liefertreue": {"perfect_orders": perfekt, "orders": len(orders)},
            "flexibilitaet": {"fulfilled_in_time": umgesetzt, "change_requests": len(changes)},
            "kosten": kosten,
        },
    }


# ------------------------------------------------- Pufferlager (Spec 08)


def optionen_pruefen(partb: dict[str, Any]) -> list[dict[str, Any]]:
    """Zeilen aus buffer_options.csv, geprüft gegen die bekannten Lieferanten."""
    if "buffer_options" not in partb:
        raise RankingError(f"buffer_options.csv fehlt in {partb.get('dir')}")
    bekannt = {s["supplier_id"] for k in partb["components"].values() for s in k["suppliers"]}
    for zeile in partb["buffer_options"]:
        sid = zeile.get("supplier_id")
        if sid not in bekannt:
            raise RankingError(f"buffer_options.csv: Lieferant {sid!r} unbekannt")
        if zeile.get("coverage_days") is None or zeile["coverage_days"] < 0:
            raise RankingError(f"buffer_options.csv: {sid}: coverage_days muss ≥ 0 sein")
        if zeile.get("carrying_rate") is None or zeile["carrying_rate"] < 0:
            raise RankingError(f"buffer_options.csv: {sid}: carrying_rate muss ≥ 0 sein")
    return partb["buffer_options"]


def kennzahlen_option(lieferant: dict[str, Any], option: dict[str, Any], demand: dict[str, Any] | None,
                      rates: list[dict[str, Any]]) -> dict[str, Any]:
    """Kennzahlen des Lieferanten mit Pufferlager (Reichweite s, Kostensatz r).

    V = min(Kapazität, Jahresbedarf), Q = round(V / 365 · s), m = mittlere
    Bestellmenge. Eine Bestellung zählt zusätzlich als perfekt, wenn nur
    Verspätung (≤ s Tage) und Fehlmenge (≤ Q) vorliegen; Dokumente und Befund
    bleiben Mängel. Eine Änderungsanfrage zählt zusätzlich als umgesetzt, wenn
    sie volume_up ist und change_pct/100 · m ≤ Q. Kosten: zusätzlich
    Q · Stückpreis (Angebot) · r / V je Stück (SCOR CO.2.3). Die Fehlerquote
    bleibt die des Basislieferanten."""
    basis = kennzahlen(lieferant, rates)
    orders, changes, konditionen = lieferant["orders"], lieferant["changes"], lieferant["conditions"] or {}
    s, r = option["coverage_days"], option["carrying_rate"]
    if not demand or konditionen.get("capacity_units_per_year") is None or konditionen.get("unit_price_eur") is None:
        raise RankingError(f"{lieferant['supplier_id']}: Kapazität, Stückpreis oder Bedarf fehlen für die Lager-Option")
    v = min(konditionen["capacity_units_per_year"], demand["annual_demand_units"])
    if v <= 0:
        raise RankingError(f"{lieferant['supplier_id']}: Jahresmenge V = {v}, Lagerkosten je Stück undefiniert")
    q = round(v / 365 * s)
    m = sum(o["qty_ordered"] for o in orders) / len(orders)

    def gedeckt(o: dict[str, Any]) -> bool:
        verspaetung = (date.fromisoformat(o["receipt_date"]) - date.fromisoformat(o["commit_date"])).days
        return o["docs"] and o["condition"] and verspaetung <= s and o["qty_ordered"] - o["qty_received"] <= q

    bestellungen = [o["po_id"] for o in orders if not _perfekt(o) and gedeckt(o)]
    anfragen = [c["cr_id"] for c in changes if not _umgesetzt(c) and c["change_type"] == "volume_up"
                and c["change_pct"] is not None and c["change_pct"] / 100 * m <= q]
    lager = q * konditionen["unit_price_eur"] * r / v
    perfekt = basis["details"]["liefertreue"]["perfect_orders"] + len(bestellungen)
    umgesetzt = basis["details"]["flexibilitaet"]["fulfilled_in_time"] + len(anfragen)
    return {
        "liefertreue": perfekt / len(orders),
        "flexibilitaet": umgesetzt / len(changes) if changes else None,
        "kosten": basis["kosten"] + lager,
        "details": {
            "liefertreue": {"perfect_orders": perfekt, "orders": len(orders), "covered_by_buffer": bestellungen},
            "flexibilitaet": {"fulfilled_in_time": umgesetzt, "change_requests": len(changes),
                              "covered_by_buffer": anfragen},
            "kosten": {**basis["details"]["kosten"], "base_per_unit": basis["kosten"], "carrying_cost_per_unit": lager,
                       "per_unit": basis["kosten"] + lager},
            "carrying_cost_per_unit": lager,
            "option": {"base_supplier": lieferant["supplier_id"], "coverage_days": s, "carrying_rate": r,
                       "buffer_qty": q, "annual_volume": v, "mean_order_qty": m},
        },
    }


# ----------------------------------------------------------------- TOPSIS


def _normieren(spalte: list[float], richtung: str, normierung: str) -> list[float]:
    if normierung == "vector":
        laenge = math.sqrt(sum(x * x for x in spalte))
        return [x / laenge if laenge else 0.0 for x in spalte]
    # Min-Max: danach sind alle Spalten Nutzenspalten (1 = bester Wert).
    lo, hi = min(spalte), max(spalte)
    if hi == lo:
        return [1.0 for _ in spalte]
    return [(x - lo) / (hi - lo) if richtung == "benefit" else (hi - x) / (hi - lo) for x in spalte]


def topsis(werte: dict[str, dict[str, float]], gewichte: dict[str, float],
           richtungen: dict[str, str] = KRITERIEN, normierung: str = "vector") -> list[dict[str, Any]]:
    """TOPSIS über die Alternativen in `werte` ({id: {kriterium: wert}}).

    Gibt je Alternative normierte und gewichtete Werte, Abstand zur idealen
    (d_plus) und zur anti-idealen Lösung (d_minus), Closeness und Rang zurück,
    sortiert nach Rang. Gleichstand bekommt denselben Rang."""
    if normierung not in NORMIERUNGEN:
        raise RankingError(f"Unbekannte Normierung {normierung!r}, erlaubt: {NORMIERUNGEN}")
    kriterien = list(richtungen)
    if set(gewichte) != set(kriterien):
        raise RankingError(f"Gewichte für {sorted(gewichte)}, erwartet {sorted(kriterien)}")
    summe = sum(gewichte.values())
    if summe <= 0 or any(w < 0 for w in gewichte.values()):
        raise RankingError("Gewichte müssen nichtnegativ sein und eine positive Summe haben")
    ids = list(werte)
    if len(ids) < 2:
        raise RankingError("Für ein Ranking braucht es mindestens zwei Lieferanten")
    for i in ids:
        fehlend = [k for k in kriterien if werte[i].get(k) is None]
        if fehlend:
            raise RankingError(f"{i}: kein Wert für {fehlend}")

    # Nach Min-Max sind alle Spalten Nutzenspalten.
    richtung = {k: "benefit" if normierung == "minmax" else richtungen[k] for k in kriterien}
    normiert = {i: {} for i in ids}
    for k in kriterien:
        for i, r in zip(ids, _normieren([werte[i][k] for i in ids], richtungen[k], normierung)):
            normiert[i][k] = r
    gewichtet = {i: {k: gewichte[k] / summe * normiert[i][k] for k in kriterien} for i in ids}
    ideal = {k: (max if richtung[k] == "benefit" else min)(gewichtet[i][k] for i in ids) for k in kriterien}
    anti = {k: (min if richtung[k] == "benefit" else max)(gewichtet[i][k] for i in ids) for k in kriterien}

    zeilen = []
    for i in ids:
        d_plus = math.sqrt(sum((gewichtet[i][k] - ideal[k]) ** 2 for k in kriterien))
        d_minus = math.sqrt(sum((gewichtet[i][k] - anti[k]) ** 2 for k in kriterien))
        naehe = d_minus / (d_plus + d_minus) if d_plus + d_minus else 0.5
        zeilen.append({"id": i, "normalized": normiert[i], "weighted": gewichtet[i],
                       "d_plus": d_plus, "d_minus": d_minus, "closeness": naehe})
    zeilen.sort(key=lambda z: -z["closeness"])
    for z in zeilen:
        z["rank"] = 1 + sum(round(y["closeness"], 12) > round(z["closeness"], 12) for y in zeilen)
    return zeilen


# --------------------------------------------------------------- Bauteil


def bauteil(partb: dict[str, Any], component: str) -> dict[str, Any]:
    if not partb.get("available"):
        raise RankingError(f"Keine Lieferantendaten: fehlend {partb.get('missing')}")
    if component not in partb["components"]:
        raise RankingError(f"Bauteil {component!r} unbekannt, bekannt: {sorted(partb['components'])}")
    return partb["components"][component]


def kennzahlen_bauteil(partb: dict[str, Any], component: str, options: bool = False) -> dict[str, dict[str, Any]]:
    """Kennzahlen aller Lieferanten eines Bauteils, unabhängig von Gewichten.
    Mit options zusätzlich je Zeile aus buffer_options.csv, deren Lieferant zum
    Bauteil gehört, die Alternative <supplier_id>+LAGER (Spec 08)."""
    rates, komp = partb["process_cost_rates"], bauteil(partb, component)
    kz = {s["supplier_id"]: kennzahlen(s, rates) for s in komp["suppliers"]}
    if options:
        lieferanten = {s["supplier_id"]: s for s in komp["suppliers"]}
        for option in optionen_pruefen(partb):
            if option["supplier_id"] in lieferanten:
                kz[option["supplier_id"] + OPTION_SUFFIX] = kennzahlen_option(
                    lieferanten[option["supplier_id"]], option, komp.get("demand"), rates)
    return kz


def ranking(partb: dict[str, Any], component: str, gewichte: dict[str, float], options: bool = False) -> dict[str, Any]:
    """Kennzahlen, Ranking mit Vektornormierung und Min-Max zur Sensitivität.
    Ohne options unverändert gegenüber vor Spec 08 (kein Schlüssel "options")."""
    kz = kennzahlen_bauteil(partb, component, options)
    werte = {sid: {k: z[k] for k in KRITERIEN} for sid, z in kz.items()}
    ergebnisse = {n: topsis(werte, gewichte, normierung=n) for n in NORMIERUNGEN}
    reihenfolge = {n: [z["id"] for z in ergebnisse[n]] for n in NORMIERUNGEN}
    return {
        **({"options": True} if options else {}),
        "component": component,
        "method": "TOPSIS",
        "normalization": "vector",
        "sensitivity": "minmax",
        "criteria": {k: {"direction": r, "indicator": KENNZAHLEN[k]} for k, r in KRITERIEN.items()},
        "weights": gewichte,
        "indicators": kz,
        "ranking": ergebnisse["vector"],
        "ranking_minmax": ergebnisse["minmax"],
        "same_order": reihenfolge["vector"] == reihenfolge["minmax"],
    }


# --------------------------------------------- aus einer fixen Gewichtsversion


def ranking_pfad(version_path: Path, options: bool = False, ordner: Path = partb_data.PARTB_DIR) -> Path:
    """…_ranking_w1.json, mit Lager-Optionen …_ranking_w1_options_<hash>.json.

    <hash> sind die ersten 8 Zeichen des SHA-256 von buffer_options.csv: Jede
    Fassung der Lager-Annahmen bekommt ihr eigenes eingefrorenes Ranking, ein
    früheres bleibt unverändert liegen."""
    version_path = Path(version_path)
    if "_weights_v" not in version_path.name:
        raise RankingError(f"{version_path.name} ist keine fixe Gewichtsversion (…_weights_v1.json)")
    ziel = version_path.with_name(version_path.name.replace("_weights_v", "_ranking_w"))
    if not options:
        return ziel
    datei = Path(ordner) / "buffer_options.csv"
    if not datei.is_file():
        raise RankingError(f"buffer_options.csv fehlt in {ordner}")
    return ziel.with_name(f"{ziel.stem}_options_{ledger.sha256(datei)[:8]}{ziel.suffix}")


def rank_from_version(version_path: Path, ordner: Path = partb_data.PARTB_DIR, options: bool = False) -> Path:
    """Ranking für den Fall einer fixen Gewichtsversion, eingefroren neben ihr."""
    version_path = Path(version_path)
    version = weight_version.load_version(version_path)
    partb = partb_data.load_partb(ordner)
    ergebnis = ranking(partb, version["case_id"], version["weights"], options)
    jetzt = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dateien = ("suppliers", "purchase_orders", "change_requests", "claims", "process_cost_rates")
    if options:  # V braucht Kapazität und Bedarf, die Option ihre eigene Datei
        dateien += ("conditions", "demand", "buffer_options")
    daten = {name: {"file": f"{name}.csv", "sha256": ledger.sha256(Path(ordner) / f"{name}.csv")} for name in dateien}
    inhalt = {
        "ranking_id": uuid.uuid4().hex,
        "created": jetzt,
        "run_id": version["run_id"],
        "run": version["run"],
        "case_id": version["case_id"],
        **ergebnis,
        # Wie in Gate 2: ein CR über der Schwelle blockiert nicht, wird aber mitgeführt.
        "cr": version["cr"],
        "cr_threshold": version["cr_threshold"],
        "weights_consistent": version["consistent"],
        "provenance": {
            "weight_version": {"file": version_path.name, "sha256": ledger.sha256(version_path),
                               "version_id": version["version_id"]},
            "partb_dir": partb["dir"],
            "partb_files": daten,
            "period": partb.get("period"),
        },
    }
    ereignis = {
        "event_id": uuid.uuid4().hex, "timestamp": jetzt, "run_id": version["run_id"], "case_id": version["case_id"],
        "run": version["run"], "actor": "code", "event_type": EVENT, "gate": None,
        "version_id": version["version_id"], "order": [z["id"] for z in ergebnis["ranking"]],
        "same_order": ergebnis["same_order"],
        **({"options": True} if options else {}),
    }
    ziel = ranking_pfad(version_path, options, ordner)
    try:
        ledger.einfrieren(ziel, inhalt, weight_version._log(version_path, version["run_id"]), [ereignis], EVENT)
    except ledger.LedgerError as exc:
        raise RankingError(str(exc)) from None
    return ziel


def load_ranking(path: Path) -> dict[str, Any]:
    """Einziger Lesezugang zu einem Ranking: prüft den Hash gegen das Log."""
    path = Path(path)
    inhalt = json.loads(path.read_text(encoding="utf-8"))
    try:
        return ledger.pruefe(path, weight_version._log(path, inhalt["run_id"]), EVENT)
    except ledger.LedgerError as exc:
        raise RankingError(str(exc)) from None


def ranking_fuer_gate2(gate2_path: Path, ordner: Path = partb_data.PARTB_DIR, options: bool = False) -> Path:
    """Schritt 8 und Teil B für eine Gate-2-Datei mit accept: fixe
    Gewichtsversion und Ranking, beide nur, wenn es sie noch nicht gibt.
    Deterministisch und ohne Modellaufruf, daher beliebig oft aufrufbar."""
    gate2_path = Path(gate2_path)
    run_id = json.loads(gate2_path.read_text(encoding="utf-8"))["run_id"]
    log = weight_version._log(gate2_path, run_id)
    version = weight_version.version_pfad(gate2_path)
    for ziel, abschluss, erzeugen in ((version, weight_version.EVENT, lambda: weight_version.freeze(gate2_path)),
                                      (ranking_pfad(version, options, ordner), EVENT,
                                       lambda: rank_from_version(version, ordner, options))):
        stand = ledger.zustand(ziel, log, abschluss)
        if stand == ledger.NACHHOLBAR:
            ledger.nachholen(ziel, log, abschluss)
        elif stand == ledger.OFFEN:
            erzeugen()
        elif stand != ledger.ABGESCHLOSSEN:
            raise RankingError(ledger.beschreibung(ziel, stand))
    return ranking_pfad(version, options, ordner)


# -------------------------------------------------------------------- CLI


def _drucken(partb: dict[str, Any], component: str, ergebnis: dict[str, Any] | None, options: bool = False) -> None:
    namen = {s["supplier_id"]: s["supplier_name"] for s in bauteil(partb, component)["suppliers"]}
    namen.update({sid + OPTION_SUFFIX: f"{name} mit Pufferlager" for sid, name in namen.items()})
    kz = ergebnis["indicators"] if ergebnis else kennzahlen_bauteil(partb, component, options)
    print(f"{'Lieferant':<14}{'Liefertreue':>13}{'Flexibilität':>14}{'Kosten €/Stk':>14}")
    for sid, z in kz.items():
        flex = f"{z['flexibilitaet']:.3f}" if z["flexibilitaet"] is not None else "–"
        print(f"{sid:<14}{z['liefertreue']:>13.3f}{flex:>14}{z['kosten']:>14.2f}")
    for sid, z in kz.items():
        if "option" in z["details"]:
            o, d = z["details"]["option"], z["details"]
            print(f"  {sid}: s = {o['coverage_days']} Tage, r = {o['carrying_rate']}, V = {o['annual_volume']}, "
                  f"Q = {o['buffer_qty']}, Lagerkosten {d['carrying_cost_per_unit']:.2f} €/Stk")
            print(f"    nur wegen des Lagers erfüllt: Bestellungen {', '.join(d['liefertreue']['covered_by_buffer']) or '–'}"
                  f" · Änderungsanfragen {', '.join(d['flexibilitaet']['covered_by_buffer']) or '–'}")
    if not ergebnis:
        return
    print("\nGewichte: " + " · ".join(f"{k} {w:.3f}" for k, w in ergebnis["weights"].items()))
    if "cr" in ergebnis:
        print(f"CR {ergebnis['cr']:.3f} (Schwelle {ergebnis['cr_threshold']})"
              + ("" if ergebnis["weights_consistent"] else "  ACHTUNG: über der Schwelle, in Gate 2 trotzdem accept"))
    print(f"\n{'Rang':<6}{'Lieferant':<14}{'Closeness':>11}{'d+':>9}{'d-':>9}   Min-Max")
    minmax = {z["id"]: z for z in ergebnis["ranking_minmax"]}
    for z in ergebnis["ranking"]:
        m = minmax[z["id"]]
        print(f"{z['rank']:<6}{z['id']:<14}{z['closeness']:>11.4f}{z['d_plus']:>9.4f}{z['d_minus']:>9.4f}"
              f"   Rang {m['rank']} ({m['closeness']:.4f})  {namen.get(z['id'], '')}")
    print("\nMin-Max-Sensitivität: " + ("gleiche Reihenfolge" if ergebnis["same_order"] else "Reihenfolge ABWEICHEND"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="partb_ranking.py", description=__doc__.splitlines()[0])
    parser.add_argument("version", type=Path, nargs="?", help="fixe Gewichtsversion (…_weights_v1.json)")
    parser.add_argument("--component", help="nur Kennzahlen dieses Bauteils, ohne Gewichte und ohne Datei")
    parser.add_argument("--partb-dir", type=Path, default=partb_data.PARTB_DIR)
    parser.add_argument("--options", action="store_true",
                        help="Pufferlager-Optionen aus buffer_options.csv als eigene Alternativen (Spec 08)")
    args = parser.parse_args(argv)
    if bool(args.version) == bool(args.component):
        parser.error("entweder eine Gewichtsversion oder --component angeben")
    partb = partb_data.load_partb(args.partb_dir)
    try:
        if args.component:
            _drucken(partb, args.component, None, args.options)
            return 0
        ziel = rank_from_version(args.version, args.partb_dir, args.options)
    except GateError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1
    ergebnis = json.loads(ziel.read_text(encoding="utf-8"))
    _drucken(partb, ergebnis["case_id"], ergebnis)
    print(f"\n-> {ziel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
