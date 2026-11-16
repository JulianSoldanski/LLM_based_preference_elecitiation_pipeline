"""Rohdaten für Teil B laden (Lieferanten von Bike Inc.).

Liest die CSV-Dateien aus data/partb/ (erzeugt mit
scripts/generate_partb_data.py) und gibt sie je Bauteil gebündelt zurück.
Werte werden nur in Zahlen umgewandelt. Zusätzlich gibt es je Bestellung
vier Ja/Nein-Merkmale (pünktlich, vollständig, Dokumente, Zustand) und je
Lieferant einfache Zählungen. Das sind Vergleiche und Zählungen auf den
Rohdaten, keine SCOR-Kennzahlen; die rechnet Teil B.

Optional: buffer_options.csv (Pufferlager-Optionen, Spec 08). Die Datei ist
eine Eingabe des Einkäufers, nicht vom Generator erzeugt; fehlt sie, fehlt nur
der Schlüssel "buffer_options".

Nur Standardbibliothek und PyYAML.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from prompt_builder import ROOT

PARTB_DIR = ROOT / "data" / "partb"

INT = {"qty_ordered", "qty_received", "qty_rejected", "docs_ok", "qty_affected", "quoted_lead_time_days",
       "moq", "capacity_units_per_year", "annual_demand_units", "fulfilled_pct", "change_pct", "coverage_days"}
FLOAT = {"unit_price_eur", "freight_eur", "freight_eur_per_shipment", "handling_cost_eur", "cost_eur",
         "max_defect_rate", "carrying_rate"}


def _wert(spalte: str, text: str) -> Any:
    if text == "":
        return None
    if spalte in INT:
        return int(text)
    if spalte in FLOAT:
        return float(text)
    return text


def _lesen(pfad: Path) -> list[dict[str, Any]]:
    with pfad.open(encoding="utf-8", newline="") as f:
        return [{k: _wert(k, v) for k, v in zeile.items()} for zeile in csv.DictReader(f)]


def _bestellung(o: dict[str, Any]) -> dict[str, Any]:
    """Merkmale je Bestellung, direkt aus den Spalten abgelesen."""
    return {
        **o,
        "on_time": o["receipt_date"] <= o["commit_date"],   # ISO-Daten: Textvergleich genügt
        "in_full": o["qty_received"] >= o["qty_ordered"],
        "docs": o["docs_ok"] == 1,
        "condition": o["qty_rejected"] == 0,
    }


def load_partb(ordner: Path = PARTB_DIR) -> dict[str, Any]:
    """Alle Rohdaten, gegliedert nach Bauteil und Lieferant.

    Fehlt der Ordner oder eine Datei, kommt {"available": False, ...} zurück,
    damit das Frontend einen Hinweis zeigen kann statt abzubrechen.
    """
    ordner = Path(ordner)
    dateien = ["suppliers", "conditions", "purchase_orders", "change_requests", "claims", "demand",
               "process_cost_rates"]
    fehlend = [d for d in dateien if not (ordner / f"{d}.csv").is_file()]
    if fehlend:
        return {"available": False, "dir": str(ordner), "missing": [f"{d}.csv" for d in fehlend]}
    t = {d: _lesen(ordner / f"{d}.csv") for d in dateien}

    zeitraum = None
    profile = ordner / "profiles.yaml"
    if profile.is_file():
        p = yaml.safe_load(profile.read_text(encoding="utf-8"))
        zeitraum = {"start": str(p.get("period_start")), "end": str(p.get("period_end"))}

    konditionen = {c["supplier_id"]: c for c in t["conditions"]}
    bedarf = {d["component"]: d for d in t["demand"]}
    komponenten: dict[str, Any] = {}
    for s in t["suppliers"]:
        sid = s["supplier_id"]
        orders = [_bestellung(o) for o in t["purchase_orders"] if o["supplier_id"] == sid]
        changes = [c for c in t["change_requests"] if c["supplier_id"] == sid]
        claims = [c for c in t["claims"] if c["supplier_id"] == sid]
        counts = {
            "orders": len(orders),
            "late": sum(not o["on_time"] for o in orders),
            "short": sum(not o["in_full"] for o in orders),
            "docs_bad": sum(not o["docs"] for o in orders),
            "rejected_orders": sum(not o["condition"] for o in orders),
            "qty_received": sum(o["qty_received"] for o in orders),
            "qty_rejected": sum(o["qty_rejected"] for o in orders),
            "changes": {r: sum(c["response"] == r for c in changes) for r in ("accepted", "partial", "rejected")},
            "claims": {k: sum(c["claim_type"] == k for c in claims)
                       for k in ("incoming_inspection", "field_warranty", "return")},
            "claim_units": sum(c["qty_affected"] for c in claims),
        }
        eintrag = komponenten.setdefault(s["component"], {"demand": bedarf.get(s["component"]), "suppliers": []})
        eintrag["suppliers"].append({**s, "conditions": konditionen.get(sid), "counts": counts,
                                     "orders": orders, "changes": changes, "claims": claims})
    ergebnis = {
        "available": True,
        "dir": str(ordner.relative_to(ROOT) if ordner.is_relative_to(ROOT) else ordner),
        "period": zeitraum,
        "process_cost_rates": t["process_cost_rates"],
        "components": komponenten,
    }
    if (ordner / "buffer_options.csv").is_file():
        ergebnis["buffer_options"] = _lesen(ordner / "buffer_options.csv")
    return ergebnis
