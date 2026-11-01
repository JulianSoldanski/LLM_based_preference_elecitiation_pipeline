"""Automatische CR-Reparatur vor Gate 2 (Spec 06, nur Demo).

Liegt CR nach Schritt 5 über der Schwelle, gehen Urteile an den Agenten
zurück, bevor ein Mensch sie sieht; höchstens repair_max_attempts Versuche
(config/ahp.yaml). Danach geht die Matrix mit Warnung an Gate 2.

Standard ist AUS. In den Evaluationsläufen bleibt die Reparatur aus und
berichtet wird CR_0: sobald repariert wird, ist CR Ziel einer Optimierung
und kein unabhängiges Qualitätsmaß mehr.

Welche Urteile zurückgehen:
    n >= 4  das Urteil mit der größten Abweichung |ln(a_ij * w_j / w_i)|
    n = 3   repair_rule_n3: "all" (alle drei gemeinsam) oder "fewest_passages"
            (das Paar mit den wenigsten Passagen aus Gate 1). Bei drei
            Kriterien sind die Abweichungen mit dem geometrischen Mittel immer
            gleich groß; kein einzelnes Urteil lässt sich herausgreifen.

Der Agent bekommt keinen Ziel-CR, keine Gewichte und keinen Hinweis auf
Inkonsistenz, nur den Revisionsprompt aus Gate 2 ohne Kommentar. Rein bis
auf `revise`, das der Aufrufer übergibt (dort liegt der Modellaufruf).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import ahp
from schema import Comparison

Pair = tuple[str, str]
# revise(paare, aktuelle Urteile) -> (gültige Revisionen je Paar, Aufrufinfos fürs Log)
ReviseFn = Callable[[list[Pair], list[Comparison]], tuple[dict[Pair, Comparison], list[dict[str, Any]]]]
RULES_N3 = ("all", "fewest_passages")


@dataclass(frozen=True)
class RepairOutcome:
    initial_judgments: list[Comparison]
    judgments: list[Comparison]
    cr_0: float
    cr_after_repair: float
    cr_threshold: float
    attempts: list[dict[str, Any]]
    warning: bool
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cr_0": self.cr_0,
            "repair_attempts": self.attempts,
            "cr_after_repair": self.cr_after_repair,
            "cr_threshold": self.cr_threshold,
            "repair_warning": self.warning,
        }


def pairs_to_repair(
    judgments: Sequence[Comparison],
    criteria: Sequence[str],
    result: ahp.AhpResult,
    rule_n3: str = "all",
    passages_per_criterion: Mapping[str, int] | None = None,
) -> list[Pair]:
    paare = [(c.a, c.b) for c in judgments]
    if len(criteria) == 3:
        if rule_n3 == "all":
            return paare
        if rule_n3 == "fewest_passages":
            if passages_per_criterion is None:
                raise ahp.AhpError("fewest_passages braucht die Passagen je Kriterium aus Gate 1")
            return [min(paare, key=lambda p: passages_per_criterion.get(p[0], 0) + passages_per_criterion.get(p[1], 0))]
        raise ahp.AhpError(f"Unbekannte repair_rule_n3 {rule_n3!r}, erlaubt sind {RULES_N3}")

    index = {c: i for i, c in enumerate(criteria)}
    matrix = ahp.build_matrix(judgments, criteria)
    w = result.weights

    def abweichung(paar: Pair) -> float:
        i, j = index[paar[0]], index[paar[1]]
        return abs(math.log(matrix[i][j] * w[paar[1]] / w[paar[0]]))

    return [max(paare, key=abweichung)]


def run_repair(
    judgments: Sequence[Comparison],
    criteria: Sequence[str],
    config: Mapping[str, Any],
    revise: ReviseFn,
    passages_per_criterion: Mapping[str, int] | None = None,
) -> RepairOutcome:
    versuche_max = int(config.get("repair_max_attempts", 3))
    regel = str(config.get("repair_rule_n3", "all"))
    aktuell = list(judgments)
    ergebnis = ahp.compute(aktuell, criteria)
    schwelle = ahp.cr_threshold(ergebnis.n, config)
    cr_0 = ergebnis.cr
    versuche: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    while ergebnis.cr > schwelle and len(versuche) < versuche_max:
        nummer = len(versuche) + 1
        paare = pairs_to_repair(aktuell, criteria, ergebnis, regel, passages_per_criterion)
        revisionen, aufrufe = revise(paare, list(aktuell))
        aktuell = [revisionen.get((c.a, c.b), c) for c in aktuell]
        ergebnis = ahp.compute(aktuell, criteria)
        versuch = {
            "attempt": nummer,
            "pairs": [list(p) for p in paare],
            "calls": [{k: v for k, v in a.items() if k != "raw"} for a in aufrufe],
            "judgments": [c.to_dict() for c in aktuell],
            "weights": ergebnis.weights,
            "cr": ergebnis.cr,
        }
        versuche.append(versuch)
        events.append({
            "actor": "agent",
            "event_type": "RepairAttempt",
            "gate": 2,
            "attempt": nummer,
            "pairs": versuch["pairs"],
            "prompt_sha256": [a.get("prompt_sha256") for a in aufrufe],
            "raw": [a.get("raw") for a in aufrufe],
            "valid": [a.get("valid") for a in aufrufe],
            "weights": ergebnis.weights,
            "cr": ergebnis.cr,
        })

    return RepairOutcome(
        initial_judgments=list(judgments),
        judgments=aktuell,
        cr_0=cr_0,
        cr_after_repair=ergebnis.cr,
        cr_threshold=schwelle,
        attempts=versuche,
        warning=ergebnis.cr > schwelle,
        events=events,
    )
