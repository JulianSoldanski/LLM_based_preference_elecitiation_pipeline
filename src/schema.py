"""Datenmodell und Validierung der Modellantworten.

Enthält bewusst keine Auswertung: hier entstehen nur Rohurteile in
strukturierter Form. Die Umrechnung in Prioritätsvektoren erfolgt später
gemeinsam für Fragebogen- und LLM-Daten.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

# Antworttoken für Indifferenz. Entspricht der Option "gleich wichtig" im
# Fragebogen: dort wird Indifferenz über die Richtungsfrage erfasst, nicht
# über den Skalenwert 1. Deshalb ist die Intensität hier 0 und der gültige
# Bereich bei gewählter Richtung 2-9.
# Erhebungsbedingungen. "history" ist die Hauptbedingung: alle Fälle laufen in
# einer Konversation, wie bei Probanden, die sich an vorherige Fälle erinnern.
# "stateless" ist die Kontrolle: ein Fall pro Aufruf, kein Verlauf.
CONDITION_HISTORY = "history"
CONDITION_STATELESS = "stateless"

INDIFFERENCE = "gleich"
# Der Prompt fragt bei Indifferenz die 0 ab. Die Skalentabelle nennt daneben die
# 1 als "gleich wichtig", und Modelle greifen erwartbar darauf zurück. Beide
# Werte kodieren dieselbe Aussage, deshalb sind beide gültig: eine formale
# Ablehnung würde ausgerechnet Indifferenz-Urteile systematisch aus den Daten
# entfernen und damit die Verteilung verzerren. Ab 2 ist die Angabe dagegen
# widersprüchlich -- Gleichwertigkeit mit Stärkegrad -- und bleibt ungültig.
INDIFFERENCE_INTENSITY = 0
INDIFFERENCE_INTENSITIES = (0, 1)
MIN_INTENSITY = 2
MAX_INTENSITY = 9

# Nur Codefences werden toleriert, keine inhaltlichen Abweichungen.
_FENCE = re.compile(r"\A\s*```[A-Za-z0-9_-]*\s*\n(?P<body>.*?)\n?\s*```\s*\Z", re.DOTALL)


@dataclass(frozen=True)
class Criterion:
    """Ein Bewertungskriterium aus config/criteria.yaml."""

    id: str
    name: str
    description: str


@dataclass(frozen=True)
class Case:
    """Ein Beschaffungsfall aus config/cases.yaml."""

    id: str
    title: str
    component: str
    statement_order: tuple[str, ...]
    statements: dict[str, str]

    def ordered_statements(self) -> list[tuple[str, str]]:
        """Aussagen in der Reihenfolge, in der die Probanden sie gesehen haben."""
        return [(cid, self.statements[cid]) for cid in self.statement_order]


@dataclass(frozen=True)
class Comparison:
    """Ein einzelner Paarvergleich: Richtung plus Intensität."""

    a: str
    b: str
    preference: str
    intensity: int


@dataclass
class Response:
    """Eine Modellantwort zu genau einem Fall, inklusive Rohtext und Metadaten."""

    case_id: str
    run: int
    provider: str
    model: str
    temperature: float
    raw: str
    timestamp: str
    comparisons: list[Comparison] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    condition: str = CONDITION_STATELESS
    # position und sequence dokumentieren die Stellung des Falls im Durchlauf.
    # Ohne sie ließe sich hinterher nicht prüfen, ob die Position das Urteil
    # beeinflusst hat. In der zustandslosen Bedingung bleiben sie leer, weil es
    # dort keinen Durchlauf mit Reihenfolge gibt.
    position: Optional[int] = None
    sequence: Optional[list[str]] = None
    seed: Optional[int] = None

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "condition": self.condition,
            "seed": self.seed,
            "case_id": self.case_id,
            "run": self.run,
            "position": self.position,
            "sequence": list(self.sequence) if self.sequence is not None else None,
            "valid": self.valid,
            "comparisons": [asdict(c) for c in self.comparisons],
            "errors": list(self.errors),
            "raw": self.raw,
        }

    def to_json_line(self) -> str:
        """Eine JSONL-Zeile. ensure_ascii=False, damit Umlaute lesbar bleiben."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


def strip_code_fence(text: str) -> str:
    """Entfernt einen umschliessenden Codefence, lässt alles andere unverändert."""
    match = _FENCE.match(text or "")
    return match.group("body") if match else (text or "")


def parse_response(
    raw: str, expected_pairs: Sequence[tuple[str, str]]
) -> tuple[list[Comparison], list[str]]:
    """Parst und validiert eine Rohantwort streng.

    Rückgabe: (Vergleiche, Fehler). Fehlerhafte Vergleiche werden nicht
    repariert und nicht übernommen; der Rohtext bleibt in der Response
    erhalten, damit später nachvollziehbar ist, was das Modell geliefert hat.
    """
    text = strip_code_fence(raw or "").strip()
    if not text:
        return [], ["Leere Antwort"]

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], [f"Antwort ist kein gültiges JSON ({exc.msg}, Zeile {exc.lineno})"]

    if not isinstance(payload, dict):
        return [], ["Antwort ist kein JSON-Objekt"]

    items = payload.get("comparisons")
    if not isinstance(items, list):
        return [], ['Feld "comparisons" fehlt oder ist keine Liste']

    errors: list[str] = []
    if len(items) != len(expected_pairs):
        errors.append(
            f"Erwartet {len(expected_pairs)} Vergleiche, erhalten {len(items)}"
        )

    comparisons: list[Comparison] = []
    for index, (item, pair) in enumerate(zip(items, expected_pairs), start=1):
        comparison, item_errors = _parse_item(item, pair, index)
        errors.extend(item_errors)
        if comparison is not None:
            comparisons.append(comparison)

    return comparisons, errors


def _parse_item(
    item: Any, pair: tuple[str, str], index: int
) -> tuple[Optional[Comparison], list[str]]:
    left, right = pair
    if not isinstance(item, dict):
        return None, [f"Vergleich {index}: kein JSON-Objekt"]

    errors: list[str] = []
    given_a, given_b = item.get("a"), item.get("b")
    if given_a != left or given_b != right:
        errors.append(
            f"Vergleich {index}: erwartetes Paar ({left}, {right}), "
            f"erhalten ({given_a!r}, {given_b!r})"
        )

    preference = item.get("preference")
    intensity = item.get("intensity")

    if preference not in (left, right, INDIFFERENCE):
        errors.append(
            f"Vergleich {index}: Präferenz {preference!r} ist weder {left!r} "
            f"noch {right!r} noch {INDIFFERENCE!r}"
        )

    # bool ist in Python ein int-Subtyp, wird hier aber nicht als Intensität akzeptiert.
    if isinstance(intensity, bool) or not isinstance(intensity, int):
        errors.append(f"Vergleich {index}: Intensität {intensity!r} ist keine ganze Zahl")
    elif preference == INDIFFERENCE and intensity not in INDIFFERENCE_INTENSITIES:
        erlaubt = " oder ".join(str(wert) for wert in INDIFFERENCE_INTENSITIES)
        errors.append(
            f'Vergleich {index}: Intensität muss bei "{INDIFFERENCE}" '
            f"{erlaubt} sein, ist {intensity}"
        )
    elif preference in (left, right) and not MIN_INTENSITY <= intensity <= MAX_INTENSITY:
        errors.append(
            f"Vergleich {index}: Intensität {intensity} liegt außerhalb von "
            f"{MIN_INTENSITY}-{MAX_INTENSITY}"
        )

    if errors:
        return None, errors
    return Comparison(a=left, b=right, preference=preference, intensity=intensity), []
