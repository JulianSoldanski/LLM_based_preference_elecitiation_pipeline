"""Schritt 6: Gewichte und Konsistenzverhältnis aus Paarvergleichen.

Reine Funktionen, ohne Modellaufruf und ohne numpy. Notebook, Gate 2 und die
fixe Gewichtsversion rechnen damit; Fragebogen- und LLM-Daten werden so mit
identischem Verfahren umgerechnet.

Die Rechnung entspricht pyDecision.ahp_method(matrix, wd="g") (5.1.x,
pyDecision/algorithm/ahp.py), mit der Kapitel 5 gerechnet ist:

* Gewichte: Zeilen-geometrisches Mittel, normiert auf Summe 1.
* lambda_max = Mittelwert über i von (A·w)_i / w_i.
* CI = (lambda_max - n) / (n - 1), CR = CI / RI(n).

Für n = 3 ist dieses lambda_max gleich dem größten Eigenwert der Matrix. Für
n >= 4 ist es eine Näherung daran (Design decision DD10). Für n <= 2 ist jede
reziproke Matrix konsistent; CR ist dann 0.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from prompt_builder import CONFIG_DIR
from schema import INDIFFERENCE, INDIFFERENCE_INTENSITIES

AHP_CONFIG = CONFIG_DIR / "ahp.yaml"
METHOD = "row_geometric_mean"
# Random Index wie in pyDecision, Index = n.
RANDOM_INDEX = (0, 0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.59)
MAX_SCALE = 9


class AhpError(ValueError):
    """Die Vergleiche ergeben keine vollständige reziproke Matrix."""


@dataclass(frozen=True)
class AhpResult:
    weights: dict[str, float]
    lambda_max: float
    ci: float
    cr: float
    n: int
    ri: float
    method: str = METHOD

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_matrix(comparisons: Iterable[Any], criteria_order: Sequence[str]) -> list[list[float]]:
    """Reziproke Matrix aus Vergleichen {"a", "b", "preference", "intensity"}
    (Dict oder Objekt mit diesen Attributen), Achsen in `criteria_order`.

    Wie to_matrix() im Notebook: "gleich" ergibt 1 (Intensität 0 und 1 sind
    beide gültig), sonst bekommt das bevorzugte Kriterium die Intensität und
    die Gegenrichtung den Kehrwert. Reziprok und mit Diagonale 1 per
    Konstruktion. Jedes Paar muss genau einmal vorkommen.
    """
    index = {c: i for i, c in enumerate(criteria_order)}
    if len(index) != len(criteria_order):
        raise AhpError(f"Doppelte Kriterien in {list(criteria_order)}")
    n = len(criteria_order)
    matrix = [[1.0] * n for _ in range(n)]
    gesehen: set[frozenset[str]] = set()

    for c in comparisons:
        a, b, preference, intensity = (_feld(c, k) for k in ("a", "b", "preference", "intensity"))
        unbekannt = {a, b} - set(index)
        if unbekannt:
            raise AhpError(f"Unbekanntes Kriterium {sorted(unbekannt)} im Vergleich ({a}, {b})")
        if a == b:
            raise AhpError(f"Vergleich eines Kriteriums mit sich selbst: {a}")
        paar = frozenset((a, b))
        if paar in gesehen:
            raise AhpError(f"Paar ({a}, {b}) kommt doppelt vor")
        gesehen.add(paar)
        wert = _wert(a, b, preference, intensity)
        i, j = index[a], index[b]
        matrix[i][j], matrix[j][i] = wert, 1.0 / wert

    fehlend = [
        (criteria_order[i], criteria_order[j])
        for i in range(n)
        for j in range(i + 1, n)
        if frozenset((criteria_order[i], criteria_order[j])) not in gesehen
    ]
    if fehlend:
        raise AhpError(f"Paare fehlen: {fehlend}")
    return matrix


def weights_and_cr(matrix: Sequence[Sequence[float]], criteria: Sequence[str] | None = None) -> AhpResult:
    """Gewichte (Zeilen-geometrisches Mittel), lambda_max, CI und CR.
    `criteria` benennt die Achsen; ohne Angabe sind die Schlüssel 0..n-1."""
    n = len(matrix)
    if n == 0 or any(len(zeile) != n for zeile in matrix):
        raise AhpError("Matrix ist nicht quadratisch")
    namen = list(criteria) if criteria is not None else list(range(n))
    if len(namen) != n:
        raise AhpError(f"{len(namen)} Kriterien für eine {n}x{n}-Matrix")
    if n >= len(RANDOM_INDEX):
        raise AhpError(f"Kein Random Index für n = {n}")

    # Reihenfolge der Operationen wie in pyDecision: Produkt der Zeile, dann
    # Wurzel, dann normieren.
    roh = [math.prod(zeile) ** (1 / n) for zeile in matrix]
    summe = sum(roh)
    w = [x / summe for x in roh]
    vektor = [sum(matrix[i][j] * w[j] for j in range(n)) / w[i] for i in range(n)]
    lambda_max = sum(vektor) / n
    ri = RANDOM_INDEX[n]
    if n <= 2:
        ci = cr = 0.0
    else:
        ci = (lambda_max - n) / (n - 1)
        cr = ci / ri
    return AhpResult(dict(zip(namen, w)), lambda_max, ci, cr, n, ri)


def compute(comparisons: Iterable[Any], criteria_order: Sequence[str]) -> AhpResult:
    """build_matrix und weights_and_cr in einem Schritt."""
    return weights_and_cr(build_matrix(comparisons, criteria_order), criteria_order)


def load_config(path: Path = AHP_CONFIG) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("weight_method", METHOD) != METHOD:
        raise AhpError(f"{path}: nur weight_method {METHOD} ist umgesetzt")
    return config


def cr_threshold(n: int, config: Mapping[str, Any]) -> float:
    schwelle = config.get("cr_threshold") or {}
    by_n = {int(k): float(v) for k, v in (schwelle.get("by_n") or {}).items()}
    return by_n.get(n, float(schwelle.get("default", 0.10)))


def is_consistent(result: AhpResult, config: Mapping[str, Any]) -> bool:
    return result.cr <= cr_threshold(result.n, config)


def evaluation(result: AhpResult, config: Mapping[str, Any]) -> dict[str, Any]:
    """Ergebnis samt der gerade gültigen Schwelle, so wie es in Dateien landet."""
    return {
        **result.to_dict(),
        "cr_threshold": cr_threshold(result.n, config),
        "consistent": is_consistent(result, config),
        "random_index_source": config.get("random_index_source"),
    }


def _feld(vergleich: Any, name: str) -> Any:
    return vergleich[name] if isinstance(vergleich, Mapping) else getattr(vergleich, name)


def _wert(a: str, b: str, preference: Any, intensity: Any) -> float:
    if isinstance(intensity, bool) or not isinstance(intensity, int):
        raise AhpError(f"({a}, {b}): Intensität {intensity!r} ist keine ganze Zahl")
    if preference == INDIFFERENCE:
        if intensity not in INDIFFERENCE_INTENSITIES:
            raise AhpError(f'({a}, {b}): "{INDIFFERENCE}" mit Intensität {intensity}')
        return 1.0
    if preference not in (a, b):
        raise AhpError(f"({a}, {b}): Präferenz {preference!r} gehört nicht zum Paar")
    if not 1 <= intensity <= MAX_SCALE:
        raise AhpError(f"({a}, {b}): Intensität {intensity} außerhalb von 1-{MAX_SCALE}")
    return float(intensity) if preference == a else 1.0 / intensity
