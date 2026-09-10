"""Lädt die Wissensbasis aus config/ und rendert die Prompt-Vorlagen.

Trennt Fachinhalt (YAML, Textvorlagen) von Agentenlogik: der Agent bekommt
fertige Prompts und kennt weder Kriterien noch Fälle. Neue Kriterien, Fälle
oder Skalenvarianten erfordern deshalb keine Änderung an agent.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from schema import (
    INDIFFERENCE,
    INDIFFERENCE_INTENSITY,
    MAX_INTENSITY,
    MIN_INTENSITY,
    Case,
    Criterion,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
PROMPTS_DIR = ROOT / "prompts"

_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")


class ConfigError(RuntimeError):
    """Wissensbasis und Vorlagen passen nicht zusammen."""


@dataclass
class PromptBuilder:
    criteria: list[Criterion]
    comparison_order: list[tuple[str, str]]
    scale: dict[str, Any]
    scenario: dict[str, Any]
    cases: list[Case]
    system_template: str
    user_template: str

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------ laden

    @classmethod
    def from_paths(
        cls, config_dir: Path | str = CONFIG_DIR, prompts_dir: Path | str = PROMPTS_DIR
    ) -> "PromptBuilder":
        config_dir, prompts_dir = Path(config_dir), Path(prompts_dir)
        criteria_raw = _load_yaml(config_dir / "criteria.yaml")
        scale_raw = _load_yaml(config_dir / "scale.yaml")
        cases_raw = _load_yaml(config_dir / "cases.yaml")

        criteria = [
            Criterion(
                id=str(entry["id"]),
                name=str(entry["name"]),
                description=_clean(entry["description"]),
            )
            for entry in criteria_raw.get("criteria", [])
        ]
        comparison_order = [
            tuple(str(cid) for cid in pair)
            for pair in criteria_raw.get("comparison_order", [])
        ]
        cases = [
            Case(
                id=str(entry["id"]),
                title=str(entry["title"]),
                component=str(entry["component"]),
                statement_order=tuple(str(cid) for cid in entry["statement_order"]),
                statements={
                    str(cid): _clean(text)
                    for cid, text in entry.get("statements", {}).items()
                },
            )
            for entry in cases_raw.get("cases", [])
        ]

        return cls(
            criteria=criteria,
            comparison_order=comparison_order,
            scale=scale_raw.get("scale", {}),
            scenario=cases_raw.get("scenario", {}),
            cases=cases,
            system_template=_read_text(prompts_dir / "system.txt"),
            user_template=_read_text(prompts_dir / "user.txt"),
        )

    # ------------------------------------------------------------- Zugriffe

    def criteria_by_id(self) -> dict[str, Criterion]:
        return {c.id: c for c in self.criteria}

    def criteria_by_name(self) -> dict[str, Criterion]:
        """Zugriff über den angezeigten Namen, u.a. für die Validierung."""
        return {c.name: c for c in self.criteria}

    def independence_note(self) -> str:
        """Hinweis auf die Unabhängigkeit der Fälle; leer, wenn nicht gepflegt."""
        return _clean(self.scenario.get("independence_note", ""))

    def case_ids(self) -> list[str]:
        return [case.id for case in self.cases]

    def case(self, case_id: str) -> Case:
        for case in self.cases:
            if case.id == case_id:
                return case
        raise KeyError(f"Unbekannter Fall: {case_id!r} (bekannt: {self.case_ids()})")

    def expected_pairs(self) -> list[tuple[str, str]]:
        """Paarreihenfolge, gegen die die Modellantwort validiert wird."""
        return list(self.comparison_order)

    # ------------------------------------------------------------- Rendering

    def build_system(self) -> str:
        return _render(
            self.system_template,
            {
                "criteria_block": self._criteria_block(),
                "independence_note": self.independence_note(),
                "scale_block": self._scale_block(),
                "scale_source": str(self.scale.get("source", "")),
                "indifference": INDIFFERENCE,
                "indifference_intensity": str(INDIFFERENCE_INTENSITY),
                "min_intensity": str(MIN_INTENSITY),
                "max_intensity": str(MAX_INTENSITY),
                "comparison_count": str(len(self.comparison_order)),
            },
        )

    def build_user(self, case_id: str) -> str:
        case = self.case(case_id)
        return _render(
            self.user_template,
            {
                "scenario_intro": _clean(self.scenario.get("intro", "")),
                "company": str(self.scenario.get("company", "")),
                "case_title": case.title,
                "component": case.component,
                "statements_block": self._statements_block(case),
                "comparison_block": self._comparison_block(),
                "comparison_count": str(len(self.comparison_order)),
            },
        )

    def build_prompts(self, case_id: str) -> tuple[str, str]:
        return self.build_system(), self.build_user(case_id)

    def _criteria_block(self) -> str:
        return "\n".join(f"- {c.name}: {c.description}" for c in self.criteria)

    def _scale_block(self) -> str:
        return "\n".join(
            f"- {level.get('values')} = {level.get('label')}: {level.get('explanation')}"
            for level in self.scale.get("levels", [])
        )

    def _statements_block(self, case: Case) -> str:
        # Mit Kriteriumslabel: die Zuordnung Aussage -> Kriterium ist vorgegeben,
        # genau wie im Fragebogen. Die Reihenfolge folgt der statement_order.
        by_id = self.criteria_by_id()
        return "\n".join(
            f"- {by_id[cid].name}: {text}" for cid, text in case.ordered_statements()
        )

    def _comparison_block(self) -> str:
        by_id = self.criteria_by_id()
        lines = []
        for index, (left, right) in enumerate(self.comparison_order, start=1):
            lines.append(
                f"{index}. {by_id[left].name} ({left}) vs. {by_id[right].name} ({right})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------ Validierung

    def _validate(self) -> None:
        """Prüft beim Start, dass Fallmaterial und Kriterienliste zusammenpassen.

        Absichtlich früh und laut: ein stiller Konfigurationsfehler würde
        Prompts erzeugen, die nicht mehr dem Fragebogen entsprechen.
        """
        problems: list[str] = []

        if len(self.criteria) < 2:
            problems.append("Es sind weniger als zwei Kriterien definiert.")

        ids = [c.id for c in self.criteria]
        if len(set(ids)) != len(ids):
            problems.append(f"Doppelte Kriterien-IDs: {ids}")
        if len(self.criteria_by_name()) != len(self.criteria):
            problems.append("Doppelte Kriteriennamen; Prompts wären nicht eindeutig.")

        known = set(ids)
        if not self.comparison_order:
            problems.append("comparison_order ist leer.")
        for position, pair in enumerate(self.comparison_order, start=1):
            if len(pair) != 2:
                problems.append(f"Vergleich {position} ist kein Paar: {pair}")
                continue
            left, right = pair
            unknown = {left, right} - known
            if unknown:
                problems.append(f"Vergleich {position} nennt unbekannte Kriterien: {sorted(unknown)}")
            elif left == right:
                problems.append(f"Vergleich {position} vergleicht {left} mit sich selbst.")

        if not self.cases:
            problems.append("Es sind keine Fälle definiert.")
        case_ids = [case.id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            problems.append(f"Doppelte Fall-IDs: {case_ids}")

        for case in self.cases:
            if set(case.statements) != known:
                problems.append(
                    f"Fall {case.id!r}: Aussagen {sorted(case.statements)} decken die "
                    f"Kriterien {sorted(known)} nicht genau ab."
                )
            if sorted(case.statement_order) != sorted(known):
                problems.append(
                    f"Fall {case.id!r}: statement_order {list(case.statement_order)} ist "
                    f"keine Permutation der Kriterien {sorted(known)}."
                )

        if not self.scale.get("levels"):
            problems.append("Die Skala enthält keine Stufen.")
        if not self.scale.get("source"):
            problems.append("Der Skala fehlt die Quellenangabe.")
        if not self.scenario.get("intro"):
            problems.append("Dem Szenario fehlt der Einleitungstext.")

        if problems:
            raise ConfigError("Konfiguration inkonsistent:\n- " + "\n- ".join(problems))


def _render(template: str, values: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise ConfigError(f"Unbekannter Platzhalter {{{{{key}}}}} in der Vorlage")
        return values[key]

    # Ein leerer optionaler Block (z.B. ein nicht gepflegter Hinweis) soll keine
    # doppelte Leerzeile hinterlassen.
    return re.sub(r"\n{3,}", "\n\n", _PLACEHOLDER.sub(replace, template))


def _clean(text: Any) -> str:
    return str(text).strip()


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise ConfigError(f"Vorlage fehlt: {path}")
    return path.read_text(encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Konfigurationsdatei fehlt: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path} enthält kein YAML-Objekt.")
    return data
