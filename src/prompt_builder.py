"""Lädt die Wissensbasis aus config/ und rendert die Prompt-Vorlagen.

Trennt Fachinhalt (YAML, Textvorlagen) von Agentenlogik: der Agent bekommt
fertige Prompts und kennt weder Kriterien noch Fälle. Neue Kriterien, Fälle
oder Skalenvarianten erfordern deshalb keine Änderung an agent.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

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

# Begründung je Paarvergleich, nur mit justification=True. Ohne die Option
# ergeben die Platzhalter wörtlich den bisherigen Prompt, damit frühere Läufe
# vergleichbar bleiben. Die Begründung steht wie in Schritt 3 vor dem Urteil.
JUSTIFICATION_FIELD = '"justification": "<Begründung in ein bis zwei Sätzen>", '
JUSTIFICATION_RULE = (
    '- "justification" begründet den Vergleich in ein bis zwei Sätzen und nennt die\n'
    '  Aussagen, auf die du dich stützt. Schreibe die Begründung vor "preference".\n'
    "- Gib keinen weiteren Fließtext und keine zusätzlichen Felder aus."
)
NO_JUSTIFICATION_RULE = "- Gib keine Begründung, keinen Fließtext und keine zusätzlichen Felder aus."


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
    # Code der Zuordnung (z.B. "DR") -> Kriterien-ID (z.B. "liefertreue")
    assignment_codes: dict[str, str] = field(default_factory=dict)
    # Modell begründet jeden Paarvergleich (Prompt und Validierung).
    justification: bool = False

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
            assignment_codes={
                str(entry["assignment_code"]): str(entry["id"])
                for entry in criteria_raw.get("criteria", [])
                if entry.get("assignment_code")
            },
        )

    def with_gate1(self, gate_results: Iterable[dict[str, Any]]) -> "PromptBuilder":
        """Neue Instanz, deren Fälle die Aussagen aus validierten Gate-1-
        Ergebnissen beziehen (je Fall eines, siehe gate1_cli.load_validated).

        Titel, Bauteil und statement_order bleiben aus cases.yaml. Mehrere
        Passagen eines Kriteriums werden in ihrer Dokumentreihenfolge zu einer
        Aussage verbunden. Die Vorlagen bleiben unverändert; die Validierung
        prüft wie bisher, dass jeder Fall jedes Kriterium genau einmal belegt.
        """
        cases = []
        for gate in gate_results:
            vorlage = self.case(gate["case_id"])
            texte = {p["passage_id"]: _clean(p["text"]) for p in sorted(gate["passages"], key=lambda p: p["position"])}
            statements = {}
            for code, passage_ids in gate["validated_evidence"].items():
                if code not in self.assignment_codes:
                    raise ConfigError(f"Fall {vorlage.id!r}: Code {code!r} hat kein assignment_code in criteria.yaml")
                reihenfolge = [pid for pid in texte if pid in passage_ids]
                statements[self.assignment_codes[code]] = " ".join(texte[pid] for pid in reihenfolge)
            cases.append(replace(vorlage, statements=statements))
        if not cases:
            raise ConfigError("Keine Gate-1-Ergebnisse übergeben.")
        # Reihenfolge aus cases.yaml, unabhängig von der Reihenfolge der Dateien:
        # in der stateless-Bedingung ist sie die erhobene Reihenfolge.
        cases.sort(key=lambda case: self.case_ids().index(case.id))
        return replace(self, cases=cases)

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
                "justification_field": JUSTIFICATION_FIELD if self.justification else "",
                "justification_rule": JUSTIFICATION_RULE if self.justification else NO_JUSTIFICATION_RULE,
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

        if self.assignment_codes and sorted(self.assignment_codes.values()) != sorted(ids):
            problems.append(
                f"assignment_code muss jedem Kriterium genau einen eigenen Code geben: {self.assignment_codes}"
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
