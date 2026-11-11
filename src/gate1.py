"""Gate 1: ein Mensch prüft die Zuordnung, bevor Paarvergleiche gemacht werden.

Nur Logik, keine Ein- oder Ausgabe. apply_gate1() enthält alle Regeln und ist
deterministisch; Zeitstempel, Event-IDs, Dateien und Einfrieren übernimmt die
Oberfläche (gate1_cli.py). Gate 1 läuft je Fall; ein Fall ist vorerst ein
Dokument (case_id = doc_id).

Aktionen je Passage (fest):
    confirm   Zuordnung bleibt
    delete    Passage fällt raus                       Begründung Pflicht
    reassign  neues Kriterium der Hierarchie oder none  Begründung Pflicht

Danach eine Entscheidung für den Fall: continue (Schritt 5 startet) oder
need_more_input (nichts geht an Schritt 5). continue verlangt eine Aktion für
jede Passage und Evidenz für jedes Kriterium, weil der Paarvergleich je Fall
genau eine Aussage pro Kriterium braucht.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from assignment import NONE, STATUS_OK

GATE = 1
CONFIRM, DELETE, REASSIGN = "confirm", "delete", "reassign"
ACTIONS = (CONFIRM, DELETE, REASSIGN)
CONTINUE, NEED_MORE_INPUT = "continue", "need_more_input"
SUPERVISED, AUTONOMOUS = "supervised", "autonomous"


class GateError(ValueError):
    """Eine Eingabe verletzt die Regeln von Gate 1."""


@dataclass(frozen=True)
class Assignment:
    """Output von Schritt 3 für eine Passage."""

    passage_id: str
    doc_id: str
    position: int
    text: str
    criterion: str  # Code aus der Hierarchie oder NONE
    reason: str


@dataclass(frozen=True)
class Action:
    passage_id: str
    action: str
    new_criterion: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class Actions:
    """Alles, was der Mensch für einen Fall entscheidet."""

    per_passage: tuple[Action, ...]
    decision: str
    reviewer: str | None
    mode: str = SUPERVISED
    decision_reason: str = ""


@dataclass(frozen=True)
class Gate1Result:
    validated_evidence: dict[str, list[str]]
    criteria_without_evidence: list[str]
    counts: dict[str, int]
    decision: str
    reviewer: str | None
    mode: str
    # Endstand je Passage (für Schritt 5 und die Nachvollziehbarkeit) und die
    # Ereignisse ohne event_id, timestamp, run_id und case_id.
    passages: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


def assignments_from_records(records: Iterable[dict[str, Any]]) -> list[Assignment]:
    """Übersetzt Datensätze aus assignment.py. Eine fehlgeschlagene Zuordnung
    kommt als "none" ins Gate, mit den Fehlern als Begründung. Was der
    frühere Relevanzfilter in der Zuordnung aussortiert hat ("relevant": false,
    nur in Läufen vom 07.11.2026), kommt nicht ins Gate."""
    ergebnis = []
    for r in records:
        if r.get("relevant") is False:
            continue
        if "doc_id" not in r or "position" not in r:
            raise GateError(f"{r['passage_id']}: doc_id/position fehlen (Lauf vor assignment_v2?)")
        if r["status"] == STATUS_OK:
            if len(r["criteria"]) > 1:
                raise GateError(f"{r['passage_id']}: mehrere Kriterien {r['criteria']}; Gate 1 braucht genau eines")
            criterion, reason = (r["criteria"] or [NONE])[0], r["justification"]
        else:
            criterion, reason = NONE, "Zuordnung fehlgeschlagen: " + "; ".join(r["errors"])
        ergebnis.append(Assignment(r["passage_id"], r["doc_id"], r["position"], r["text"], criterion, reason))
    return sorted(ergebnis, key=lambda a: (a.doc_id, a.position))


def apply_gate1(assignments: list[Assignment], actions: Actions, hierarchy: list[str]) -> Gate1Result:
    """Wendet die Entscheidungen des Menschen an. Wirft GateError bei jedem Regelverstoß."""
    erlaubt = [*hierarchy, NONE]
    nach_id = {a.passage_id: a for a in assignments}
    if len(nach_id) != len(assignments):
        raise GateError("Doppelte passage_id in der Zuordnung")
    for a in assignments:
        if a.criterion not in erlaubt:
            raise GateError(f"{a.passage_id}: Kriterium {a.criterion!r} ist nicht in der Hierarchie {hierarchy}")
    if actions.mode not in (SUPERVISED, AUTONOMOUS):
        raise GateError(f"Unbekannter Modus {actions.mode!r}")
    if actions.mode == SUPERVISED and not (actions.reviewer or "").strip():
        raise GateError("Im Modus supervised ist ein reviewer Pflicht")
    if actions.decision not in (CONTINUE, NEED_MORE_INPUT):
        raise GateError(f"Unbekannte Entscheidung {actions.decision!r}, erlaubt sind {CONTINUE}, {NEED_MORE_INPUT}")

    je_passage: dict[str, Action] = {}
    for action in actions.per_passage:
        _pruefe_aktion(action, nach_id, erlaubt)
        if action.passage_id in je_passage:
            raise GateError(f"{action.passage_id}: mehr als eine Aktion")
        je_passage[action.passage_id] = action

    offen = [a.passage_id for a in assignments if a.passage_id not in je_passage]
    if actions.decision == CONTINUE and offen:
        raise GateError(f"continue erst, wenn jede Passage eine Aktion hat; offen: {offen}")

    actor = "auto" if actions.mode == AUTONOMOUS else "human"
    passages, events = [], []
    evidence: dict[str, list[str]] = {code: [] for code in hierarchy}
    counts = {"confirmed": 0, "deleted": 0, "reassigned": 0}
    for a in sorted(assignments, key=lambda a: (a.doc_id, a.position)):
        action = je_passage.get(a.passage_id)
        final = _endstand(a, action)
        passages.append(
            {
                "passage_id": a.passage_id,
                "doc_id": a.doc_id,
                "position": a.position,
                "text": a.text,
                "criterion_assigned": a.criterion,
                "agent_reason": a.reason,
                "action": action.action if action else None,
                "criterion": final,
                "reason": action.reason.strip() if action else "",
            }
        )
        if action is None:
            continue
        counts[{CONFIRM: "confirmed", DELETE: "deleted", REASSIGN: "reassigned"}[action.action]] += 1
        events.append(_event(actor, actions.reviewer, "GateAction", a.passage_id, a.criterion, final, action))
        # Gelöschte Passagen und Endstand "none" gehen nicht an Schritt 5.
        if final not in (None, NONE):
            evidence[final].append(a.passage_id)

    ohne = [code for code in hierarchy if not evidence[code]]
    if actions.decision == CONTINUE and ohne:
        raise GateError(f"continue nicht möglich, Kriterien ohne Evidenz: {ohne}; need_more_input wählen")
    events.append(
        {
            "actor": actor,
            "event_type": "GateDecision",
            "gate": GATE,
            "passage_id": None,
            "action": None,
            "old_value": None,
            "new_value": actions.decision,
            "reason": actions.decision_reason.strip(),
            "reviewer": actions.reviewer,
        }
    )
    return Gate1Result(
        validated_evidence={code: ids for code, ids in evidence.items() if ids},
        criteria_without_evidence=ohne,
        counts=counts,
        decision=actions.decision,
        reviewer=actions.reviewer,
        mode=actions.mode,
        passages=passages,
        events=events,
    )


def autonomous(assignments: list[Assignment], hierarchy: list[str]) -> Gate1Result:
    """Autonomer Modus: alles bestätigen. continue, solange jedes Kriterium
    Evidenz hat, sonst need_more_input."""
    alles_bestaetigt = tuple(Action(a.passage_id, CONFIRM) for a in assignments)
    belegt = {a.criterion for a in assignments}
    decision = CONTINUE if set(hierarchy) <= belegt else NEED_MORE_INPUT
    return apply_gate1(assignments, Actions(alles_bestaetigt, decision, reviewer=None, mode=AUTONOMOUS), hierarchy)


def _pruefe_aktion(action: Action, nach_id: dict[str, Assignment], erlaubt: list[str]) -> None:
    pid = action.passage_id
    if pid not in nach_id:
        raise GateError(f"Aktion für unbekannte Passage {pid!r}")
    if action.action not in ACTIONS:
        raise GateError(f"{pid}: unbekannte Aktion {action.action!r}, erlaubt sind {list(ACTIONS)}")
    if action.action in (DELETE, REASSIGN) and not action.reason.strip():
        raise GateError(f"{pid}: {action.action} braucht eine Begründung")
    if action.action == REASSIGN:
        if action.new_criterion not in erlaubt:
            raise GateError(f"{pid}: reassign auf unbekanntes Kriterium {action.new_criterion!r}, erlaubt sind {erlaubt}")
        if action.new_criterion == nach_id[pid].criterion:
            raise GateError(f"{pid}: reassign auf das bisherige Kriterium; dafür confirm verwenden")
    elif action.new_criterion is not None:
        raise GateError(f"{pid}: new_criterion nur bei reassign")


def _endstand(a: Assignment, action: Action | None) -> str | None:
    """Kriterium nach der Aktion; None bei delete oder ohne Aktion."""
    if action is None or action.action == DELETE:
        return None
    return action.new_criterion if action.action == REASSIGN else a.criterion


def _event(actor: str, reviewer: str | None, typ: str, pid: str, alt: str, neu: str | None, action: Action) -> dict[str, Any]:
    return {
        "actor": actor,
        "event_type": typ,
        "gate": GATE,
        "passage_id": pid,
        "action": action.action,
        "old_value": alt,
        "new_value": neu,
        "reason": action.reason.strip(),
        "reviewer": reviewer,
    }
