"""Gate 2: ein Mensch prüft je Fall die Paarvergleiche und die Gewichte.

Nur Logik, keine Ein- oder Ausgabe und kein Modellaufruf; wie gate1.py.
Zeitstempel, Dateien, Einfrieren und die Revision durch den Agenten übernimmt
die Oberfläche (gate2_cli.py, Viewer). Gewichte und CR rechnet immer ahp.py,
nie das Modell.

Aktionen je Urteil (fest):
    confirm   Urteil bleibt
    revise    Urteil geht mit Kommentar an den Agenten zurück    Kommentar Pflicht
    set       Mensch setzt preference und intensity selbst       Begründung Pflicht

Ablauf eines Zyklus:
    submit(actions)       prüft die Aktionen, übernimmt set und gibt die
                          revise-Paare mit Kommentar zurück
    complete(revisions)   übernimmt die Revisionen (ungültig: altes Urteil
                          bleibt, mit Hinweis), rechnet neu, legt den Zyklus ab
Ein Zyklus nur aus confirm beendet das Gate. Höchstens max_cycles Zyklen; im
letzten sind nur confirm und set erlaubt. Danach decide(accept|reject). Ein CR
über der Schwelle verhindert accept nicht, wird aber im Ergebnis vermerkt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import ahp
from gate1 import AUTONOMOUS, SUPERVISED, GateError
from schema import INDIFFERENCE, MAX_INTENSITY, MIN_INTENSITY, Comparison

GATE = 2
CONFIRM, REVISE, SET = "confirm", "revise", "set"
ACTIONS = (CONFIRM, REVISE, SET)
ACCEPT, REJECT = "accept", "reject"
STATUS_PASSED, STATUS_STOPPED = "passed_gate2", "stopped_at_gate2"
DEFAULT_MAX_CYCLES = 3
# set: "gleich" nur mit 0, sonst MIN-MAX (wie die Antwortregeln von Schritt 5).
SET_INDIFFERENCE_INTENSITY = 0

Pair = tuple[str, str]


@dataclass(frozen=True)
class JudgmentAction:
    pair: Pair
    action: str
    preference: str | None = None
    intensity: int | None = None
    reason: str = ""


@dataclass(frozen=True)
class Gate2Result:
    case_id: str
    initial: dict[str, Any]
    cycles: list[dict[str, Any]]
    final: dict[str, Any]
    counts: dict[str, int]
    decision: str
    reviewer: str | None
    mode: str
    hints: list[str]
    # Reparatur vor dem Gate (Spec 06), sonst None.
    repair: dict[str, Any] | None = None
    # Ereignisse ohne event_id, timestamp, run_id und case_id.
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        return STATUS_PASSED if self.decision == ACCEPT else STATUS_STOPPED


def value(c: Comparison) -> dict[str, Any]:
    return {"preference": c.preference, "intensity": c.intensity}


class Gate2Session:
    """Zustand von Gate 2 für einen Fall. Rein: keine Ein-/Ausgabe."""

    def __init__(
        self,
        case_id: str,
        criteria: Sequence[str],
        judgments: Sequence[Comparison],
        config: Mapping[str, Any],
        *,
        reviewer: str | None = None,
        mode: str = SUPERVISED,
        max_cycles: int | None = None,
        repair: Any = None,
    ) -> None:
        if mode not in (SUPERVISED, AUTONOMOUS):
            raise GateError(f"Unbekannter Modus {mode!r}")
        if mode == SUPERVISED and not (reviewer or "").strip():
            raise GateError("Im Modus supervised ist ein reviewer Pflicht")
        self.case_id, self.criteria, self.config = case_id, list(criteria), config
        self.reviewer, self.mode = reviewer, mode
        self.max_cycles = int(max_cycles or config.get("gate2_max_cycles") or DEFAULT_MAX_CYCLES)
        self.actor = "auto" if mode == AUTONOMOUS else "human"
        self.repair = repair

        # Mit Reparatur (Spec 06) ist initial der Stand davor (CR_0); das Gate
        # beginnt mit den reparierten Urteilen.
        anfang = list(repair.initial_judgments) if repair is not None else list(judgments)
        self.judgments: list[Comparison] = list(judgments)
        paare = [(c.a, c.b) for c in self.judgments]
        if len(set(paare)) != len(paare) or sorted(paare) != sorted((c.a, c.b) for c in anfang):
            raise GateError(f"Urteile passen nicht zusammen: {paare}")
        self.initial = {"judgments": [c.to_dict() for c in anfang], "evaluation": self._bewerte(anfang)}
        self.cycles: list[dict[str, Any]] = []
        self.counts = {"confirmed": 0, "revised": 0, "set": 0}
        self.hints: list[str] = []
        self.events: list[dict[str, Any]] = [self._weights_event(0, self.initial["evaluation"])]
        if repair is not None:
            self.events += list(repair.events)
        self._offen: dict[str, Any] | None = None
        self.decision: str | None = None

    # ------------------------------------------------------------- Zustand

    @property
    def cycle(self) -> int:
        """Nummer des laufenden bzw. nächsten Zyklus (ab 1)."""
        return len(self.cycles) + 1

    @property
    def ended(self) -> bool:
        """Letzter Zyklus nur aus confirm: das Gate ist beendet."""
        return bool(self.cycles) and all(a["action"] == CONFIRM for a in self.cycles[-1]["actions"])

    @property
    def can_decide(self) -> bool:
        return self.decision is None and self._offen is None and (self.ended or len(self.cycles) >= self.max_cycles)

    @property
    def revise_allowed(self) -> bool:
        return self.cycle < self.max_cycles

    @property
    def pending(self) -> list[tuple[Pair, str]]:
        """revise-Paare mit Kommentar, solange die Revision aussteht."""
        return list(self._offen["revise"].items()) if self._offen else []

    def evaluation(self) -> dict[str, Any]:
        return self._bewerte(self.judgments)

    def history(self, pair: Pair) -> list[dict[str, Any]]:
        """Frühere Eingriffe an einem Paar (revise/set) mit Kommentar und dem
        Urteil davor und danach, damit sie in späteren Zyklen sichtbar bleiben."""
        verlauf = []
        for zyklus in self.cycles:
            for action, vorher, nachher in zip(zyklus["actions"], zyklus["before"], zyklus["after"]):
                if tuple(action["pair"]) == tuple(pair) and action["action"] != CONFIRM:
                    verlauf.append({
                        "cycle": zyklus["cycle"],
                        "action": action["action"],
                        "reason": action["reason"],
                        "before": {"preference": vorher["preference"], "intensity": vorher["intensity"]},
                        "after": {"preference": nachher["preference"], "intensity": nachher["intensity"]},
                    })
        return verlauf

    # ------------------------------------------------------------- Zyklus

    def submit(self, actions: Sequence[JudgmentAction]) -> list[tuple[Pair, str]]:
        """Prüft die Aktionen eines Zyklus und übernimmt set. Gibt die Paare
        zurück, die an den Agenten gehen (leer: Zyklus ist abgeschlossen)."""
        if self.decision is not None:
            raise GateError("Gate 2 ist bereits entschieden")
        if self._offen is not None:
            raise GateError("Die Revision des laufenden Zyklus steht noch aus")
        if self.ended:
            raise GateError("Gate 2 ist beendet (Zyklus nur mit confirm); jetzt accept oder reject")
        if len(self.cycles) >= self.max_cycles:
            raise GateError(f"Höchstens {self.max_cycles} Zyklen; jetzt accept oder reject")

        nach_paar = {(c.a, c.b): c for c in self.judgments}
        je_paar: dict[Pair, JudgmentAction] = {}
        for action in actions:
            paar = tuple(action.pair)
            self._pruefe(action, paar, nach_paar)
            if paar in je_paar:
                raise GateError(f"{paar}: mehr als eine Aktion")
            je_paar[paar] = action
        offen = [p for p in nach_paar if p not in je_paar]
        if offen:
            raise GateError(f"Jedes Urteil braucht eine Aktion; offen: {offen}")

        vorher = list(self.judgments)
        nachher = [
            Comparison(c.a, c.b, je_paar[(c.a, c.b)].preference, je_paar[(c.a, c.b)].intensity)
            if je_paar[(c.a, c.b)].action == SET else c
            for c in vorher
        ]
        self._offen = {
            "before": vorher,
            "after_set": nachher,
            "actions": [je_paar[(c.a, c.b)] for c in vorher],
            "revise": {(c.a, c.b): je_paar[(c.a, c.b)].reason.strip() for c in vorher if je_paar[(c.a, c.b)].action == REVISE},
        }
        if not self._offen["revise"]:
            self._abschliessen({}, [])
            return []
        return self.pending

    def complete(self, revisions: Mapping[Pair, Comparison], calls: Sequence[Mapping[str, Any]] = ()) -> None:
        """Übernimmt die Revisionen des laufenden Zyklus. Ein Paar ohne gültige
        Revision behält sein Urteil; das Gate zeigt dazu einen Hinweis.
        `calls`: je Revisionsaufruf model, provider, prompt_version,
        prompt_sha256, raw, valid, errors, pairs, attempt (für das Log)."""
        if self._offen is None:
            raise GateError("Keine Revision offen")
        fremd = set(revisions) - set(self._offen["revise"])
        if fremd:
            raise GateError(f"Revision für nicht zurückgegebene Paare: {sorted(fremd)}")
        for p, c in revisions.items():
            if (c.a, c.b) != tuple(p):
                raise GateError(f"Revision für {p} trägt das Paar {(c.a, c.b)}")
        self._abschliessen(dict(revisions), list(calls))

    def decide(self, decision: str, reason: str = "") -> Gate2Result:
        if decision not in (ACCEPT, REJECT):
            raise GateError(f"Unbekannte Entscheidung {decision!r}, erlaubt sind {ACCEPT}, {REJECT}")
        if not self.can_decide:
            raise GateError("Entscheidung erst nach einem Zyklus nur mit confirm oder nach dem letzten Zyklus")
        self.decision = decision
        final = {"judgments": [c.to_dict() for c in self.judgments], "evaluation": self.evaluation()}
        self.events.append({
            "actor": self.actor,
            "event_type": "GateDecision",
            "gate": GATE,
            "cycle": len(self.cycles),
            "pair": None,
            "action": None,
            "old_value": None,
            "new_value": decision,
            "reason": reason.strip(),
            "reviewer": self.reviewer,
            "cr": final["evaluation"]["cr"],
            "cr_threshold": final["evaluation"]["cr_threshold"],
            "cr_above_threshold": not final["evaluation"]["consistent"],
        })
        return Gate2Result(
            case_id=self.case_id,
            initial=self.initial,
            cycles=self.cycles,
            final=final,
            counts=dict(self.counts),
            decision=decision,
            reviewer=self.reviewer,
            mode=self.mode,
            hints=list(self.hints),
            repair=self.repair.to_dict() if self.repair is not None else None,
            events=list(self.events),
        )

    # ------------------------------------------------------------- intern

    def _pruefe(self, action: JudgmentAction, paar: Pair, nach_paar: dict[Pair, Comparison]) -> None:
        if paar not in nach_paar:
            raise GateError(f"Aktion für unbekanntes Paar {paar}")
        if action.action not in ACTIONS:
            raise GateError(f"{paar}: unbekannte Aktion {action.action!r}, erlaubt sind {list(ACTIONS)}")
        if action.action in (REVISE, SET) and not action.reason.strip():
            raise GateError(f"{paar}: {action.action} braucht eine Begründung")
        if action.action == REVISE and not self.revise_allowed:
            raise GateError(f"{paar}: im letzten Zyklus ({self.max_cycles}) nur confirm oder set")
        if action.action != SET:
            if action.preference is not None or action.intensity is not None:
                raise GateError(f"{paar}: preference/intensity nur bei set")
            return
        a, b = paar
        p, i = action.preference, action.intensity
        if isinstance(i, bool) or not isinstance(i, int):
            raise GateError(f"{paar}: Intensität {i!r} ist keine ganze Zahl")
        if p == INDIFFERENCE:
            if i != SET_INDIFFERENCE_INTENSITY:
                raise GateError(f'{paar}: "{INDIFFERENCE}" nur mit Intensität {SET_INDIFFERENCE_INTENSITY}, nicht {i}')
        elif p in (a, b):
            if not MIN_INTENSITY <= i <= MAX_INTENSITY:
                raise GateError(f"{paar}: Intensität {i} außerhalb von {MIN_INTENSITY}-{MAX_INTENSITY}")
        else:
            raise GateError(f"{paar}: Präferenz {p!r} ist weder {a!r} noch {b!r} noch {INDIFFERENCE!r}")
        bisher = nach_paar[paar]
        if (p, i) == (bisher.preference, bisher.intensity) or (p == bisher.preference == INDIFFERENCE):
            raise GateError(f"{paar}: set auf den bisherigen Wert; dafür confirm verwenden")

    def _abschliessen(self, revisions: dict[Pair, Comparison], calls: list[Mapping[str, Any]]) -> None:
        offen, nummer = self._offen, len(self.cycles) + 1
        hinweise = []
        nachher = []
        for c in offen["after_set"]:
            paar = (c.a, c.b)
            if paar in offen["revise"]:
                neu = revisions.get(paar)
                if neu is None:
                    hinweise.append(f"{c.a} vs. {c.b}: Revision ungültig, bisheriges Urteil bleibt")
                nachher.append(neu or c)
            else:
                nachher.append(c)

        for call in calls:
            self.events.append({"actor": "agent", "event_type": "AgentRevision", "gate": GATE, "cycle": nummer, **call})
        for action, alt, neu in zip(offen["actions"], offen["before"], nachher):
            self.counts[{CONFIRM: "confirmed", REVISE: "revised", SET: "set"}[action.action]] += 1
            self.events.append({
                "actor": self.actor,
                "event_type": "GateAction",
                "gate": GATE,
                "cycle": nummer,
                "pair": [alt.a, alt.b],
                "action": action.action,
                "old_value": value(alt),
                "new_value": value(neu),
                "reason": action.reason.strip(),
                "reviewer": self.reviewer,
            })

        geaendert = any(a.action != CONFIRM for a in offen["actions"])
        self.judgments = nachher
        bewertung = self.evaluation()
        if geaendert:
            self.events.append(self._weights_event(nummer, bewertung))
        self.cycles.append({
            "cycle": nummer,
            "before": [c.to_dict() for c in offen["before"]],
            "actions": [
                {"pair": [a.pair[0], a.pair[1]], "action": a.action, "preference": a.preference,
                 "intensity": a.intensity, "reason": a.reason.strip()}
                for a in offen["actions"]
            ],
            "after": [c.to_dict() for c in nachher],
            "evaluation": bewertung,
            "hints": hinweise,
        })
        self.hints += hinweise
        self._offen = None

    def _bewerte(self, judgments: Sequence[Comparison]) -> dict[str, Any]:
        return ahp.evaluation(ahp.compute(judgments, self.criteria), self.config)

    def _weights_event(self, nummer: int, bewertung: dict[str, Any]) -> dict[str, Any]:
        return {
            "actor": "code",
            "event_type": "WeightsComputed",
            "gate": GATE,
            "cycle": nummer,
            "weights": bewertung["weights"],
            "cr": bewertung["cr"],
            "cr_threshold": bewertung["cr_threshold"],
            "consistent": bewertung["consistent"],
        }


def autonomous(
    case_id: str, criteria: Sequence[str], judgments: Sequence[Comparison], config: Mapping[str, Any], repair: Any = None
) -> Gate2Result:
    """Autonomer Modus: alles confirm in Zyklus 1, Entscheidung accept. Kein
    Modellaufruf."""
    sitzung = Gate2Session(case_id, criteria, judgments, config, mode=AUTONOMOUS, repair=repair)
    sitzung.submit([JudgmentAction((c.a, c.b), CONFIRM) for c in sitzung.judgments])
    return sitzung.decide(ACCEPT)


def judgments_from_response(response: Mapping[str, Any]) -> list[Comparison]:
    """Urteile aus einer gültigen Antwort von Schritt 5 (Ergebnis-JSON)."""
    if not response.get("valid"):
        raise GateError(f"Fall {response.get('case_id')}: ungültige Antwort, kommt nicht in Gate 2")
    return [
        Comparison(c["a"], c["b"], c["preference"], c["intensity"], c.get("justification"),
                   tuple(c["evidence"]) if c.get("evidence") is not None else None)
        for c in response["comparisons"]
    ]
