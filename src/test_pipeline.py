"""Tests ohne Netzzugriff.

Alle Modellaufrufe laufen über den EchoClient. Geprüft werden der Promptaufbau
(damit die sprachlichen Eingaben identisch zum Fragebogen bleiben) und die
Validierung der Antworten (damit ungültige Urteile nicht still in die Daten
laufen).
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent import AhpAgent, EchoClient, build_client, load_env, require_key
from experiment import run_experiment
from prompt_builder import ConfigError, PromptBuilder
from runner import case_sequences, default_out_path, resolve_seed
from schema import (
    CONDITION_HISTORY,
    CONDITION_STATELESS,
    Case,
    Criterion,
    parse_response,
)

GUELTIGE_ANTWORT = json.dumps(
    {
        "comparisons": [
            {"a": "flexibilitaet", "b": "liefertreue", "preference": "liefertreue", "intensity": 5},
            {"a": "kosten", "b": "flexibilitaet", "preference": "kosten", "intensity": 3},
            {"a": "liefertreue", "b": "kosten", "preference": "liefertreue", "intensity": 7},
        ]
    }
)


def antwort(*eintraege: tuple[str, object]) -> str:
    """Baut eine Antwort in der konfigurierten Paarreihenfolge."""
    paare = [("flexibilitaet", "liefertreue"), ("kosten", "flexibilitaet"), ("liefertreue", "kosten")]
    return json.dumps(
        {
            "comparisons": [
                {"a": a, "b": b, "preference": praeferenz, "intensity": intensitaet}
                for (a, b), (praeferenz, intensitaet) in zip(paare, eintraege)
            ]
        }
    )


class PromptAufbauTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = PromptBuilder.from_paths()

    def test_system_prompt_nennt_skalenquelle(self) -> None:
        system = self.builder.build_system()
        self.assertIn("Saaty (1980)", system)
        self.assertIn("Extrem viel wichtiger", system)

    def test_system_prompt_fordert_keine_begruendung(self) -> None:
        system = self.builder.build_system()
        self.assertIn("keine Begründung", system)

    def test_umlaute_bleiben_intakt(self) -> None:
        system = self.builder.build_system()
        for begriff in ("Flexibilität", "zuverlässig", "Qualität", "Gewährleistungsfälle"):
            self.assertIn(begriff, system)
        user = self.builder.build_user("konnektivitaetsmodule")
        self.assertIn("Konnektivitätsmodule", user)

    def test_erste_aussage_je_fall_stimmt(self) -> None:
        # Die Reihenfolge der Aussagen rotiert über die Fälle; sie muss exakt
        # der Reihenfolge im Fragebogen entsprechen.
        erwartet = {
            "ketten": "liefertreue",
            "batteriepacks": "flexibilitaet",
            "konnektivitaetsmodule": "kosten",
        }
        namen = {c.id: c.name for c in self.builder.criteria}
        for case_id, criterion_id in erwartet.items():
            case = self.builder.case(case_id)
            self.assertEqual(case.statement_order[0], criterion_id)
            user = self.builder.build_user(case_id)
            aussagen = [zeile[2:] for zeile in user.splitlines() if zeile.startswith("- ")]
            self.assertEqual(
                aussagen[0], f"{namen[criterion_id]}: {case.statements[criterion_id]}"
            )

    def test_aussagen_sind_kriterien_zugeordnet(self) -> None:
        # Die Zuordnung Aussage -> Kriterium ist vorgegeben, nicht Teil der Aufgabe.
        case = self.builder.case("ketten")
        user = self.builder.build_user("ketten")
        for criterion_id, text in case.statements.items():
            name = self.builder.criteria_by_id()[criterion_id].name
            self.assertIn(f"- {name}: {text}", user)

    def test_vergleichsreihenfolge_ist_konfiguriert(self) -> None:
        self.assertEqual(
            self.builder.expected_pairs(),
            [("flexibilitaet", "liefertreue"), ("kosten", "flexibilitaet"), ("liefertreue", "kosten")],
        )

    def test_criteria_by_name(self) -> None:
        self.assertEqual(self.builder.criteria_by_name()["Flexibilität"].id, "flexibilitaet")

    def test_unbekannter_fall(self) -> None:
        with self.assertRaises(KeyError):
            self.builder.build_user("gibtesnicht")


class KonfigurationsPruefungTest(unittest.TestCase):
    def test_fall_ohne_passende_aussagen_wird_abgelehnt(self) -> None:
        kriterien = [
            Criterion("a", "A", "Beschreibung A"),
            Criterion("b", "B", "Beschreibung B"),
        ]
        with self.assertRaises(ConfigError):
            PromptBuilder(
                criteria=kriterien,
                comparison_order=[("a", "b")],
                scale={"source": "Test", "levels": [{"values": "1", "label": "x", "explanation": "y"}]},
                scenario={"intro": "Intro"},
                cases=[
                    Case(
                        id="fall",
                        title="Fall",
                        component="Teil",
                        statement_order=("a",),
                        statements={"a": "Aussage"},  # Kriterium b fehlt
                    )
                ],
                system_template="{{criteria_block}}",
                user_template="{{statements_block}}",
            )


class AntwortValidierungTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = PromptBuilder.from_paths()

    def judge(self, rohantwort: str):
        client = EchoClient(responses=[rohantwort])
        agent = AhpAgent(builder=self.builder, client=client, temperature=0.0)
        return agent.judge("ketten", run=1)

    def test_gueltige_antwort(self) -> None:
        response = self.judge(GUELTIGE_ANTWORT)
        self.assertTrue(response.valid, response.errors)
        self.assertEqual(len(response.comparisons), 3)
        self.assertEqual(response.comparisons[0].preference, "liefertreue")
        self.assertEqual(response.comparisons[1].intensity, 3)

    def test_codefence_wird_toleriert(self) -> None:
        response = self.judge(f"```json\n{GUELTIGE_ANTWORT}\n```")
        self.assertTrue(response.valid, response.errors)

    def test_indifferenz(self) -> None:
        response = self.judge(antwort(("gleich", 0), ("kosten", 4), ("gleich", 0)))
        self.assertTrue(response.valid, response.errors)
        self.assertEqual(response.comparisons[0].preference, "gleich")
        self.assertEqual(response.comparisons[0].intensity, 0)

    def test_kriterium_gehoert_nicht_zum_paar(self) -> None:
        response = self.judge(antwort(("kosten", 4), ("kosten", 3), ("gleich", 0)))
        self.assertFalse(response.valid)
        self.assertIn("weder", response.errors[0])
        self.assertEqual(len(response.comparisons), 2)

    def test_intensitaet_ausserhalb_des_bereichs(self) -> None:
        for wert in (1, 10, 0):
            with self.subTest(wert=wert):
                response = self.judge(antwort(("liefertreue", wert), ("kosten", 3), ("gleich", 0)))
                self.assertFalse(response.valid)
                self.assertIn("außerhalb", response.errors[0])

    def test_intensitaet_bei_gleich(self) -> None:
        response = self.judge(antwort(("gleich", 3), ("kosten", 3), ("gleich", 0)))
        self.assertFalse(response.valid)
        self.assertIn("gleich", response.errors[0])

    def test_indifferenz_mit_skalenwert_eins(self) -> None:
        # Die Skalentabelle nennt 1 als "gleich wichtig"; das ist dieselbe
        # Aussage wie 0 und darf nicht als ungültig verworfen werden.
        response = self.judge(antwort(("gleich", 1), ("kosten", 3), ("gleich", 0)))
        self.assertTrue(response.valid, response.errors)
        self.assertEqual(response.comparisons[0].preference, "gleich")
        self.assertEqual(response.comparisons[0].intensity, 1)

    def test_unbekannte_praeferenz(self) -> None:
        response = self.judge(antwort(("beide", 2), ("kosten", 3), ("gleich", 0)))
        self.assertFalse(response.valid)
        self.assertIn("weder", response.errors[0])

    def test_kriteriumsname_statt_id(self) -> None:
        response = self.judge(antwort(("Liefertreue", 5), ("kosten", 3), ("gleich", 0)))
        self.assertFalse(response.valid)

    def test_freitext_statt_json(self) -> None:
        response = self.judge("Ich halte Liefertreue für deutlich wichtiger als Flexibilität.")
        self.assertFalse(response.valid)
        self.assertIn("kein gültiges JSON", response.errors[0])

    def test_zu_wenige_vergleiche(self) -> None:
        response = self.judge(antwort(("liefertreue", 5), ("kosten", 3)))
        self.assertFalse(response.valid)
        self.assertIn("Erwartet 3 Vergleiche, erhalten 2", response.errors)

    def test_falsche_paarreihenfolge(self) -> None:
        vertauscht = json.dumps(
            {
                "comparisons": [
                    {"a": "liefertreue", "b": "flexibilitaet", "preference": "liefertreue", "intensity": 5},
                    {"a": "kosten", "b": "flexibilitaet", "preference": "kosten", "intensity": 3},
                    {"a": "liefertreue", "b": "kosten", "preference": "gleich", "intensity": 0},
                ]
            }
        )
        response = self.judge(vertauscht)
        self.assertFalse(response.valid)
        self.assertIn("erwartetes Paar", response.errors[0])

    def test_leere_antwort(self) -> None:
        response = self.judge("")
        self.assertFalse(response.valid)
        self.assertEqual(response.errors, ["Leere Antwort"])

    def test_intensitaet_als_text(self) -> None:
        response = self.judge(antwort(("liefertreue", "5"), ("kosten", 3), ("gleich", 0)))
        self.assertFalse(response.valid)
        self.assertIn("keine ganze Zahl", response.errors[0])

    def test_netzfehler_wird_nicht_geworfen(self) -> None:
        client = EchoClient(fail_with=TimeoutError("Zeitüberschreitung"))
        agent = AhpAgent(builder=self.builder, client=client)
        response = agent.judge("ketten", run=2)
        self.assertFalse(response.valid)
        self.assertIn("Client-Fehler: TimeoutError", response.errors[0])
        self.assertEqual(response.raw, "")


class AgentVerhaltenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = PromptBuilder.from_paths()

    def test_zustandslos_pro_fall(self) -> None:
        # Jeder Aufruf schickt genau einen System- und einen User-Prompt,
        # ohne Verlauf früherer Fälle.
        client = EchoClient(responses=[GUELTIGE_ANTWORT])
        agent = AhpAgent(builder=self.builder, client=client)
        agent.judge("ketten", run=1)
        agent.judge("batteriepacks", run=1)

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["system"], client.calls[1]["system"])
        for aufruf in client.calls:
            self.assertEqual(len(aufruf["messages"]), 1)
            self.assertEqual(aufruf["messages"][0]["role"], "user")
        erster, zweiter = (a["messages"][0]["content"] for a in client.calls)
        self.assertNotEqual(erster, zweiter)
        self.assertNotIn(self.builder.case("ketten").component, zweiter)

    def test_zustandslose_response_traegt_die_bedingung(self) -> None:
        agent = AhpAgent(builder=self.builder, client=EchoClient(responses=[GUELTIGE_ANTWORT]))
        response = agent.judge("ketten", run=1)
        self.assertEqual(response.condition, CONDITION_STATELESS)
        self.assertIsNone(response.position)
        self.assertIsNone(response.sequence)
        self.assertIsNone(response.seed)

    def test_metadaten_in_der_response(self) -> None:
        agent = AhpAgent(
            builder=self.builder,
            client=EchoClient(model_id="echo-1", responses=[GUELTIGE_ANTWORT]),
            temperature=0.3,
        )
        response = agent.judge("ketten", run=4)
        self.assertEqual((response.provider, response.model), ("echo", "echo-1"))
        self.assertEqual(response.temperature, 0.3)
        self.assertEqual(response.run, 4)

    def test_build_client_kennt_echo(self) -> None:
        client = build_client("echo", "echo-1")
        self.assertEqual(client.provider, "echo")
        with self.assertRaises(ValueError):
            build_client("gibtesnicht", "x")


class SerialisierungTest(unittest.TestCase):
    def test_jsonl_zeile(self) -> None:
        builder = PromptBuilder.from_paths()
        agent = AhpAgent(builder=builder, client=EchoClient(responses=[GUELTIGE_ANTWORT]))
        zeile = agent.judge("ketten", run=1).to_json_line()

        self.assertNotIn("\n", zeile)
        daten = json.loads(zeile)
        self.assertEqual(
            set(daten),
            {
                "timestamp",
                "provider",
                "model",
                "temperature",
                "condition",
                "seed",
                "case_id",
                "run",
                "position",
                "sequence",
                "valid",
                "comparisons",
                "errors",
                "raw",
            },
        )
        self.assertTrue(daten["valid"])
        self.assertEqual(daten["case_id"], "ketten")
        self.assertEqual(len(daten["comparisons"]), 3)
        self.assertEqual(
            set(daten["comparisons"][0]), {"a", "b", "preference", "intensity"}
        )

    def test_umlaute_bleiben_unescaped(self) -> None:
        builder = PromptBuilder.from_paths()
        agent = AhpAgent(builder=builder, client=EchoClient(responses=["Flexibilität ist wichtiger."]))
        zeile = agent.judge("ketten", run=1).to_json_line()
        self.assertIn("Flexibilität", zeile)


class SequenzTest(unittest.TestCase):
    """History-Bedingung: alle Fälle in einer Konversation."""

    def setUp(self) -> None:
        self.builder = PromptBuilder.from_paths()
        self.folge = ["batteriepacks", "ketten", "konnektivitaetsmodule"]

    def agent(self, *antworten: str) -> tuple[AhpAgent, EchoClient]:
        client = EchoClient(model_id="echo-1", responses=antworten)
        return AhpAgent(builder=self.builder, client=client, temperature=0.0), client

    def test_sequenz_liefert_eine_response_je_fall_in_reihenfolge(self) -> None:
        agent, client = self.agent(GUELTIGE_ANTWORT)
        responses = agent.judge_sequence(self.folge, run=3)

        self.assertEqual([r.case_id for r in responses], self.folge)
        self.assertEqual(len(client.calls), 3)
        for response in responses:
            self.assertTrue(response.valid, response.errors)
            self.assertEqual(response.run, 3)

    def test_position_und_sequence_sind_gesetzt(self) -> None:
        agent, _ = self.agent(GUELTIGE_ANTWORT)
        responses = agent.judge_sequence(self.folge, run=1, seed=99)

        self.assertEqual([r.position for r in responses], [0, 1, 2])
        for response in responses:
            self.assertEqual(response.condition, CONDITION_HISTORY)
            self.assertEqual(response.sequence, self.folge)
            self.assertEqual(response.seed, 99)

    def test_verlauf_waechst_mit_jedem_fall(self) -> None:
        # Vor Fall n stehen n-1 vollständige Paare plus der neue User-Prompt.
        agent, client = self.agent(GUELTIGE_ANTWORT)
        agent.judge_sequence(self.folge, run=1)

        for index, aufruf in enumerate(client.calls):
            rollen = [nachricht["role"] for nachricht in aufruf["messages"]]
            self.assertEqual(len(rollen), 2 * index + 1)
            self.assertEqual(rollen, ["user", "assistant"] * index + ["user"])
            self.assertEqual(rollen.count("user"), index + 1)
            self.assertEqual(rollen.count("assistant"), index)

        # Die Falltexte stehen wörtlich im Verlauf, in der übergebenen Reihenfolge.
        letzte = client.calls[-1]["messages"]
        user_inhalte = [m["content"] for m in letzte if m["role"] == "user"]
        for case_id, inhalt in zip(self.folge, user_inhalte):
            self.assertEqual(inhalt, self.builder.build_user(case_id))

    def test_systemprompt_bleibt_ueber_die_sequenz_gleich(self) -> None:
        agent, client = self.agent(GUELTIGE_ANTWORT)
        agent.judge_sequence(self.folge, run=1)
        system = {aufruf["system"] for aufruf in client.calls}
        self.assertEqual(len(system), 1)

    def test_ungueltige_antwort_bricht_die_sequenz_nicht_ab(self) -> None:
        agent, client = self.agent(GUELTIGE_ANTWORT, "kein JSON", GUELTIGE_ANTWORT)
        responses = agent.judge_sequence(self.folge, run=1)

        self.assertEqual([r.valid for r in responses], [True, False, True])
        self.assertEqual([r.case_id for r in responses], self.folge)

        # Die ungültige Rohantwort steht unverändert im Verlauf: nur so ist es
        # der Kontext, den das Modell tatsächlich gesehen hat.
        verlauf = client.calls[-1]["messages"]
        assistants = [m["content"] for m in verlauf if m["role"] == "assistant"]
        self.assertEqual(assistants, [GUELTIGE_ANTWORT, "kein JSON"])

    def test_netzfehler_bricht_die_sequenz_ab(self) -> None:
        client = EchoClient(
            responses=[GUELTIGE_ANTWORT],
            fail_with=TimeoutError("Zeitüberschreitung"),
            fail_after=1,  # erster Aufruf gelingt, zweiter scheitert
        )
        agent = AhpAgent(builder=self.builder, client=client)
        responses = agent.judge_sequence(self.folge, run=1)

        self.assertEqual(len(responses), 2)
        self.assertTrue(responses[0].valid)
        self.assertFalse(responses[1].valid)
        self.assertIn("Client-Fehler: TimeoutError", responses[1].errors[0])
        self.assertEqual(responses[1].position, 1)
        self.assertEqual(responses[1].condition, CONDITION_HISTORY)
        self.assertEqual(len(client.calls), 2)  # dritter Fall wird nicht mehr gestellt


class FallreihenfolgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case_ids = PromptBuilder.from_paths().case_ids()

    def test_gleicher_seed_gleiche_reihenfolge(self) -> None:
        erste = case_sequences(self.case_ids, 5, CONDITION_HISTORY, seed=42)
        zweite = case_sequences(self.case_ids, 5, CONDITION_HISTORY, seed=42)
        self.assertEqual(erste, zweite)
        for folge in erste:
            self.assertEqual(sorted(folge), sorted(self.case_ids))

    def test_ohne_verlauf_bleibt_die_konfigurierte_reihenfolge(self) -> None:
        folgen = case_sequences(self.case_ids, 3, CONDITION_STATELESS, seed=42)
        self.assertEqual(folgen, [self.case_ids] * 3)

    def test_seed_wird_nur_mit_verlauf_gezogen(self) -> None:
        self.assertEqual(resolve_seed(CONDITION_HISTORY, 7), 7)
        self.assertIsNone(resolve_seed(CONDITION_STATELESS, 7))
        gezogen = resolve_seed(CONDITION_HISTORY, None)
        self.assertIsInstance(gezogen, int)

    def test_ausgabepfad_trennt_die_bedingungen(self) -> None:
        historie = default_out_path("openai", "gpt-4o", CONDITION_HISTORY)
        zustandslos = default_out_path("openai", "gpt-4o", CONDITION_STATELESS)
        self.assertNotEqual(historie, zustandslos)
        self.assertTrue(historie.name.endswith("_history.jsonl"))


class ExperimentTest(unittest.TestCase):
    """Dokumentierter Durchlauf: Log und JSON, ohne Netzzugriff."""

    def setUp(self) -> None:
        self.builder = PromptBuilder.from_paths()
        verzeichnis = tempfile.TemporaryDirectory()
        self.addCleanup(verzeichnis.cleanup)
        self.out_dir = Path(verzeichnis.name)

    def durchlauf(
        self,
        *antworten: str,
        runs: int = 2,
        condition: str = CONDITION_HISTORY,
        seed: int | None = 1,
    ):
        client = EchoClient(model_id="echo-1", responses=antworten)
        json_path, log_path = run_experiment(
            self.builder,
            client,
            runs=runs,
            case_ids=self.builder.case_ids(),
            temperature=0.0,
            condition=condition,
            seed=seed,
            out_dir=self.out_dir,
            quiet=True,
        )
        return json_path, log_path

    def test_json_enthaelt_metadaten_prompts_und_antworten(self) -> None:
        json_path, log_path = self.durchlauf(GUELTIGE_ANTWORT)
        daten = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(set(daten), {"meta", "prompts", "summary", "responses"})
        self.assertEqual(daten["meta"]["runs"], 2)
        self.assertEqual(daten["meta"]["model"], "echo-1")
        self.assertIsNotNone(daten["meta"]["finished"])
        self.assertEqual(daten["meta"]["cases"], self.builder.case_ids())

        # Die sprachliche Eingabe gehört zum Messwert und wird mitgespeichert.
        self.assertIn("Saaty (1980)", daten["prompts"]["system"])
        self.assertEqual(set(daten["prompts"]["user"]), set(self.builder.case_ids()))

        self.assertEqual(len(daten["responses"]), 2 * len(self.builder.case_ids()))
        self.assertEqual(daten["summary"], {"total": 6, "valid": 6, "invalid": 0})
        self.assertEqual([r["run"] for r in daten["responses"]], [1, 1, 1, 2, 2, 2])
        self.assertTrue(log_path.is_file())

    def test_ungueltige_antworten_werden_gezaehlt(self) -> None:
        json_path, _ = self.durchlauf(GUELTIGE_ANTWORT, "kein JSON", runs=1)
        daten = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(daten["summary"], {"total": 3, "valid": 2, "invalid": 1})

    def test_log_enthaelt_prompts_und_jede_durchfuehrung(self) -> None:
        _, log_path = self.durchlauf(GUELTIGE_ANTWORT, "kein JSON", runs=1)
        protokoll = log_path.read_text(encoding="utf-8")

        self.assertIn("---- SYSTEM-PROMPT ----", protokoll)
        for case_id in self.builder.case_ids():
            self.assertIn(f"---- USER-PROMPT: {case_id} ----", protokoll)
            self.assertIn(f"[Lauf 1/1 | {case_id}]", protokoll)
        self.assertIn("Fehler: Antwort ist kein gültiges JSON", protokoll)
        self.assertIn("Rohantwort:", protokoll)
        self.assertIn("liefertreue (5)", protokoll)  # geparster Vergleich im Klartext

    def test_history_bedingung_wird_dokumentiert(self) -> None:
        json_path, log_path = self.durchlauf(GUELTIGE_ANTWORT, runs=2, seed=7)
        daten = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(daten["meta"]["condition"], CONDITION_HISTORY)
        self.assertEqual(daten["meta"]["seed"], 7)
        self.assertEqual(len(daten["meta"]["sequences"]), 2)
        for folge in daten["meta"]["sequences"]:
            self.assertEqual(sorted(folge), sorted(self.builder.case_ids()))

        for response in daten["responses"]:
            self.assertEqual(response["condition"], CONDITION_HISTORY)
            self.assertEqual(response["seed"], 7)
            self.assertIn(response["position"], (0, 1, 2))
            self.assertEqual(sorted(response["sequence"]), sorted(self.builder.case_ids()))

        # Die Reihenfolge jedes Laufs steht auch im Protokoll.
        protokoll = log_path.read_text(encoding="utf-8")
        self.assertIn("Lauf 1/2 Reihenfolge:", protokoll)
        self.assertIn("condition=history", protokoll)

    def test_zustandslose_bedingung_bleibt_ohne_verlauf(self) -> None:
        json_path, _ = self.durchlauf(
            GUELTIGE_ANTWORT, runs=1, condition=CONDITION_STATELESS, seed=7
        )
        daten = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(daten["meta"]["condition"], CONDITION_STATELESS)
        self.assertIsNone(daten["meta"]["seed"])
        self.assertEqual([f for f in daten["meta"]["sequences"]], [self.builder.case_ids()])
        for response in daten["responses"]:
            self.assertEqual(response["condition"], CONDITION_STATELESS)
            self.assertIsNone(response["position"])
            self.assertIsNone(response["sequence"])

    def test_dateiname_nennt_die_bedingung(self) -> None:
        json_path, _ = self.durchlauf(GUELTIGE_ANTWORT, runs=1, condition=CONDITION_STATELESS)
        self.assertTrue(json_path.stem.endswith("_stateless"), json_path.stem)

    def test_dateien_teilen_den_zeitstempel(self) -> None:
        json_path, log_path = self.durchlauf(GUELTIGE_ANTWORT, runs=1)
        self.assertEqual(json_path.stem, log_path.stem)
        self.assertIn("echo_echo-1", json_path.stem)
        # Keine halbfertige temporäre Datei zurückgelassen
        self.assertEqual(sorted(p.suffix for p in self.out_dir.iterdir()), [".json", ".log"])


class EnvLadenTest(unittest.TestCase):
    def schreibe_env(self, inhalt: str) -> Path:
        verzeichnis = tempfile.TemporaryDirectory()
        self.addCleanup(verzeichnis.cleanup)
        pfad = Path(verzeichnis.name) / ".env"
        pfad.write_text(inhalt, encoding="utf-8")
        return pfad

    def test_schluessel_wird_uebernommen(self) -> None:
        pfad = self.schreibe_env(
            "# Kommentar\n\nTEST_KEY_A=sk-abc\nexport TEST_KEY_B=\"sk-def\"\nkaputte_zeile\n"
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            gesetzt = load_env(pfad)
            self.assertEqual(sorted(gesetzt), ["TEST_KEY_A", "TEST_KEY_B"])
            self.assertEqual(os.environ["TEST_KEY_A"], "sk-abc")
            self.assertEqual(os.environ["TEST_KEY_B"], "sk-def")  # Anführungszeichen entfernt

    def test_umgebung_hat_vorrang(self) -> None:
        pfad = self.schreibe_env("TEST_KEY_A=aus-datei\n")
        with mock.patch.dict(os.environ, {"TEST_KEY_A": "aus-shell"}, clear=True):
            self.assertEqual(load_env(pfad), [])
            self.assertEqual(os.environ["TEST_KEY_A"], "aus-shell")

    def test_fehlende_datei_ist_kein_fehler(self) -> None:
        self.assertEqual(load_env(Path("/gibt/es/nicht/.env")), [])

    def test_require_key_meldet_fehlenden_schluessel(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as fehler:
                require_key("TEST_KEY_A")
            self.assertIn("TEST_KEY_A", str(fehler.exception))
            self.assertIn(".env", str(fehler.exception))


class ParserDirektTest(unittest.TestCase):
    def test_liste_statt_objekt(self) -> None:
        comparisons, errors = parse_response("[]", [("a", "b")])
        self.assertEqual(comparisons, [])
        self.assertEqual(errors, ["Antwort ist kein JSON-Objekt"])

    def test_feld_fehlt(self) -> None:
        comparisons, errors = parse_response('{"urteile": []}', [("a", "b")])
        self.assertEqual(comparisons, [])
        self.assertIn("comparisons", errors[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
