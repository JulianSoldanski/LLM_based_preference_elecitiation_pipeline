"""Tests der Zuordnung ohne Netzzugriff (EchoClient statt API)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent import EchoClient
from assignment import (
    CONFIG_DIR,
    CRITERIA_FILE,
    PASSAGES_FILE,
    PROMPT_FILE,
    STATUS_FAILED,
    STATUS_OK,
    assign,
    load_criteria,
    prepare,
    run_assignment,
)
from assignment_eval import evaluate, evaluate_file
from prompt_builder import ConfigError, PromptBuilder

CODES = ["DR", "FR", "CT"]


def antwort(criteria: list[str], passage_id: str = "P01") -> str:
    return json.dumps({"passage_id": passage_id, "justification": "…", "criteria": criteria})


class ZuordnungTest(unittest.TestCase):
    def zuordnen(self, *antworten: str, **kwargs) -> tuple[dict, EchoClient]:
        client = EchoClient(responses=antworten, **kwargs)
        schema = {"type": "object"}
        return assign(client, "system", "user", "P01", CODES, schema, 0.0), client

    def test_gueltige_antwort(self) -> None:
        ergebnis, client = self.zuordnen(antwort(["DR"]))
        self.assertEqual(ergebnis["status"], STATUS_OK)
        self.assertEqual(ergebnis["criteria"], ["DR"])
        self.assertEqual(len(client.calls), 1)

    def test_unbekannter_code(self) -> None:
        ergebnis, client = self.zuordnen(antwort(["QU"]))
        self.assertEqual(ergebnis["status"], STATUS_FAILED)
        self.assertIsNone(ergebnis["criteria"])
        self.assertIn("QU", ergebnis["calls"][-1]["errors"][0])
        self.assertEqual(len(client.calls), 2)

    def test_leere_liste_ist_gueltig(self) -> None:
        ergebnis, _ = self.zuordnen(antwort([]))
        self.assertEqual(ergebnis["status"], STATUS_OK)
        self.assertEqual(ergebnis["criteria"], [])

    def test_zwei_kriterien(self) -> None:
        ergebnis, _ = self.zuordnen(antwort(["DR", "CT"]))
        self.assertEqual(ergebnis["criteria"], ["DR", "CT"])

    def test_ungueltiges_json_wird_einmal_identisch_wiederholt(self) -> None:
        ergebnis, client = self.zuordnen("Das ist DR.", antwort(["DR"]))
        self.assertEqual(ergebnis["status"], STATUS_OK)
        self.assertEqual(len(ergebnis["calls"]), 2)
        self.assertIn("kein gültiges JSON", ergebnis["calls"][0]["errors"][0])
        self.assertEqual(client.calls[0], client.calls[1])

    def test_failed_nach_wiederholung_behaelt_beide_rohantworten(self) -> None:
        ergebnis, client = self.zuordnen("kaputt")
        self.assertEqual(ergebnis["status"], STATUS_FAILED)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual([c["raw"] for c in ergebnis["calls"]], ["kaputt", "kaputt"])

    def test_falsche_passage_id(self) -> None:
        ergebnis, _ = self.zuordnen(antwort(["DR"], passage_id="P02"))
        self.assertEqual(ergebnis["status"], STATUS_FAILED)
        self.assertIn("passage_id", ergebnis["calls"][0]["errors"][0])

    def test_client_fehler_zaehlt_als_versuch(self) -> None:
        ergebnis, client = self.zuordnen(fail_with=TimeoutError("weg"))
        self.assertEqual(ergebnis["status"], STATUS_FAILED)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("TimeoutError", ergebnis["calls"][0]["errors"][0])


class EingabeTest(unittest.TestCase):
    def test_prompt_ohne_fallkontext_und_mit_wortlaut(self) -> None:
        eingabe = prepare(PASSAGES_FILE, CRITERIA_FILE, PROMPT_FILE)
        alles = eingabe["system"] + "".join(eingabe["user"].values())
        self.assertNotIn("Bike Inc.", alles)
        self.assertIn("- DR (Delivery reliability): delivery performance", eingabe["system"])
        self.assertEqual(eingabe["schema"]["properties"]["criteria"]["items"]["enum"], CODES)
        # Wortlaut der Passagen identisch zu den Aussagen im Paarvergleich.
        aussagen = {text for case in PromptBuilder.from_paths().cases for text in case.statements.values()}
        self.assertEqual({p["text"] for p in eingabe["passages"]}, aussagen)
        self.assertIn(f'Passage: """{eingabe["passages"][0]["text"]}"""', eingabe["user"]["P01"])

    def test_leerer_katalog_bricht_ab(self) -> None:
        with self.assertRaises(ConfigError):
            load_criteria(CONFIG_DIR / "criteria_scor_current.yaml")


class DurchlaufTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.passagen = Path(self.tmp.name) / "passagen.csv"
        self.passagen.write_text(
            "passage_id,source,text,expected_criteria\n"
            "A,evaluation,Lieferung kam zu spät.,DR\n"
            "B,evaluation,Teuer und spät.,DR;CT\n"
            "C,evaluation,Kein Bezug.,-\n",
            encoding="utf-8",
        )
        antworten = [
            json.dumps({"passage_id": "A", "justification": "spät", "criteria": ["DR"]}),
            json.dumps({"passage_id": "B", "justification": "spät", "criteria": ["DR"]}),
            "kaputt",
            "kaputt",
        ]
        json_path, self.log_path = run_assignment(
            EchoClient(responses=antworten),
            passages_path=self.passagen,
            runs=2,
            out_dir=Path(self.tmp.name),
            quiet=True,
        )
        self.dokument = json.loads(json_path.read_text(encoding="utf-8"))
        self.json_path = json_path

    def test_ein_datensatz_je_passage_und_lauf(self) -> None:
        records = self.dokument["records"]
        self.assertEqual(len(records), 6)
        self.assertEqual([r["status"] for r in records[:3]], [STATUS_OK, STATUS_OK, STATUS_FAILED])
        self.assertNotIn("expected_criteria", records[0])

    def test_metadaten_und_fehlende_kriterien(self) -> None:
        meta = self.dokument["meta"]
        self.assertEqual(meta["prompt_version"], "assignment_v1")
        self.assertEqual(len(meta["criteria_sha256"]), 64)
        self.assertEqual(meta["criteria"], CODES)
        self.assertEqual(self.dokument["summary"][0]["missing_criteria"], ["FR", "CT"])
        self.assertEqual(self.dokument["ground_truth"], {"A": ["DR"], "B": ["DR", "CT"], "C": []})
        self.assertIn("Rohantwort: 'kaputt'", self.log_path.read_text(encoding="utf-8"))

    def test_auswertung(self) -> None:
        lauf = evaluate(self.dokument)[0]
        self.assertAlmostEqual(lauf["exact_match"], 1 / 3)  # A richtig, B unvollständig, C failed
        self.assertEqual(lauf["failed"], 1)
        self.assertEqual(lauf["criteria"]["DR"], {"precision": 1.0, "recall": 1.0, "tp": 2, "fp": 0, "fn": 0})
        self.assertEqual(lauf["criteria"]["CT"]["recall"], 0.0)
        self.assertIsNone(lauf["criteria"]["FR"]["precision"])
        eval_path, tabelle = evaluate_file(self.json_path)
        self.assertTrue(eval_path.is_file())
        self.assertIn("Exact Match 0.33", tabelle)


if __name__ == "__main__":
    unittest.main(verbosity=2)
