# LLM-basierte Präferenzerhebung (AHP-Paarvergleiche)

Erhebungswerkzeug für die Masterarbeit zur Frage, inwieweit LLM-generierte
AHP-Gewichte von menschlichen Gewichten abweichen, wenn beide aus identischen
natürlichsprachlichen Beschreibungen von Lieferantenerfahrung abgeleitet werden.

Ein LLM-Agent bearbeitet dieselbe Aufgabe wie die Probanden im
SoSci-Fragebogen: Er liest pro Fall drei Aussagen (eine je Kriterium) und füllt
drei Paarvergleiche aus — zuerst die Richtung (Kriterium A, Kriterium B oder
`gleich`), bei gewählter Richtung zusätzlich die Intensität auf der Saaty-Skala.

Die Erhebung läuft in zwei Bedingungen, siehe
[Erhebungsbedingungen](#erhebungsbedingungen).

## Umfang

Das Projekt **erhebt ausschließlich Rohurteile** und schreibt sie als JSONL.
Es berechnet keine Prioritätsvektoren, keine Vergleichsmatrix und kein
Konsistenzmaß. Aus dem Fragebogen liegen ebenfalls nur Rohurteile vor; die
Umrechnung muss für beide Datenquellen mit demselben, noch festzulegenden
Verfahren erfolgen und gehört deshalb in einen separaten Auswertungsschritt
(siehe [Nächste Schritte](#nächste-schritte)).

## Erhebungsbedingungen

| | `history` (Standard, Hauptbedingung) | `stateless` (Kontrolle) |
| --- | --- | --- |
| Aufrufe | alle Fälle in **einer** Konversation | ein Fall pro Aufruf |
| Kontext | vorherige Falltexte und eigene Antworten im Verlauf | frischer Kontext |
| Reihenfolge | je Durchlauf neu permutiert (`--seed`) | konfigurierte Reihenfolge |
| Methode | `AhpAgent.judge_sequence()` | `AhpAgent.judge()` |

**Warum History die Hauptbedingung ist.** Die Probanden bearbeiten die drei
Fälle nacheinander und erinnern sich dabei an vorherige Fälle. Die
Fragebogeninstruktion sagt ihnen deshalb ausdrücklich, dass die Fälle unabhängig
voneinander sind. Im zustandslosen Aufbau hat dieser Satz auf Modellseite keinen
Adressaten — es gibt nichts, wovon abzugrenzen wäre. Mit Verlauf bekommt er
einen, und die beiden Gruppen liegen näher beieinander.

**Warum die zustandslose Bedingung bleibt.** Der Verlauf gibt dem Modell
*wörtlichen* Zugriff auf frühere Falltexte und die eigenen Urteile. Menschliche
Erinnerung ist dagegen unvollständig und rekonstruktiv: Probanden haben den
vorherigen Fall nicht mehr vor Augen und erinnern ihn allenfalls sinngemäß. Die
History-Bedingung überschätzt den Übertragungseffekt also tendenziell, die
zustandslose Bedingung grenzt ihn nach unten ein. Beide zusammen klammern den
Bereich, in dem der menschliche Fall liegt.

**Was im Verlauf steht.** Die Rohantwort des Modells wird unverändert als
Assistant-Nachricht angehängt, auch wenn sie ungültig war — ein repariertes oder
weggelassenes Turn wäre nicht mehr der Kontext, den das Modell tatsächlich
gesehen hat. Bricht ein Aufruf mit einem Netzfehler ab, endet die Sequenz: die
bisherigen Antworten werden behalten, der fehlgeschlagene Fall wird als
ungültige Response festgehalten, die restlichen Fälle entfallen. Ein Weiterlaufen
mit lückenhaftem Verlauf wäre eine dritte, ungewollte Bedingung.

**Der Unabhängigkeitshinweis** steht einmal im System-Prompt (Feld
`scenario.independence_note` in `config/cases.yaml`), so wie er bei den Probanden
einmal in der Einleitung steht — er wird **nicht** vor jedem Fall wiederholt.
Eine Wiederholung würde die Instruktion für das Modell stärker machen als für
die Menschen. Solange das Feld leer ist, erscheint der Satz nicht im Prompt und
`runner.py`/`experiment.py` warnen in der History-Bedingung.

**Die Fallreihenfolge** wird in der History-Bedingung je Durchlauf neu
permutiert, weil sie dort — anders als im zustandslosen Fall — Bedeutung hat:
jeder Fall ist Kontext des nächsten. `--seed` macht das reproduzierbar; ohne
Angabe wird ein Seed gezogen und in den Metadaten festgehalten. Über `position`
und `sequence` in jeder Zeile lässt sich hinterher prüfen, ob die Position das
Urteil beeinflusst hat.

## Architektur

```
config/     criteria.yaml, scale.yaml, cases.yaml   Wissensbasis (Fachinhalt)
prompts/    system.txt, user.txt                    Vorlagen mit {{platzhaltern}}
src/        prompt_builder.py                       lädt + validiert + rendert
            agent.py                                Client-Schicht + AhpAgent
            schema.py                               Datenmodell + Validierung
            runner.py                               CLI, JSONL-Ausgabe
            experiment.py                           dokumentierter Durchlauf
            test_pipeline.py                        Tests ohne Netzzugriff
results/    <provider>_<model>_<condition>.jsonl    Rohdaten (runner.py)
            <stempel>_<provider>_<model>_<cond>.json  Ergebnisse (experiment.py)
            <stempel>_<provider>_<model>_<cond>.log   Protokoll  (experiment.py)
.env        OPENAI_API_KEY, ANTHROPIC_API_KEY       Zugangsdaten (nicht im Git)
```

Leitprinzip: **Der Agent kennt keine Fachinhalte.** Kriterien, Skala und
Fallmaterial liegen in `config/`, die Formulierung der Aufgabe in `prompts/`.
Der `PromptBuilder` setzt beides zusammen und übergibt dem Agenten fertige
Prompts. Ein weiteres Kriterium, ein weiterer Fall oder eine andere
Skalenvariante erfordert deshalb keine Änderung an `agent.py`.

Datenfluss:

```
config/*.yaml ─┐
               ├─> PromptBuilder ─> (system, user) ─> LLMClient ─> Rohtext
prompts/*.txt ─┘                                                     │
                                          schema.parse_response ─────┘
                                                    │
                                          Response ─> results/*.jsonl
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # pyyaml + openai
pip install anthropic                    # nur für --provider anthropic
```

Die API-Schlüssel liegen in `.env` im Projektverzeichnis (per `.gitignore`
ausgeschlossen, Vorlage in `.env.example`):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...      # nur für --provider anthropic
```

Die Datei wird beim Erzeugen eines Clients automatisch eingelesen; ein
zusätzliches Paket ist dafür nicht nötig. Bereits gesetzte Umgebungsvariablen
haben Vorrang, ein `export OPENAI_API_KEY=...` in der Shell übersteuert die
Datei also gezielt. Fehlt der Schlüssel, bricht der Lauf mit einer klaren
Meldung ab, bevor ein Modell aufgerufen wird.

## Nutzung

Es gibt zwei Einstiegspunkte mit derselben Erhebungslogik: `runner.py`
schreibt schlank JSONL, `experiment.py` dokumentiert einen Durchlauf zusätzlich
in einer Logdatei und bündelt die Ergebnisse in einer JSON-Datei.

```bash
# Alle Prompts ansehen, ohne ein Modell aufzurufen
python3 src/runner.py --dry-run

# Erhebung in der Hauptbedingung: drei Fälle in einer Konversation, 5 Durchläufe
python3 src/runner.py --model gpt-4o --runs 5 --seed 42

# Kontrollbedingung: jeder Fall in eigenem Kontext
python3 src/runner.py --model gpt-4o --runs 5 --condition stateless

# Anderer Anbieter, eigener Ausgabepfad, nur ein Fall
python3 src/runner.py --provider anthropic --model claude-opus-4-5 \
        --case ketten --temperature 0.7 --out results/pilot.jsonl

# Dokumentierter Durchlauf: 10 Wiederholungen über alle Fälle
python3 src/experiment.py --model gpt-4o --runs 10 --seed 42

# Tests (kein Netzzugriff nötig)
python3 src/test_pipeline.py
```

| Option | Bedeutung |
| --- | --- |
| `--provider` | `openai` (Default) oder `anthropic` |
| `--condition` | `history` (Default) oder `stateless` |
| `--seed` | Seed der Fallreihenfolge; nur in `history` wirksam |
| `--model` | Modell-ID des Anbieters. Pflicht außer bei `--dry-run` |
| `--runs` | Wiederholungen je Fall (Default 1) |
| `--temperature` | Default 0.0 |
| `--case` | Fall-ID, mehrfach angebbar. Ohne Angabe alle Fälle |
| `--out` | Zieldatei. Default `results/<provider>_<model>_<condition>.jsonl` |
| `--dry-run` | Gibt System- und User-Prompts aus, ruft kein Modell auf |

Die Ausgabe ist **append-only** und wird nach jeder Zeile geflusht: ein
abgebrochener Lauf vernichtet keine bereits erhobenen Urteile, und ein
späterer Lauf ergänzt dieselbe Datei.

## Ausgabeformat

### `runner.py`: JSONL

Eine Zeile je Fall und Wiederholung:

```json
{
  "timestamp": "2026-09-04T09:45:37+00:00",
  "provider": "openai",
  "model": "gpt-4o",
  "temperature": 0.0,
  "condition": "history",
  "seed": 42,
  "case_id": "ketten",
  "run": 1,
  "position": 1,
  "sequence": ["batteriepacks", "ketten", "konnektivitaetsmodule"],
  "valid": true,
  "comparisons": [
    {"a": "flexibilitaet", "b": "liefertreue", "preference": "liefertreue", "intensity": 6},
    {"a": "kosten", "b": "flexibilitaet", "preference": "kosten", "intensity": 4},
    {"a": "liefertreue", "b": "kosten", "preference": "gleich", "intensity": 0}
  ],
  "errors": [],
  "raw": "{\"comparisons\": [ ... ]}"
}
```

* `condition` — `"history"` oder `"stateless"`.
* `position` / `sequence` — Stellung des Falls im Durchlauf (0-basiert) und die
  vollständige Fallreihenfolge dieses Durchlaufs. In der zustandslosen Bedingung
  `null`, weil es dort keinen Durchlauf mit Reihenfolge gibt.
* `seed` — Seed der Permutation; `null` ohne Verlauf.
* `preference` — Kennung des bevorzugten Kriteriums oder `"gleich"`.
* `intensity` — bei `"gleich"` `0` oder `1` (beide kodieren Indifferenz),
  sonst ganzzahlig `2`–`9`. Für die Auswertung zählt bei Indifferenz die
  `preference`, nicht der Zahlenwert.
* `valid` — `false`, sobald `errors` nicht leer ist. Ungültige Antworten werden
  **nicht repariert**; sie bleiben mit Fehlerliste und Rohtext in den Daten.
* `comparisons` — enthält nur Vergleiche, die vollständig valide sind. Bei
  einer fehlerhaften Antwort ist die Liste entsprechend kürzer oder leer; der
  vollständige Modelloutput steht immer in `raw`.
* `errors` — Freitext auf Deutsch, z.B. `Vergleich 1: Intensität 12 liegt
  außerhalb von 2-9`.

### `experiment.py`: JSON plus Protokoll

Ein Lauf erzeugt zwei Dateien mit gemeinsamem Zeitstempel im Namen, z.B.
`results/20260904-134716_openai_gpt-4o.json` und `.log`.

Die JSON-Datei bündelt den gesamten Lauf:

```json
{
  "meta": {
    "started": "2026-09-04T11:47:16+00:00",
    "finished": "2026-09-04T11:52:03+00:00",
    "provider": "openai", "model": "gpt-4o", "temperature": 0.0,
    "condition": "history", "seed": 42,
    "runs": 10, "cases": ["ketten", "batteriepacks", "konnektivitaetsmodule"],
    "sequences": [["kosten…"], "… eine Fallreihenfolge je Durchlauf"]
  },
  "prompts": { "system": "...", "user": { "ketten": "...", "...": "..." } },
  "summary": { "total": 30, "valid": 29, "invalid": 1 },
  "responses": [ { "…wie eine JSONL-Zeile…" } ]
}
```

Die Prompts stehen bewusst mit in der Datei: die sprachliche Eingabe ist Teil
des Messwerts, und eine spätere Änderung an `config/` oder `prompts/` darf
nicht dazu führen, dass sich ein alter Lauf nicht mehr rekonstruieren lässt.
Geschrieben wird nach jeder einzelnen Antwort über eine temporäre Datei mit
anschließendem Umbenennen — ein Abbruch hinterlässt also nie eine halbe JSON.

Die Logdatei protokolliert denselben Lauf im Klartext: Kopfzeile mit
Konfiguration, danach der vollständige System-Prompt und je ein User-Prompt pro
Fall, anschließend jede Durchführung einzeln.

```
2026-09-04 13:47:16 | Start 2026-09-04T11:47:16+00:00 | provider=openai model=gpt-4o temperature=0.0 runs=10 cases=ketten, batteriepacks, konnektivitaetsmodule
2026-09-04 13:47:16 | ---- SYSTEM-PROMPT ----
Du beurteilst, wie wichtig die folgenden Kriterien bei der Auswahl eines
...
2026-09-04 13:47:19 | [Lauf 1/10 | ketten] gültig
2026-09-04 13:47:19 |   1. flexibilitaet vs. liefertreue -> liefertreue (6)
  2. kosten vs. flexibilitaet -> kosten (4)
  3. liefertreue vs. kosten -> gleich (0)
  Rohantwort: '{"comparisons": [...]}'
2026-09-04 13:47:22 | [Lauf 1/10 | batteriepacks] UNGÜLTIG
2026-09-04 13:47:22 |   Fehler: Antwort ist kein gültiges JSON (Expecting value, Zeile 1)
  Rohantwort: 'Liefertreue ist wichtiger.'
```

Auf der Konsole erscheinen nur die Fortschrittszeilen; Prompts und Rohantworten
stehen ausschließlich in der Datei. `--quiet` unterdrückt die Konsolenausgabe
ganz, `--out-dir` verlegt beide Dateien.

## Designentscheidungen

**Keine Auswertung im Projekt.** Kein `ahp.py`, keine Gewichtung, kein CR.
Fragebogen- und LLM-Daten müssen mit identischem Verfahren umgerechnet werden;
eine hier eingebaute Berechnung würde diesen Schritt vorwegnehmen und den
Vergleich der beiden Gruppen methodisch schwächen.

**Wortlaut in `config/` ist unantastbar.** Kriterienbeschreibungen und
Fallaussagen entsprechen exakt dem Fragebogen, inklusive Umlauten,
Gedankenstrichen und Interpunktion. Die Forschungsfrage verlangt identische
sprachliche Eingaben für beide Gruppen; jede stille Umformulierung verletzt die
Kernbedingung des Designs. Ein Test prüft deshalb, dass die Umlaute im
gerenderten Prompt intakt sind und die erste Aussage je Fall stimmt.

**Aussagen mit Kriteriumslabel.** Im User-Prompt steht vor jeder Aussage das
Kriterium, auf das sie sich bezieht. Die Zuordnung ist damit vorgegeben und
nicht Teil der Aufgabe — wie im Fragebogen, wo die Probanden die Aussage
ebenfalls unter dem jeweiligen Kriterium gesehen haben. Zu beurteilen ist
allein die relative Wichtigkeit. Die `statement_order` je Fall rotiert (Ketten:
Liefertreue zuerst, Batteriepacks: Flexibilität, Konnektivitätsmodule: Kosten),
damit jedes Kriterium einmal an erster Stelle steht — identisch zum Fragebogen.
Soll die Darstellung geändert werden, geschieht das in `prompts/user.txt`, nicht
im Code.

**Kein Studienkontext im System-Prompt.** Der Prompt erwähnt weder Masterarbeit
noch KI-gestützte Entscheidungsunterstützung. Ein Modell, das weiß, dass sein
Output als KI-Antwort untersucht wird, kann sein Verhalten ändern.

**Keine Begründung angefordert.** Der System-Prompt verlangt reines JSON ohne
Fließtext. Probanden begründen ebenfalls nicht, und Chain-of-Thought verändert
AHP-Urteile messbar.

**Indifferenz über die Richtung, nicht über die Skala.** Wie im Fragebogen wird
Gleichwertigkeit über die Richtungsfrage erfasst (`preference: "gleich"`); bei
gewählter Richtung ist der gültige Bereich 2–9. Der Skalenwert 1 wäre sonst
doppelt belegt — einmal als "gleich wichtig", einmal als schwächste Präferenz.
Die Konstanten liegen in `schema.py` und werden über Platzhalter in den
System-Prompt gerendert, damit Prompt und Validierung nicht auseinanderlaufen
können.

Bei `"gleich"` gelten `0` **und** `1` als gültig. Der Prompt fragt die `0` ab,
die Skalentabelle nennt daneben die `1` als "gleich wichtig", und Modelle
greifen erwartbar darauf zurück. Beide Werte kodieren dieselbe Aussage; eine
formale Ablehnung würde ausgerechnet Indifferenz-Urteile systematisch aus den
Daten entfernen und die Verteilung verzerren — ein Ausfall, der nicht zufällig
über die Urteilstypen streut, sondern genau einen trifft. Ab `2` bleibt die
Angabe ungültig, weil Gleichwertigkeit mit Stärkegrad widersprüchlich ist.

**Strenge Validierung, keine Reparatur.** Toleriert wird nur ein umschließender
Codefence. Inhaltliche Abweichungen — Präferenz außerhalb des Paares,
Intensität außerhalb des Bereichs, Intensität ≠ 0 bei `gleich`, falsche
Paarreihenfolge, zu wenige Vergleiche, Freitext statt JSON — werden als Fehler
markiert und nicht korrigiert. Eine stillschweigend geheilte Antwort wäre kein
Messwert mehr. Die Ausfallrate ist damit selbst ein auswertbares Ergebnis.

**Zwei Bedingungen statt einer.** `judge()` (zustandslos) und
`judge_sequence()` (mit Verlauf) teilen sich Prompts, Validierung und
Ausgabeformat; unterschiedlich ist allein, ob ein Verlauf mitgeführt wird. Die
Begründung steht unter [Erhebungsbedingungen](#erhebungsbedingungen). Im Runner
liegen die Wiederholungen außen und die Fälle innen, damit ein Abbruch
vollständige Durchgänge über alle Fälle hinterlässt statt eines
überrepräsentierten Falls.

**Schmale Client-Schicht.** `LLMClient` ist ein `Protocol` mit `complete()`,
`model_id` und `provider`. Die SDKs werden erst in `__post_init__` importiert,
sodass nur das Paket des tatsächlich genutzten Anbieters installiert sein muss.
Ein weiterer Anbieter ist eine Dataclass mit einer Methode. Netzfehler werden
als ungültige Response zurückgegeben statt geworfen, damit ein einzelner
Ausfall den Lauf nicht beendet.

**Zwei Einstiegspunkte statt Schaltern.** `runner.py` erhebt und schreibt
JSONL, `experiment.py` erhebt und dokumentiert. Die Erhebungslogik liegt in
beiden Fällen im `AhpAgent`; `experiment.py` importiert `resolve_cases()` und
`model_slug()` aus `runner.py`, statt sie zu duplizieren. Ein einzelner
Einstiegspunkt mit `--log`-Schalter hätte die Argumentliste aufgebläht, und die
JSONL-Datei soll pro Modell fortschreibbar bleiben, während ein Experiment
bewusst je Lauf einen eigenen, abgeschlossenen Datensatz erzeugt.

**Getrennte Dateien je Modell und Bedingung.** Ohne `--out` schreibt der Runner
nach `results/<provider>_<model>_<condition>.jsonl`. Modelle und Bedingungen
landen so automatisch getrennt, ohne dass Zeilen nachträglich sortiert werden
müssen.

**Konfigurationsprüfung beim Start.** Der `PromptBuilder` prüft, dass jedes
Kriterium genau eine Aussage je Fall hat, dass `statement_order` eine
Permutation der Kriterien ist und dass `comparison_order` nur bekannte
Kriterien paart. Ein stiller Konfigurationsfehler würde Prompts erzeugen, die
nicht mehr dem Fragebogen entsprechen. Bewusst *nicht* erzwungen wird die
zyklische Balance der Vergleichsreihenfolge: sie ist bei drei Kriterien Teil des
Designs, wäre aber bei einer Erweiterung auf mehr Kriterien nicht mehr
herstellbar.

**`.env` ohne python-dotenv.** Der Loader in `agent.py` ist eine Schleife über
`NAME=wert`-Zeilen: Kommentare und Unsinn überspringen, Anführungszeichen
abstreifen, bereits gesetzte Variablen nicht überschreiben.
Für dieses Format lohnt keine weitere Abhängigkeit. Die Funktion gibt nur die
Namen der gesetzten Variablen zurück, damit Schlüssel nicht versehentlich in
Logs landen.

**Kleine Abhängigkeiten.** Nur `pyyaml` und das SDK des genutzten Anbieters.
Kein pandas, kein pydantic; Tests laufen mit `unittest` aus der Standardbibliothek.

## Tests

`python3 src/test_pipeline.py` — 53 Tests, alle ohne Netzzugriff über den
`EchoClient`. Abgedeckt sind Promptaufbau (Skalenquelle, Umlaute, erste Aussage
je Fall, Zuordnung der Aussagen zu den Kriterien), Konfigurationsprüfung,
gültige Antwort,
Codefence, Indifferenz, sämtliche Fehlerfälle, Netzfehler, Zustandslosigkeit,
das Einlesen von `.env`, die Serialisierung einer JSONL-Zeile sowie Protokoll
und JSON-Dokument eines dokumentierten Durchlaufs. Für die History-Bedingung
zusätzlich: Reihenfolge und Anzahl der Responses, `position`/`sequence`, das
Wachsen der Nachrichtenliste, eine ungültige Antwort mitten in der Sequenz, der
Abbruch bei Netzfehler und die Reproduzierbarkeit der Permutation bei gleichem
Seed.

## Nächste Schritte

* **Offen: Auswertungsschritt.** Die Umrechnung der Rohurteile in
  Prioritätsvektoren ist bewusst nicht Teil dieses Projekts und noch nicht
  festgelegt. Zu entscheiden sind unter anderem: Aggregationsverfahren
  (Eigenvektor vs. geometrisches Mittel), Umgang mit Inkonsistenz und mit
  ungültigen bzw. fehlenden Antworten sowie das Distanzmaß für den Vergleich
  von LLM- und Probandengewichten. Das Verfahren muss auf beide Datenquellen
  identisch angewendet werden und gehört in ein eigenes Modul, das `results/`
  und den Fragebogenexport liest.
* **Offen: Wortlaut des Unabhängigkeitshinweises.** `scenario.independence_note`
  in `config/cases.yaml` ist noch leer und muss wörtlich aus der
  Fragebogeninstruktion übernommen werden.
* Fragebogendaten aus SoSci in dasselbe Rohformat überführen; `condition` dort
  auf den menschlichen Vergleichswert setzen.
* Weitere Modelle und Anbieter ergänzen (jeweils eine Dataclass in `agent.py`).
