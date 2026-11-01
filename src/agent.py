"""Client-Schicht und Agent.

Der Agent kennt keine Fachinhalte: er holt Prompts vom PromptBuilder, schickt
sie an einen Client und lässt die Antwort von schema.py validieren. Die
Client-Schicht ist bewusst schmal gehalten, damit weitere Anbieter mit wenigen
Zeilen ergänzt werden können.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from prompt_builder import REVISION_PROMPT, ROOT, PromptBuilder
from schema import (
    CONDITION_HISTORY,
    CONDITION_STATELESS,
    Comparison,
    Response,
    parse_response,
)

# Eine Chat-Nachricht im Format der Anbieter-SDKs.
Message = dict[str, str]

ENV_FILE = ROOT / ".env"


@dataclass(frozen=True)
class Completion:
    """Antwort auf eine Anfrage mit JSON-Schema.

    `model` ist die Modell-ID aus der API-Antwort, also der tatsächlich genutzte
    Snapshot, nicht die angefragte Kennung.
    """

    text: str
    model: str


def load_env(path: Path = ENV_FILE) -> list[str]:
    """Übernimmt die API-Schlüssel aus .env in die Prozessumgebung.

    Bewusst ohne python-dotenv: für Zeilen der Form NAME=wert lohnt keine
    weitere Abhängigkeit. Bereits gesetzte Umgebungsvariablen haben Vorrang.
    Zurückgegeben werden nur die Namen, damit keine Schlüssel in Logs landen.
    """
    if not path.is_file():
        return []

    gesetzt: list[str] = []
    for zeile in path.read_text(encoding="utf-8").splitlines():
        name, trenner, wert = zeile.partition("=")
        name = name.strip().removeprefix("export ").strip()
        if not trenner or not name or name.startswith("#") or name in os.environ:
            continue
        os.environ[name] = wert.strip().strip("\"'")
        gesetzt.append(name)
    return gesetzt


def require_key(name: str) -> None:
    """Früher, verständlicher Abbruch statt eines SDK-internen Fehlers."""
    if not os.environ.get(name):
        raise RuntimeError(
            f"{name} ist nicht gesetzt. Schlüssel in {ENV_FILE} hinterlegen "
            f"(Vorlage: .env.example) oder als Umgebungsvariable exportieren."
        )


@runtime_checkable
class LLMClient(Protocol):
    """Minimale Schnittstelle, die ein Anbieter erfüllen muss."""

    provider: str
    model_id: str

    def complete(self, system: str, user: str, temperature: float) -> str:
        """Liefert den Rohtext der Modellantwort auf eine einzelne Frage."""
        ...

    def complete_chat(
        self, system: str, messages: list[Message], temperature: float
    ) -> str:
        """Wie complete(), aber mit vollständigem Gesprächsverlauf."""
        ...

    def complete_json(
        self, system: str, user: str, temperature: float, name: str, schema: dict[str, Any]
    ) -> Completion:
        """Wie complete(), die Antwort ist aber über die API an `schema` gebunden."""
        ...


@dataclass
class OpenAIClient:
    """Standard-Client. Das SDK wird erst hier importiert, damit das Projekt
    ohne installiertes openai-Paket lauffähig bleibt (z.B. für Anthropic
    oder Tests)."""

    model_id: str
    provider: str = "openai"
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        load_env()
        require_key("OPENAI_API_KEY")
        from openai import OpenAI

        self._client = OpenAI()

    def complete(self, system: str, user: str, temperature: float) -> str:
        return self.complete_chat(system, [{"role": "user", "content": user}], temperature)

    def complete_chat(self, system: str, messages: list[Message], temperature: float) -> str:
        completion = self._client.chat.completions.create(
            model=self.model_id,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, *messages],
        )
        return completion.choices[0].message.content or ""

    def complete_json(
        self, system: str, user: str, temperature: float, name: str, schema: dict[str, Any]
    ) -> Completion:
        completion = self._client.chat.completions.create(
            model=self.model_id,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        )
        return Completion(completion.choices[0].message.content or "", completion.model)


@dataclass
class AnthropicClient:
    """Analog zu OpenAIClient, SDK ebenfalls erst im __post_init__."""

    model_id: str
    provider: str = "anthropic"
    max_tokens: int = 1024
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        load_env()
        require_key("ANTHROPIC_API_KEY")
        from anthropic import Anthropic

        self._client = Anthropic()

    def complete(self, system: str, user: str, temperature: float) -> str:
        return self.complete_chat(system, [{"role": "user", "content": user}], temperature)

    def complete_chat(self, system: str, messages: list[Message], temperature: float) -> str:
        message = self._client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            # SDK 1.x kennt temperature nicht mehr, die API nimmt sie bei
            # älteren Modellen (z.B. Sonnet 4.6, Haiku 4.5) aber weiter an.
            # Neuere Modelle (Sonnet 5, Opus 4.7+) antworten mit 400, statt
            # still mit ihrem Default zu laufen und die Protokolle zu verfälschen.
            extra_body={"temperature": temperature},
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )

    def complete_json(
        self, system: str, user: str, temperature: float, name: str, schema: dict[str, Any]
    ) -> Completion:
        # Strukturierte Ausgabe erst ab Haiku 4.5; ältere Modelle antworten mit 400.
        message = self._client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
            extra_body={"temperature": temperature},
        )
        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        return Completion(text, message.model)


@dataclass
class MistralClient:
    """Analog zu OpenAIClient, SDK (mistralai>=2) ebenfalls erst im __post_init__."""

    model_id: str
    provider: str = "mistral"
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        load_env()
        require_key("MISTRAL_API_KEY")
        from mistralai.client import Mistral

        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def complete(self, system: str, user: str, temperature: float) -> str:
        return self.complete_chat(system, [{"role": "user", "content": user}], temperature)

    def complete_chat(self, system: str, messages: list[Message], temperature: float) -> str:
        completion = self._client.chat.complete(
            model=self.model_id,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, *messages],
        )
        content = completion.choices[0].message.content
        # Bei aktiviertem Reasoning kommt eine Liste von Chunks statt eines
        # Strings; übernommen wird wie bei Anthropic nur der Antworttext.
        if isinstance(content, list):
            return "".join(
                chunk.text for chunk in content if getattr(chunk, "type", "") == "text"
            )
        return content or ""

    def complete_json(
        self, system: str, user: str, temperature: float, name: str, schema: dict[str, Any]
    ) -> Completion:
        completion = self._client.chat.complete(
            model=self.model_id,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema_definition": schema},
            },
        )
        content = completion.choices[0].message.content
        if isinstance(content, list):
            content = "".join(
                chunk.text for chunk in content if getattr(chunk, "type", "") == "text"
            )
        return Completion(content or "", completion.model)


@dataclass
class EchoClient:
    """Testdouble ohne Netzzugriff.

    Gibt die vorbereiteten Antworten der Reihe nach zurück (danach zyklisch) und
    protokolliert jeden Aufruf samt vollständigem Nachrichtenverlauf, damit sich
    auch mehrstufige Läufe prüfen lassen. `fail_with` erzwingt einen Fehler,
    `fail_after` erst ab dem n-ten Aufruf — für Abbrüche mitten in einer Sequenz.
    """

    model_id: str = "echo"
    provider: str = "echo"
    responses: Sequence[str] = ()
    calls: list[dict[str, Any]] = field(default_factory=list)
    fail_with: Exception | None = None
    fail_after: int | None = None

    def complete(self, system: str, user: str, temperature: float) -> str:
        return self.complete_chat(system, [{"role": "user", "content": user}], temperature)

    def complete_chat(self, system: str, messages: list[Message], temperature: float) -> str:
        # Kopie: der Agent hängt an dieselbe Liste weitere Nachrichten an.
        self.calls.append(
            {
                "system": system,
                "messages": [dict(m) for m in messages],
                "temperature": temperature,
            }
        )
        if self.fail_with is not None and (
            self.fail_after is None or len(self.calls) > self.fail_after
        ):
            raise self.fail_with
        if not self.responses:
            return ""
        return self.responses[(len(self.calls) - 1) % len(self.responses)]

    def complete_json(
        self, system: str, user: str, temperature: float, name: str, schema: dict[str, Any]
    ) -> Completion:
        text = self.complete(system, user, temperature)
        self.calls[-1]["schema"] = schema
        return Completion(text, self.model_id)


def build_client(provider: str, model: str, **kwargs: Any) -> LLMClient:
    """Fabrik für die Client-Implementierungen."""
    key = provider.lower()
    if key == "openai":
        return OpenAIClient(model_id=model)
    if key == "anthropic":
        return AnthropicClient(model_id=model)
    if key == "mistral":
        return MistralClient(model_id=model)
    if key == "echo":
        # Nur für Tests und Trockenläufe, nicht über die CLI erreichbar.
        return EchoClient(model_id=model, **kwargs)
    raise ValueError(f"Unbekannter Anbieter: {provider!r}")


@dataclass(frozen=True)
class Revision:
    """Ein Revisionsaufruf in Gate 2: Prompt, Rohantwort und validierte Urteile
    nur für die zurückgegebenen Paare."""

    case_id: str
    pairs: tuple[tuple[str, str], ...]
    system: str
    user: str
    raw: str
    comparisons: tuple[Comparison, ...]
    errors: tuple[str, ...]
    prompt_version: str = REVISION_PROMPT.stem

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(f"{self.system}\n\n{self.user}".encode("utf-8")).hexdigest()


@dataclass
class AhpAgent:
    """Beurteilt Fälle in einer der beiden Erhebungsbedingungen.

    `judge()` ist die zustandslose Kontrollbedingung: ein Fall pro Aufruf,
    frischer Kontext, kein Gedächtnis über Fälle hinweg.

    `judge_sequence()` ist die History-Hauptbedingung: alle Fälle laufen in
    einer Konversation, wie bei Probanden, die die Fälle nacheinander
    bearbeiten und sich an vorherige erinnern.
    """

    builder: PromptBuilder
    client: LLMClient
    temperature: float = 0.0

    def judge(self, case_id: str, run: int = 1) -> Response:
        system, user = self.builder.build_prompts(case_id)

        try:
            raw = self.client.complete(system, user, self.temperature)
        except Exception as exc:  # Netzfehler beenden den Lauf nicht
            return self._response(
                case_id, run, raw="", errors=[f"Client-Fehler: {type(exc).__name__}: {exc}"]
            )

        comparisons, errors = self._parse(raw, case_id)
        return self._response(case_id, run, raw=raw, comparisons=comparisons, errors=errors)

    def judge_sequence(
        self,
        case_ids: Sequence[str],
        run: int = 1,
        *,
        seed: int | None = None,
        on_case: Callable[[str, Response | None], None] | None = None,
    ) -> list[Response]:
        """Arbeitet alle Fälle in einer Konversation ab.

        Der Unabhängigkeitshinweis wird bewusst nicht vor jedem Fall wiederholt:
        er steht einmal im System-Prompt, so wie er bei den Probanden einmal in
        der Einleitung steht. Eine Wiederholung würde die Instruktion für das
        Modell stärker machen als für die Menschen.

        on_case(case_id, None) vor jedem Aufruf und on_case(case_id, response)
        danach, nur für Fortschrittsanzeigen; am Verlauf ändert es nichts.
        """
        system = self.builder.build_system()
        sequence = list(case_ids)
        messages: list[Message] = []
        responses: list[Response] = []

        for position, case_id in enumerate(sequence):
            messages.append({"role": "user", "content": self.builder.build_user(case_id)})
            if on_case:
                on_case(case_id, None)
            meta = {
                "condition": CONDITION_HISTORY,
                "position": position,
                "sequence": sequence,
                "seed": seed,
            }

            try:
                raw = self.client.complete_chat(system, messages, self.temperature)
            except Exception as exc:
                # Abbruch statt Weiterlaufen: ohne die Antwort fehlt dem Verlauf
                # ein Turn, und die folgenden Fälle liefen in einem anderen
                # Kontext als dem, der gemessen werden soll.
                responses.append(
                    self._response(
                        case_id,
                        run,
                        raw="",
                        errors=[f"Client-Fehler: {type(exc).__name__}: {exc}"],
                        **meta,
                    )
                )
                if on_case:
                    on_case(case_id, responses[-1])
                break

            # Rohantwort unverändert in den Verlauf, auch wenn sie ungültig ist:
            # repariert oder weggelassen wäre es nicht mehr der Kontext, den das
            # Modell tatsächlich gesehen hat.
            messages.append({"role": "assistant", "content": raw})
            comparisons, errors = self._parse(raw, case_id)
            responses.append(
                self._response(case_id, run, raw=raw, comparisons=comparisons, errors=errors, **meta)
            )
            if on_case:
                on_case(case_id, responses[-1])

        return responses

    def revise(
        self,
        case_id: str,
        current: Sequence[Any],
        pairs: Sequence[tuple[str, str]],
        comments: dict[tuple[str, str], str] | None = None,
    ) -> Revision:
        """Gate 2: lässt nur `pairs` anhand der Aussagen neu beurteilen
        (prompts/revision_v1.md). Ein Aufruf; eine Wiederholung bei ungültiger
        Antwort übernimmt der Aufrufer. Der Prompt von Schritt 5 bleibt
        unverändert, Gewichte und CR stehen nicht im Revisionsprompt."""
        pairs = tuple(tuple(p) for p in pairs)
        system, user = self.builder.build_revision(case_id, current, list(pairs), comments)
        try:
            raw = self.client.complete(system, user, self.temperature)
        except Exception as exc:  # wie judge(): Netzfehler beenden nichts
            return Revision(case_id, pairs, system, user, "", (), (f"Client-Fehler: {type(exc).__name__}: {exc}",))
        comparisons, errors = parse_response(
            raw, list(pairs), self.builder.justification, self.builder.evidence_ids(case_id)
        )
        return Revision(case_id, pairs, system, user, raw, tuple(comparisons), tuple(errors))

    def _parse(self, raw: str, case_id: str) -> tuple[list[Comparison], list[str]]:
        return parse_response(
            raw, self.builder.expected_pairs(), self.builder.justification, self.builder.evidence_ids(case_id)
        )

    def _response(
        self,
        case_id: str,
        run: int,
        raw: str,
        comparisons: list[Comparison] | None = None,
        errors: list[str] | None = None,
        *,
        condition: str = CONDITION_STATELESS,
        position: int | None = None,
        sequence: list[str] | None = None,
        seed: int | None = None,
    ) -> Response:
        return Response(
            case_id=case_id,
            run=run,
            provider=self.client.provider,
            model=self.client.model_id,
            temperature=self.temperature,
            raw=raw,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            comparisons=comparisons or [],
            errors=errors or [],
            condition=condition,
            position=position,
            sequence=list(sequence) if sequence is not None else None,
            seed=seed,
        )
