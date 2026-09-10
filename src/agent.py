"""Client-Schicht und Agent.

Der Agent kennt keine Fachinhalte: er holt Prompts vom PromptBuilder, schickt
sie an einen Client und lässt die Antwort von schema.py validieren. Die
Client-Schicht ist bewusst schmal gehalten, damit weitere Anbieter mit wenigen
Zeilen ergänzt werden können.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from prompt_builder import ROOT, PromptBuilder
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
            temperature=temperature,
            system=system,
            messages=messages,
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )


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


def build_client(provider: str, model: str, **kwargs: Any) -> LLMClient:
    """Fabrik für die Client-Implementierungen."""
    key = provider.lower()
    if key == "openai":
        return OpenAIClient(model_id=model)
    if key == "anthropic":
        return AnthropicClient(model_id=model)
    if key == "echo":
        # Nur für Tests und Trockenläufe, nicht über die CLI erreichbar.
        return EchoClient(model_id=model, **kwargs)
    raise ValueError(f"Unbekannter Anbieter: {provider!r}")


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

        comparisons, errors = parse_response(raw, self.builder.expected_pairs())
        return self._response(case_id, run, raw=raw, comparisons=comparisons, errors=errors)

    def judge_sequence(
        self, case_ids: Sequence[str], run: int = 1, *, seed: int | None = None
    ) -> list[Response]:
        """Arbeitet alle Fälle in einer Konversation ab.

        Der Unabhängigkeitshinweis wird bewusst nicht vor jedem Fall wiederholt:
        er steht einmal im System-Prompt, so wie er bei den Probanden einmal in
        der Einleitung steht. Eine Wiederholung würde die Instruktion für das
        Modell stärker machen als für die Menschen.
        """
        system = self.builder.build_system()
        sequence = list(case_ids)
        messages: list[Message] = []
        responses: list[Response] = []

        for position, case_id in enumerate(sequence):
            messages.append({"role": "user", "content": self.builder.build_user(case_id)})
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
                break

            # Rohantwort unverändert in den Verlauf, auch wenn sie ungültig ist:
            # repariert oder weggelassen wäre es nicht mehr der Kontext, den das
            # Modell tatsächlich gesehen hat.
            messages.append({"role": "assistant", "content": raw})
            comparisons, errors = parse_response(raw, self.builder.expected_pairs())
            responses.append(
                self._response(case_id, run, raw=raw, comparisons=comparisons, errors=errors, **meta)
            )

        return responses

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
