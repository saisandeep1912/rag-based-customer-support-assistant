from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# logger = logging.getLogger(__name__)


class XAIClientError(RuntimeError):
    """Raised when the xAI API call fails."""


@dataclass(slots=True)
class XAIClient:
    api_key: str | None
    base_url: str
    model: str
    temperature: float
    max_output_tokens: int

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise XAIClientError(
                "XAI_API_KEY is not configured. Set it before sending queries."
            )

        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        request = Request(
            url=f"{self.base_url.rstrip('/')}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise XAIClientError(f"xAI API error {exc.code}: {details}") from exc
        except URLError as exc:
            raise XAIClientError(f"Unable to reach xAI API: {exc.reason}") from exc

        text = self._extract_text(body)
        return text

    def _extract_text(self, body: dict[str, Any]) -> str:
        output = body.get("output", [])
        parts: list[str] = []
        for item in output:
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    parts.append(content.get("text", ""))
        return "\n".join(part for part in parts if part).strip()
