from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentEvent:
    """Flat JSON event emitted by the voice-agent runtime."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("event type must not be empty")
        if "type" in self.data:
            raise ValueError("'type' is reserved for the event name")

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.data}

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False)


def agent_event(type: str, **data: Any) -> AgentEvent:
    return AgentEvent(type=type, data=data)
