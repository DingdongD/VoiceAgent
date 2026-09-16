from __future__ import annotations

from qwen_asr_vllm.agent.events import AgentEvent, agent_event


class VoiceTurnStateMachine:
    """Small local turn-taking state machine for full-duplex voice sessions."""

    def __init__(self):
        self.state = "idle"

    def on_user_audio(self) -> list[AgentEvent]:
        if self.state == "speaking":
            self.state = "listening"
            return [
                agent_event("turn_interrupted", from_state="speaking"),
                agent_event("turn_listening"),
            ]
        if self.state == "thinking":
            return []
        if self.state != "listening":
            self.state = "listening"
            return [agent_event("turn_listening")]
        return []

    def on_agent_start(self) -> list[AgentEvent]:
        if self.state != "thinking":
            self.state = "thinking"
            return [agent_event("turn_thinking")]
        return []

    def on_agent_audio(self) -> list[AgentEvent]:
        if self.state != "speaking":
            self.state = "speaking"
            return [agent_event("turn_speaking")]
        return []

    def on_agent_done(self) -> list[AgentEvent]:
        if self.state != "listening":
            self.state = "listening"
            return [agent_event("turn_listening")]
        return []

