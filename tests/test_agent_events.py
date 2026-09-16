import json

from qwen_asr_vllm.agent.events import AgentEvent, agent_event


def test_agent_event_serializes_type_and_payload_as_flat_json():
    event = agent_event("llm_chunk", text="hello", sequence=3)

    assert json.loads(event.to_json()) == {
        "type": "llm_chunk",
        "text": "hello",
        "sequence": 3,
    }


def test_agent_event_rejects_payload_type_collision():
    try:
        AgentEvent("asr_final", {"type": "nested"})
    except ValueError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("AgentEvent accepted a payload with reserved key 'type'")
