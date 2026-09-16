from qwen_asr_vllm.agent.turn import VoiceTurnStateMachine


def test_turn_state_moves_from_listening_to_thinking_to_speaking_to_listening():
    turns = VoiceTurnStateMachine()

    assert [event.type for event in turns.on_user_audio()] == ["turn_listening"]
    assert [event.type for event in turns.on_agent_start()] == ["turn_thinking"]
    assert [event.type for event in turns.on_agent_audio()] == ["turn_speaking"]
    assert [event.type for event in turns.on_agent_done()] == ["turn_listening"]


def test_turn_state_interrupts_agent_when_user_speaks_during_speaking():
    turns = VoiceTurnStateMachine()
    turns.on_user_audio()
    turns.on_agent_start()
    turns.on_agent_audio()

    events = turns.on_user_audio()

    assert [event.type for event in events] == ["turn_interrupted", "turn_listening"]
    assert events[0].data["from_state"] == "speaking"



def test_turn_state_does_not_interrupt_while_agent_is_only_thinking():
    turns = VoiceTurnStateMachine()
    turns.on_user_audio()
    turns.on_agent_start()

    assert turns.on_user_audio() == []
    assert turns.state == "thinking"
