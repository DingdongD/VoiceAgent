import pytest

from qwen_asr_vllm.agent.shared_bytes import (
    SharedBytesDescriptor,
    SharedBytesTransport,
    _open_segment,
)


def test_shared_bytes_transport_keeps_small_payloads_inline_and_unlinks_large_payloads():
    sender = SharedBytesTransport(threshold=8)
    receiver = SharedBytesTransport(threshold=8)
    try:
        assert sender.encode(b"small") == b"small"
        descriptor = sender.encode(b"large-payload")
        assert isinstance(descriptor, SharedBytesDescriptor)

        assert receiver.decode(descriptor) == b"large-payload"
        with pytest.raises(FileNotFoundError):
            _open_segment(descriptor.name)

        assert sender.metrics()["created_segments"] == 1
        assert receiver.metrics() == {
            "created_segments": 0,
            "created_bytes": 0,
            "received_segments": 1,
            "received_bytes": 13,
            "unlinked_segments": 1,
            "cleanup_segments": 0,
            "outstanding_segments": 0,
            "threshold": 8,
        }
    finally:
        receiver.close()
        sender.close()


def test_shared_bytes_transport_close_cleans_unconsumed_segments():
    sender = SharedBytesTransport(threshold=1)
    descriptor = sender.encode(b"orphan")

    sender.close()

    with pytest.raises(FileNotFoundError):
        _open_segment(descriptor.name)
    assert sender.metrics()["cleanup_segments"] == 1
    assert sender.metrics()["outstanding_segments"] == 0
