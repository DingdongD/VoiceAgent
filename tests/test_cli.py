"""Argument parsing and file discovery.

The commands themselves need a GPU, but everything that decides *what* they will do
is ordinary logic and worth holding still -- a silently dropped ``--language`` or a
directory walk that misses ``.flac`` is the kind of bug that only shows up as bad
output much later.
"""
import pytest

from qwen_asr_vllm.cli import build_parser, collect_audio_paths, main


class TestParser:
    def test_transcribe_defaults(self):
        args = build_parser().parse_args(["transcribe", "--model", "/ckpt", "a.wav"])
        assert args.command == "transcribe"
        assert args.inputs == ["a.wav"]
        assert args.language is None
        assert args.output_format == "text"
        assert args.max_num_seqs == 32

    def test_serve_defaults(self):
        args = build_parser().parse_args(["serve", "--model", "/ckpt"])
        assert (args.host, args.port) == ("0.0.0.0", 8000)
        assert args.frontend_workers == 4

    def test_engine_flags_are_shared_by_both_commands(self):
        parser = build_parser()
        for command, tail in (("transcribe", ["a.wav"]), ("serve", [])):
            args = parser.parse_args(
                [command, "--model", "/ckpt", "--gpu-memory-utilization", "0.5", *tail]
            )
            assert args.gpu_memory_utilization == 0.5

    def test_prefix_cache_flag_inverts(self):
        parser = build_parser()
        assert parser.parse_args(["serve", "--model", "/c"]).no_prefix_cache is False
        assert (
            parser.parse_args(["serve", "--model", "/c", "--no-prefix-cache"]).no_prefix_cache
            is True
        )

    def test_model_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["transcribe", "a.wav"])

    def test_a_command_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_main_dispatches_to_the_subcommand(self, monkeypatch):
        seen = {}

        def fake_serve(args):
            seen["port"] = args.port
            return 0

        monkeypatch.setattr("qwen_asr_vllm.cli.serve_command", fake_serve)

        assert main(["serve", "--model", "/ckpt", "--port", "9001"]) == 0
        assert seen == {"port": 9001}


class TestAudioDiscovery:
    def test_finds_audio_recursively_and_sorted(self, tmp_path):
        (tmp_path / "nested").mkdir()
        for name in ["b.wav", "a.flac", "nested/c.mp3", "notes.txt", "cover.png"]:
            (tmp_path / name).write_bytes(b"")

        found = [path.name for path in collect_audio_paths([str(tmp_path)])]

        assert found == ["a.flac", "b.wav", "c.mp3"]

    def test_explicit_files_bypass_the_suffix_filter(self, tmp_path):
        odd = tmp_path / "recording.dat"
        odd.write_bytes(b"")
        assert collect_audio_paths([str(odd)]) == [odd]

    def test_missing_path_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            collect_audio_paths([str(tmp_path / "absent.wav")])

    def test_empty_directory_is_reported(self, tmp_path):
        with pytest.raises(ValueError, match="no audio files"):
            collect_audio_paths([str(tmp_path)])
