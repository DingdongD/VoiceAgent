from bench import nano_llm_batching_probe


def test_probe_reports_warmup_and_online_metrics_separately():
    backend = type(
        "Backend",
        (),
        {
            "stats": {"step_calls": 7, "max_step_batch_size": 4},
            "warmup_stats": {
                "batch_sizes": [1, 2, 4],
                "step_calls": 3,
                "max_step_batch_size": 4,
            },
        },
    )()

    assert nano_llm_batching_probe.backend_metrics(backend) == {
        "backend_stats": {"step_calls": 7, "max_step_batch_size": 4},
        "warmup_stats": {
            "batch_sizes": [1, 2, 4],
            "step_calls": 3,
            "max_step_batch_size": 4,
        },
    }


def test_probe_parser_accepts_fixed_cache_and_warmup_shapes():
    args = nano_llm_batching_probe.build_parser().parse_args(
        [
            "--model-path",
            "/model",
            "--num-kvcache-blocks",
            "64",
            "--warmup-batch-sizes",
            "1",
            "2",
            "4",
        ]
    )

    assert args.num_kvcache_blocks == 64
    assert args.warmup_batch_sizes == [1, 2, 4]
