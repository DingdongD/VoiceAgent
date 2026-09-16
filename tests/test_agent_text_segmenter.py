from qwen_asr_vllm.agent.text_segmenter import TextSegmenter


def test_text_segmenter_does_not_split_common_abbreviations_or_decimals():
    segmenter = TextSegmenter(min_chars=10)

    emitted = []
    for token in ["Please see Dr. ", "Lee at 3.14 p.m. ", "Thanks. "]:
        emitted.extend(segmenter.add(token))

    assert emitted == ["Please see Dr. Lee at 3.14 p.m. Thanks."]
    assert segmenter.flush() is None


def test_text_segmenter_splits_chinese_sentence_endings():
    segmenter = TextSegmenter(min_chars=1)

    assert segmenter.add("你好。继续") == ["你好。"]
    assert segmenter.flush() == "继续"


def test_text_segmenter_flushes_long_fragment_on_word_boundary():
    segmenter = TextSegmenter(min_chars=1, flush_chars=12)

    assert segmenter.add("alpha beta gamma ") == ["alpha beta"]
    assert segmenter.flush() == "gamma"


def test_text_segmenter_ignores_standalone_sentence_punctuation():
    segmenter = TextSegmenter(min_chars=1)

    assert segmenter.add(".") == []
    assert segmenter.flush() == "."
