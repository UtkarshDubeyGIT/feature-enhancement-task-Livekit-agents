import asyncio

from livekit.agents.voice.interrupt_handler import InterruptionFilter


def _ensure_event_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def _make_filter() -> InterruptionFilter:
    _ensure_event_loop()
    return InterruptionFilter(ignored_words=["uh","umm","hmm","haan","han","mhm","mm","ah","aha","oh","Uh-huh"])


def test_filler_word_ignored_while_speaking():
    filt = _make_filter()
    asyncio.run(filt.set_agent_speaking(True))

    assert filt.filter_transcript("uh umm", 0.9) == "ignore"


def test_filler_word_passes_when_silent():
    filt = _make_filter()
    asyncio.run(filt.set_agent_speaking(False))

    assert filt.filter_transcript("uh", 0.9) == "speech"


def test_real_interrupt_triggers_stop():
    filt = _make_filter()
    asyncio.run(filt.set_agent_speaking(True))

    assert filt.filter_transcript("stop talking please", 0.8) == "interrupt"


def test_mixed_filler_and_command_interrupts():
    filt = _make_filter()
    asyncio.run(filt.set_agent_speaking(True))

    assert filt.filter_transcript("umm can you help", 0.7) == "interrupt"


def test_low_confidence_filler_is_ignored():
    filt = _make_filter()
    asyncio.run(filt.set_agent_speaking(True))

    assert filt.filter_transcript("umm", 0.4) == "ignore"


def test_handle_transcription_event_async_interrupts():
    filt = _make_filter()
    asyncio.run(filt.set_agent_speaking(True))

    decision = asyncio.run(
        filt.handle_transcription_event("please stop right now", 0.9, "speaker-1")
    )

    assert decision == "interrupt"


def test_update_ignored_words_changes_behavior():
    _ensure_event_loop()
    filt = InterruptionFilter(ignored_words=["huh"])
    asyncio.run(filt.set_agent_speaking(True))

    assert filt.filter_transcript("huh", 0.9) == "ignore"

    asyncio.run(filt.update_ignored_words(["zzz"]))

    assert filt.filter_transcript("huh", 0.9) == "interrupt"
