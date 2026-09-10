import ast
import time
from queue import Queue
from threading import Event, Thread
from pathlib import Path

import numpy as np


def test_tail_batches_match_audio_duration():
    source = (Path(__file__).parents[1] / "avatars" / "ditto_avatar.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_tail_frame_counts"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<tail>", "exec"), namespace)
    plan = namespace["_tail_frame_counts"]

    assert plan(2, 0) == [1]
    assert plan(50, 20) == [5]
    assert plan(52, 20) == [5, 1]
    assert plan(50, 25) == []

    assert "self._frame_keep.put(i < keep_frames)" in source
    assert "if not keep_frame:" in source
    # Audio is bound to its frame when the frame is generated, and the pump only
    # ever plays the audio carried by the frame it is showing. Draining the two
    # from separate queues slips 40ms every time either side starves.
    assert "self._ditto_frames.put((frame_bgr, _take_audio_pair(self._audio_out)))" in source
    assert "audio_delay.extend(frame_audio)" in source
    assert "self._audio_out.get_nowait()" not in source.split("def _pump")[1]
    assert "if in_speech and got_ditto:" not in source
    # Idle only once the SDK owes no frames AND nothing is left to play.
    assert "return not self._audio_out.empty() or not self._ditto_frames.empty()" in source
    assert 'DITTO_HOLD", "0.10"' in source
    assert 'DITTO_START_BUFFER", "6"' in source
    assert "DITTO_IDLE_FADE_MS" not in source
    assert "idle_blend" in source
    assert "final queues drained; holding" in source
    assert "and self._audio_out.empty() and self._ditto_frames.empty()" in source
    assert "and not self._final_pending" in source
    assert 'DITTO_FINAL_HOLD_MS", "370"' in source
    assert "final audio played; holding" in source
    assert "_END_HOLD = max(_HOLD, _AUDIO_DELAY_CHUNKS * 0.02)" in source
    assert "while (np.any(a) and" in source
    assert "ditto final audio received: flushing tail" in source
    neutralize = source.split("def _neutralize_source_lips", 1)[1].split("def _pump", 1)[0]
    assert "audio2motion.setup(" not in neutralize
    assert 'DITTO_AV_OFFSET_MS", "260"' in source
    assert "self._audio_cap" not in source
    assert "self._audio_out.qsize() >=" not in source
    assert "ditto stop fence" not in source
    assert "SDK tail stalled; draining final audio" in source


def test_idle_transition_matches_pose_before_blending():
    from avatars.ditto_avatar import (
        _blend_to_idle,
        _closest_idle_index,
        _frame_thumb,
        _resize_idle_frame,
    )

    final = np.full((8, 8, 3), 190, dtype=np.uint8)
    idle = [np.zeros_like(final), np.full_like(final, 200)]
    assert _closest_idle_index(final, [_frame_thumb(frame) for frame in idle]) == 1
    transition = _blend_to_idle(final, [idle[1]] * 4)
    assert len(transition) == 4
    assert all(np.mean(transition[i]) < np.mean(transition[i + 1])
               for i in range(3))
    assert _blend_to_idle(final, [np.zeros((4, 4, 3), dtype=np.uint8)]) == []

    recorded_idle = np.zeros((16, 24, 3), dtype=np.uint8)
    fitted_idle = _resize_idle_frame(recorded_idle, final.shape)
    assert fitted_idle.shape == final.shape
    assert len(_blend_to_idle(final, [fitted_idle] * 4)) == 4


def test_lip_response_sharpens_only_lip_motion():
    from avatars.ditto_avatar import _LIP_KEYPOINTS, _sharpen_lip_sequence

    sequence = np.zeros((1, 3, 265), dtype=np.float32)
    expression = sequence[..., -63:].reshape(1, 3, 21, 3)
    expression[0, 1, list(_LIP_KEYPOINTS), :] = 1.0
    expression[0, 2, list(_LIP_KEYPOINTS), :] = 1.5
    expression[0, :, 0, :] = np.array([0.0, 2.0, 4.0])[:, None]

    sharpened, previous = _sharpen_lip_sequence(sequence, 1.15)
    sharpened_exp = sharpened[..., -63:].reshape(1, 3, 21, 3)

    np.testing.assert_allclose(sharpened_exp[0, 0, list(_LIP_KEYPOINTS), :], 0.0)
    np.testing.assert_allclose(sharpened_exp[0, 1, list(_LIP_KEYPOINTS), :], 1.15)
    np.testing.assert_allclose(sharpened_exp[0, 2, list(_LIP_KEYPOINTS), :], 1.575)
    np.testing.assert_allclose(sharpened_exp[0, :, 0, :], expression[0, :, 0, :])
    np.testing.assert_allclose(previous, 1.5)


def test_semantic_pause_closes_only_lips_over_three_frames():
    from avatars.ditto_avatar import _LIP_KEYPOINTS, _SemanticPauseMotion

    class Stitch:
        def __call__(self, _source, driving, **_kwargs):
            return driving

    pauses = iter([True, True, True, False])
    neutral = np.zeros((len(_LIP_KEYPOINTS), 3), dtype=np.float32)
    wrapper = _SemanticPauseMotion(Stitch(), lambda: next(pauses), neutral, 3)

    source = {"exp": np.ones((1, 63), dtype=np.float32)}
    outputs = [wrapper({}, source) for _ in range(4)]
    lips = [out["exp"].reshape(21, 3)[list(_LIP_KEYPOINTS)] for out in outputs]

    np.testing.assert_allclose(lips[0], 2.0 / 3.0)
    np.testing.assert_allclose(lips[1], 1.0 / 3.0)
    np.testing.assert_allclose(lips[2], 0.0)
    np.testing.assert_allclose(lips[3], 1.0)
    np.testing.assert_allclose(outputs[2]["exp"].reshape(21, 3)[0], 1.0)


def test_terminal_phoneme_closure_can_be_partial_or_complete():
    from avatars.ditto_avatar import _LIP_KEYPOINTS, _SemanticPauseMotion

    class Stitch:
        def __call__(self, _source, driving, **_kwargs):
            return driving

    markers = iter([(False, 0.45), (False, 1.0), (False, 0.0)])
    neutral = np.zeros((len(_LIP_KEYPOINTS), 3), dtype=np.float32)
    wrapper = _SemanticPauseMotion(Stitch(), lambda: next(markers), neutral, 3)
    source = {"exp": np.ones((1, 63), dtype=np.float32)}

    outputs = [wrapper({}, source) for _ in range(3)]
    lips = [out["exp"].reshape(21, 3)[list(_LIP_KEYPOINTS)] for out in outputs]

    np.testing.assert_allclose(lips[0], 0.55)
    np.testing.assert_allclose(lips[1], 0.0)
    np.testing.assert_allclose(lips[2], 1.0)


def test_semantic_pause_requires_a_full_40ms_silence_frame():
    from avatars.ditto_avatar import DittoReal

    avatar = object.__new__(DittoReal)
    avatar._pause_frames = Queue()
    avatar._pause_packets = []
    avatar._vad_silent_frames = 0
    avatar._vad_rms = 0.004
    silence = np.zeros(320, dtype=np.float32)
    speech = np.full(320, 0.02, dtype=np.float32)

    # One quiet 40ms frame is ignored so tiny gaps inside a word do not make
    # the mouth chatter. The second consecutive quiet frame closes the lips.
    avatar._queue_pause_packet(silence, {})
    avatar._queue_pause_packet(silence, {})
    assert avatar._next_pause_frame() == (False, 0.0)

    avatar._queue_pause_packet(silence, {})
    avatar._queue_pause_packet(silence, {})
    assert avatar._next_pause_frame() == (True, 0.0)

    avatar._queue_pause_packet(speech, {"lip_close_strength": 0.25})
    avatar._queue_pause_packet(speech, {"lip_close_strength": 0.75})
    assert avatar._next_pause_frame() == (False, 0.5)

    # An explicit semantic pause still closes even if its samples are not
    # perfectly silent after TTS gain processing.
    avatar._queue_pause_packet(speech, {"semantic_pause": True})
    avatar._queue_pause_packet(speech, {"semantic_pause": True})
    assert avatar._next_pause_frame() == (True, 0.0)


def test_pause_marker_delay_is_audio_aligned_and_lip_only():
    from avatars.ditto_avatar import _LIP_KEYPOINTS, _SemanticPauseMotion

    class Stitch:
        def __call__(self, _source, driving, **_kwargs):
            return driving

    markers = iter([(True, 0.0), (False, 0.0), (False, 0.0)])
    neutral = np.zeros((len(_LIP_KEYPOINTS), 3), dtype=np.float32)
    wrapper = _SemanticPauseMotion(
        Stitch(), lambda: next(markers), neutral, close_frames=1, delay_frames=2)
    source = {"exp": np.ones((1, 63), dtype=np.float32)}

    outputs = [wrapper({}, source) for _ in range(3)]
    lips = [out["exp"].reshape(21, 3)[list(_LIP_KEYPOINTS)] for out in outputs]

    np.testing.assert_allclose(lips[0], 1.0)
    np.testing.assert_allclose(lips[1], 1.0)
    np.testing.assert_allclose(lips[2], 0.0)
    np.testing.assert_allclose(outputs[2]["exp"].reshape(21, 3)[0], 1.0)


def test_pause_marker_reset_drops_delayed_state():
    from avatars.ditto_avatar import _LIP_KEYPOINTS, _SemanticPauseMotion

    class Stitch:
        def __call__(self, _source, driving, **_kwargs):
            return driving

    markers = iter([(True, 0.0)] + [(False, 0.0)] * 4)
    neutral = np.zeros((len(_LIP_KEYPOINTS), 3), dtype=np.float32)
    wrapper = _SemanticPauseMotion(
        Stitch(), lambda: next(markers), neutral, close_frames=1, delay_frames=2)
    source = {"exp": np.ones((1, 63), dtype=np.float32)}

    wrapper({}, source)  # The close marker is buffered, not rendered yet.
    wrapper.reset()
    outputs = [wrapper({}, source) for _ in range(3)]

    for output in outputs:
        np.testing.assert_allclose(
            output["exp"].reshape(21, 3)[list(_LIP_KEYPOINTS)], 1.0)


def test_ditto_startup_keeps_known_good_direct_engine_load():
    source = (Path(__file__).parents[1] / "avatars" / "ditto_avatar.py").read_text(
        encoding="utf-8"
    )

    assert "GPU_INIT_LOCK" not in source
    assert 'self.sdk = StreamSDK(self.cfg["cfg_pkl"], self.cfg["data_root"])' in source


def test_pump_drains_stranded_final_audio_and_returns_idle(monkeypatch):
    from avatars.ditto_avatar import DittoReal

    monkeypatch.setenv("DITTO_START_BUFFER", "1")
    monkeypatch.setenv("DITTO_AV_OFFSET_MS", "0")
    monkeypatch.setenv("DITTO_FINAL_HOLD_MS", "370")

    events = []

    class Output:
        def push_video_frame(self, frame):
            events.append((time.perf_counter(), "video", int(frame[0, 0, 0])))

        def push_audio_frame(self, _pcm, data):
            events.append((time.perf_counter(), "audio", data))

    avatar = object.__new__(DittoReal)
    avatar._idle_bgr = [np.zeros((2, 2, 3), dtype=np.uint8)]
    avatar._ditto_frames = Queue()
    avatar._audio_out = Queue()
    avatar._ditto_frames.put((np.full((2, 2, 3), 255, dtype=np.uint8),
                              [(None, {}), (None, {})]))
    avatar._audio_out.put((np.ones(320, dtype=np.float32), {}))
    avatar._audio_out.put((np.ones(320, dtype=np.float32),
                           {"status": "end", "final": True}))
    avatar._final_pending = True
    avatar._utt_active = False
    avatar._last_ditto_frame_at = time.perf_counter() - 1.0
    avatar._tail_audio_fallback = False
    avatar._utt_show_pending = False
    avatar._dbg = False
    avatar._sync_csv = None
    avatar._prof_frames_used = avatar._prof_holds = avatar._prof_idle = 0
    avatar.speaking = False
    avatar.output = Output()
    avatar.record_video_data = lambda _frame: None
    avatar.record_audio_data = lambda _pcm: None
    avatar._prof_log = lambda force=False: None

    quit_event = Event()
    thread = Thread(target=avatar._pump, args=(quit_event,))
    thread.start()
    deadline = time.perf_counter() + 1.5
    while time.perf_counter() < deadline:
        final = next((t for t, kind, data in events
                      if kind == "audio" and data.get("final")), None)
        if final is not None and any(t > final and kind == "video" and data == 0
                                     for t, kind, data in events):
            break
        time.sleep(0.01)
    quit_event.set()
    thread.join(timeout=1.0)

    final = next(t for t, kind, data in events
                 if kind == "audio" and data.get("final"))
    idle = next(t for t, kind, data in events
                if t > final and kind == "video" and data == 0)
    assert idle - final >= 0.35
    assert idle - final < 0.70
    assert any(t > final and kind == "video" and 0 < data < 255
               for t, kind, data in events)
    assert not avatar._final_pending


def test_alignment_flush_lands_on_sdk_batch_boundary():
    """An utterance must never strand half a 10-frame batch inside the SDK."""
    source = (Path(__file__).parents[1] / "avatars" / "ditto_avatar.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_alignment_flush_chunks"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<align>", "exec"), namespace)
    flush = namespace["_alignment_flush_chunks"]

    assert flush(2, 10, 5) == 0      # already on a boundary → no padding
    assert flush(3, 10, 5) == 1      # 15 frames fed, 5 stranded → one chunk
    for run_chunks in range(200):
        pad = flush(run_chunks, 10, 5)
        assert pad <= 1, "padding must never exceed one batch"
        assert (run_chunks + pad) * 5 % 10 == 0, "utterance still strands a half batch"

    # padding frames must be discarded, never paired with real audio
    assert "keep_frames=0)" in source


def test_priming_fills_the_complete_non_rendered_d0_batch():
    from avatars.ditto_avatar import _priming_chunk_count

    assert _priming_chunk_count(10, 5) == 2
    assert _priming_chunk_count(5, 5) == 1
    assert _priming_chunk_count(12, 5) == 3

    source = (Path(__file__).parents[1] / "avatars" / "ditto_avatar.py").read_text(
        encoding="utf-8"
    )
    assert "for _ in range(priming_chunks):" in source
    assert "count_expected=False" in source
    assert "self._drop_ditto_frames += _CHUNKSIZE[1]" not in source


def test_flush_reserves_only_unaccounted_inflight_frames():
    source = (Path(__file__).parents[1] / "avatars" / "ditto_avatar.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_reserve_drop_frames"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<drop>", "exec"), namespace)
    reserve = namespace["_reserve_drop_frames"]

    assert reserve(5, 5) == 5
    assert reserve(5, 20) == 20
    assert reserve(0, 20) == 20


def test_tts_silence_tail_marks_only_its_final_frame():
    source = (Path(__file__).parents[1] / "tts" / "elevenlabs_tts.py").read_text(
        encoding="utf-8"
    )
    assert 'DITTO_TAIL_MS", "500"' in source
    assert "for index in range((pause_ms + 19) // 20):" in source
    assert "if index * 20 + 20 >= pause_ms:" in source
    assert 'status="end" if final else "segment_end"' in source
    assert "final=final" in source
    assert 'eventpoint["semantic_pause"] = True' in source


def _load_segment_gain():
    """_segment_gain without importing the module (needs elevenlabs + API key)."""
    source = (Path(__file__).parents[1] / "tts" / "elevenlabs_tts.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    wanted = [
        node for node in tree.body
        if (isinstance(node, ast.Assign)
            and any(getattr(t, "id", "").startswith(
                ("_TARGET", "_MAX", "_CEILING", "_SPEECH_GATE"))
                    for t in node.targets))
        or (isinstance(node, ast.FunctionDef)
            and node.name in ("_segment_gain", "_level_frame"))
    ]
    namespace = {"np": np, "os": __import__("os")}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "<gain>", "exec"), namespace)
    return (namespace["_segment_gain"], namespace["_level_frame"],
            namespace["_TARGET_RMS"], namespace["_CEILING"])


def _load_terminal_lip_close_strength():
    source = (Path(__file__).parents[1] / "tts" / "elevenlabs_tts.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    wanted = [
        node for node in tree.body
        if (isinstance(node, ast.Assign)
            and any(getattr(target, "id", "") == "_N_CLOSE_STRENGTH"
                    for target in node.targets))
        or (isinstance(node, ast.FunctionDef)
            and node.name == "_terminal_lip_close_strength")
    ]
    namespace = {"re": __import__("re"), "os": __import__("os")}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "<phoneme>", "exec"), namespace)
    return namespace["_terminal_lip_close_strength"]


def test_terminal_lip_close_strength_distinguishes_bilabial_and_nasal_endings():
    strength = _load_terminal_lip_close_strength()

    assert strength("Help them.") == 1.0
    assert strength("Please stop!") == 1.0
    assert strength("You can:") == 0.45
    assert strength("Around them.") == 1.0
    assert strength("Welcome today.") == 0.0


def test_segment_gain_matches_loudness_across_segments():
    """Hot and quiet segments must land at the same level, without clipping."""
    gain, level, target, _ = _load_segment_gain()
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(3200).astype(np.float32)
    noise /= np.max(np.abs(noise))

    def leveled(rms):
        scaled = noise * (rms / float(np.sqrt(np.mean(np.square(noise)))))
        frames = [scaled[i:i + 320] for i in range(0, 3200, 320)]
        g = gain(frames)
        out = np.concatenate([level(f, g) for f in frames])
        return float(np.sqrt(np.mean(np.square(out)))), float(np.max(np.abs(out)))

    hot_rms, hot_peak = leveled(0.30)     # "Certainly!"
    quiet_rms, quiet_peak = leveled(0.02)  # a long flat sentence
    assert abs(hot_rms - quiet_rms) < 0.2 * target, (hot_rms, quiet_rms)
    assert hot_peak <= 1.0 and quiet_peak <= 1.0
    # Silence must pass through untouched, not be amplified into noise.
    assert gain([np.zeros(320, dtype=np.float32)]) == 1.0
    assert gain([]) == 1.0


def test_segment_gain_ignores_leading_silence_and_can_lift_quiet_speech():
    gain, _, target, _ = _load_segment_gain()
    silence = [np.zeros(320, dtype=np.float32) for _ in range(4)]
    quiet = [np.full(320, 0.008, dtype=np.float32) for _ in range(4)]

    g = gain(silence + quiet)
    assert g > 4.0
    assert abs(0.008 * g - target) < 0.01


def test_level_frame_limits_hot_audio_without_clipping_or_retiming():
    _, level, target, ceiling = _load_segment_gain()
    frame = np.linspace(-1.0, 1.0, 320, dtype=np.float32)

    output = level(frame, 4.0)

    assert output.shape == frame.shape
    assert output.dtype == np.float32
    assert float(np.max(np.abs(output))) <= ceiling + 1e-6
    assert np.allclose(output, frame * ceiling, atol=1e-6)
    assert target > 0
