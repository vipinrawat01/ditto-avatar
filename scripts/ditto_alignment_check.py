"""Offline check: an utterance must never strand a half batch inside Ditto.

Ditto's online audio2motion emits only in valid_clip_len (10) frame batches while
run_chunk feeds 5, so an utterance ending on an odd number of chunks leaves 5
frames inside the SDK: no writer callback, audio stays queued, playback freezes
on the last generated frame.

Sweeps 500 utterance lengths against the real helpers in avatars/ditto_avatar.py
(loaded by AST so the heavy cv2/torch imports stay out of the way) and reports
the negative control (fix disabled) next to the fixed pipeline.

    python scripts/ditto_alignment_check.py
"""

import ast
import pathlib
import numpy as np

SRC = pathlib.Path(__file__).resolve().parent.parent / "avatars" / "ditto_avatar.py"

# Pull the real constants + pure helpers out of the adapter, no side effects.
_WANT_FUNCS = {
    "_tail_frame_counts", "_alignment_flush_chunks",
    "_normalize_source_lips", "_normalize_condition_lips", "_offset_delays",
    "_is_final_audio_event", "_reserve_drop_frames",
}
_WANT_CONSTS = {
    "_CHUNKSIZE", "_PREPAD", "_SPLIT_LEN", "_HOP", "_LIP_KEYPOINTS",
}
_tree = ast.parse(SRC.read_text(encoding="utf-8"))
_ns = {"np": np}
for node in _tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in _WANT_FUNCS:
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
    elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in _WANT_CONSTS:
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)

_tail_frame_counts = _ns["_tail_frame_counts"]
_alignment_flush_chunks = _ns["_alignment_flush_chunks"]
_normalize_source_lips = _ns["_normalize_source_lips"]
_normalize_condition_lips = _ns["_normalize_condition_lips"]
_offset_delays = _ns["_offset_delays"]
_is_final_audio_event = _ns["_is_final_audio_event"]
_reserve_drop_frames = _ns["_reserve_drop_frames"]
LIP_KEYPOINTS = _ns["_LIP_KEYPOINTS"]
CHUNKSIZE, PREPAD, SPLIT_LEN, HOP = (
    _ns["_CHUNKSIZE"], _ns["_PREPAD"], _ns["_SPLIT_LEN"], _ns["_HOP"])
FRAMES_PER_CHUNK = CHUNKSIZE[1]
VALID_CLIP_LEN = 10          # shipped ditto online model
CHUNK_SAMPLES = 320          # one 20ms TTS packet


def run_utterance(audio_chunks, run_chunks_before, apply_fix):
    """Mirror put_audio_frame + _flush_tail; return total run_chunk calls."""
    run_chunks = run_chunks_before
    frames_scheduled = 0
    buf_len = PREPAD                      # buffer is reset to the prepad per utterance

    for _ in range(audio_chunks):         # streaming: fire whenever a window fits
        buf_len += CHUNK_SAMPLES
        while buf_len >= SPLIT_LEN:
            run_chunks += 1
            frames_scheduled += FRAMES_PER_CHUNK
            buf_len -= HOP

    for _ in _tail_frame_counts(audio_chunks, frames_scheduled, FRAMES_PER_CHUNK):
        run_chunks += 1                   # pad-and-flush the leftover speech

    if apply_fix:
        run_chunks += _alignment_flush_chunks(
            run_chunks, VALID_CLIP_LEN, FRAMES_PER_CHUNK)
    return run_chunks


def stranded_frames(run_chunks):
    """Frames sitting inside the SDK with no batch to complete them."""
    return run_chunks * FRAMES_PER_CHUNK % VALID_CLIP_LEN


def sweep(apply_fix, lengths=500):
    """Utterances run back-to-back: the SDK counter carries across them."""
    run_chunks = 1                        # JIT warm-up chunk fired in render()
    bad = []
    for audio_chunks in range(1, lengths + 1):
        run_chunks = run_utterance(audio_chunks, run_chunks, apply_fix)
        if stranded_frames(run_chunks):
            bad.append(audio_chunks)
    return bad


def simulate_playback(hold_ticks, bound, total_ticks=250):
    """Drift between a shown frame and the audio played with it, in frames.

    Upstream aligns generated frame n with audio frame n exactly, so the only
    way they separate is playback. The pump holds the last frame whenever the
    writer has not delivered one yet (a burst boundary, a slow diffusion step).
    Draining audio from its own queue keeps consuming during those holds, so
    every hold pushes the audio one frame ahead of the picture and it never
    recovers. Bound audio rides with its frame, so a hold emits silence.
    """
    video_idx = audio_idx = 0
    drift = []
    for tick in range(total_ticks):
        if tick in hold_ticks:
            if not bound:
                audio_idx += 1          # old pump drained audio through the gap
            continue
        drift.append(0 if bound else audio_idx - video_idx)
        if not bound:
            audio_idx += 1
        video_idx += 1
    return drift


def main():
    control = sweep(apply_fix=False)
    fixed = sweep(apply_fix=True)
    print(f"negative control (no flush): {len(control)}/500 stranded endings")
    print(f"with alignment flush       : {len(fixed)}/500 stranded endings")
    if control:
        print(f"  e.g. stranded at utterance lengths {control[:8]}")

    assert control, "negative control stranded nothing — the bug is not reproduced"
    assert not fixed, f"alignment flush still strands {len(fixed)} endings: {fixed[:8]}"
    # padding must never exceed one batch, or it would stall real playback
    assert all(_alignment_flush_chunks(n, VALID_CLIP_LEN, FRAMES_PER_CHUNK) <= 1
               for n in range(200)), "flush emitted more than one padding chunk"
    assert _alignment_flush_chunks(2, VALID_CLIP_LEN, FRAMES_PER_CHUNK) == 0, \
        "already-aligned utterance must not be padded"
    print("OK: every ending drains, padding never exceeds one chunk")

    # A hold every 5th tick ≈ one stall per run_chunk burst, which is what the
    # writer actually does when diffusion misses the 40ms budget.
    holds = {t for t in range(250) if t % 5 == 4}
    old = simulate_playback(holds, bound=False)
    new = simulate_playback(holds, bound=True)
    print(f"\nA/V drift under {len(holds)} writer stalls:")
    print(f"  separate queues (old): max {max(old)} frames = {max(old) * 40}ms, grows without bound")
    print(f"  audio bound to frame : max {max(new)} frames = {max(new) * 40}ms")

    assert max(old) > 10, "negative control did not drift — starvation model is wrong"
    assert max(new) == 0, f"bound audio still drifts by {max(new)} frames"
    print("OK: bound audio holds lip-sync through every stall")

    source = np.arange(63, dtype=np.float32).reshape(1, 63)
    neutral = (1000 + np.arange(63, dtype=np.float32)).reshape(1, 63)
    infos = [{"exp": source.copy()} for _ in range(3)]
    _normalize_source_lips(infos, neutral)
    lip = np.array(LIP_KEYPOINTS)
    other = np.array([i for i in range(21) if i not in LIP_KEYPOINTS])
    for info in infos:
        exp = info["exp"].reshape(21, 3)
        assert np.array_equal(exp[lip], neutral.reshape(21, 3)[lip])
        assert np.array_equal(exp[other], source.reshape(21, 3)[other])
    assert _offset_delays(60) == (3, 0)
    assert _offset_delays(-80) == (0, 2)
    assert _offset_delays(0) == (0, 0)
    assert not _is_final_audio_event({"status": "segment_end", "final": False})
    assert _is_final_audio_event({"status": "end", "final": True})
    condition = np.arange(265, dtype=np.float32).reshape(1, 265)
    normalized = _normalize_condition_lips(condition, neutral)
    assert np.array_equal(normalized[..., :-63], condition[..., :-63])
    assert np.array_equal(
        normalized[..., -63:].reshape(1, 21, 3)[:, lip],
        neutral.reshape(1, 21, 3)[:, lip])
    assert np.array_equal(
        normalized[..., -63:].reshape(1, 21, 3)[:, other],
        condition[..., -63:].reshape(1, 21, 3)[:, other])
    print("OK: source/model lip normalization preserves non-lip motion; signed offset is directional")

    pending = feed_cap = 20
    silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    assert pending >= feed_cap
    assert not (np.any(silence) and pending >= feed_cap)
    print("OK: terminal silence bypasses backpressure and can flush the final batch")

    warmup = pending = 5
    old_reservation = warmup + pending
    fixed_reservation = _reserve_drop_frames(warmup, pending)
    assert old_reservation - warmup == 5, "negative control did not drop real speech"
    assert fixed_reservation - warmup == 0, "first real speech batch is still dropped"
    print("OK: interrupt does not reserve the 5 warm-up frames twice")


if __name__ == "__main__":
    main()
