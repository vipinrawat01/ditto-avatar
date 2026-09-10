###############################################################################
#  Ditto talking-head adapter for LiveTalking
#
#  Bridges antgroup/ditto-talkinghead (StreamSDK online pipeline) into
#  LiveTalking's TTS + WebRTC output.
#
#  Key facts about the two sides (learned the hard way):
#   * Ditto's StreamSDK runs 6 worker threads; the LAST one (writer_worker)
#     calls self.writer(frame_rgb, fmt="rgb") for every finished frame. We
#     REPLACE self.sdk.writer with our own sink so those frames go to WebRTC
#     instead of a file. (Do NOT read writer_queue — that races the writer.)
#   * LiveTalking's WebRTC tracks each play on their OWN fixed clock: video at
#     25fps (VIDEO_PTIME=0.040), audio at 20ms. A/V sync therefore depends on
#     feeding both queues at real-time rate. So we pair each emitted Ditto
#     frame with the 2 audio chunks (2x20ms = 40ms = 1 frame) that produced it
#     and push them together from one 25fps pump thread. Frame[k] keeps its
#     own audio[k] → lip sync is preserved regardless of pipeline latency.
#   * Ditto only emits frames while fed audio. When idle we loop the source
#     frames (source_info["img_rgb_lst"]) + silence so the video isn't black.
#
#  Env vars:
#     DITTO_REPO       path to the cloned ditto-talkinghead repo
#     DITTO_CFG        cfg pkl (use the *pytorch* one on non-Ampere GPUs)
#     DITTO_DATA_ROOT  model dir (ditto_pytorch)
#     DITTO_PROF       log stage timings, queue sizes, and frame accounting
#     DITTO_FEED_CAP   max frames the SDK may run ahead (default 20 ≈ 0.8s). Lower
#                      → snappier interrupt + less buffered lag; too low → stutter.
#     DITTO_DEBUG      dump the first N writer + source frames and log per-frame
#                      writer-vs-source / writer-vs-previous diffs (proves whether
#                      Ditto is generating or just replaying the source video).
#     DITTO_DEBUG_DIR  where to dump (default /tmp/ditto_debug)
#     DITTO_DEBUG_N    how many writer frames to dump/compare (default 20)
#     DITTO_NEUTRAL_LIPS  prevent source-video lip motion from leaking into the
#                         audio-driven mouth (default 1; set 0 to compare)
#
#  Mouth/expression shaping (Ditto setup params — all opt-in; unset = baseline):
#     DITTO_EMO        emotion index (default 4; 0 = neutral)
#     DITTO_EXP        scale expression/mouth AMPLITUDE, e.g. 0.85 → smaller mouth
#                      WITHOUT blurring lip-sync (this is the knob for "small mouth
#                      AND good sync"). Builds use_d_keys with head keys kept full.
#     DITTO_SMO_K_D    temporal smoothing of driving motion — LOW (1) = sharpest
#                      lip-sync (<=1 disables it); HIGH = smaller-but-mushy mouth.
#     DITTO_LIP_RESPONSE  amplify frame-to-frame lip motion only (default 1.2).
#                         1.0 disables; values above 1 make opening/closing react
#                         faster without shifting audio/video timestamps.
#     DITTO_PAUSE_CLOSE_MS  time used to blend lips closed during punctuation
#                           silence (default 120ms; head and eyes are untouched).
#     DITTO_SMO_K_S    smoothing of source (head/body) motion — NOT mouth-related
#     DITTO_FADE_TYPE / DITTO_OVERLAP  online-chunk blending
#
#  use_d_keys MUST include the head keys (pitch/yaw/roll/t) or the head freezes and
#  output looks like raw source playback — DITTO_EXP builds it correctly; never
#  pass a bare {"exp": ...}. If source.mp4 shows teeth/open mouth, generation
#  inherits it regardless of these knobs.
###############################################################################

import os
import sys
import time
import queue
import numpy as np
import cv2
from collections import deque
from queue import Queue
from threading import Thread, Event

from avatars.base_avatar import BaseAvatar
from registry import register
from utils.logger import logger

_DITTO_REPO = os.environ.get("DITTO_REPO", "/workspace/ditto-talkinghead")
if _DITTO_REPO not in sys.path:
    sys.path.insert(0, _DITTO_REPO)

# 2 audio chunks (20ms each) per 25fps video frame. The WebRTC video track is
# hardwired to 25fps in server/webrtc.py, so this ratio is fixed.
_AUDIO_CHUNKS_PER_FRAME = 2
_SILENCE = np.zeros(320, dtype=np.int16)
_LIP_KEYPOINTS = (6, 12, 14, 17, 19, 20)

# hubert sliding window, copied from ditto's inference.py (chunksize=(3,5,2)):
#   prepad 3*640 zeros, window = sum(chunksize)*0.04*16k + 80 = 6480,
#   hop = 5*640 = 3200 → each run_chunk emits 5 frames (5*640 = 3200, balanced).
_CHUNKSIZE = (3, 5, 2)
_PREPAD = _CHUNKSIZE[0] * 640          # 1920
_SPLIT_LEN = int(sum(_CHUNKSIZE) * 0.04 * 16000) + 80   # 6480
_HOP = _CHUNKSIZE[1] * 640             # 3200


def _tail_frame_counts(audio_chunks, scheduled_frames, batch_frames=5):
    """Return real frame counts for each fixed-size final Ditto batch."""
    remaining = max(0, (audio_chunks + 1) // 2 - scheduled_frames)
    return [min(batch_frames, remaining - i)
            for i in range(0, remaining, batch_frames)]


def _take_audio_pair(audio_queue):
    """The 40ms of speech that drove this frame — 2 x 20ms packets, in order.

    Upstream feeds run_chunk a 10-frame window (3 left ctx + 5 valid + 2 right)
    and emits the middle 5, so generated frame n lines up with audio frame n
    exactly. Binding the pair here, at generation time, is what keeps that
    alignment: draining audio and video from separate queues during playback
    slips a frame every time either side starves, and never recovers."""
    pair = []
    for _ in range(_AUDIO_CHUNKS_PER_FRAME):
        try:
            pair.append(audio_queue.get_nowait())
        except queue.Empty:
            pair.append((None, {}))     # past the end of real speech
    return pair


def _alignment_flush_chunks(run_chunks, valid_clip_len, frames_per_chunk=5):
    """Padding chunks needed so the SDK's batch boundary lands on our last frame.

    Ditto's online audio2motion only emits once it has valid_clip_len frames
    (10 in the shipped model) but run_chunk feeds 5, so an utterance ending on
    an odd number of chunks strands half a batch inside the SDK: no callback,
    audio stays queued, playback freezes on the final frame."""
    pending = run_chunks * frames_per_chunk % valid_clip_len
    return 0 if pending == 0 else (valid_clip_len - pending + frames_per_chunk - 1) // frames_per_chunk


def _priming_chunk_count(valid_clip_len, frames_per_chunk=5):
    """Chunks of silence needed to establish online Ditto's non-rendered d0."""
    return max(1, (int(valid_clip_len) + frames_per_chunk - 1) // frames_per_chunk)


def _reserve_drop_frames(already_reserved, pending):
    """Reserve each in-flight frame once, including the JIT warm-up batch."""
    return max(already_reserved, pending)


def _frame_thumb(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    # The avatars are centred. Compare the face and upper body instead of the
    # static background, which otherwise dominates the nearest-idle match.
    crop = gray[height // 20:height * 9 // 10, width // 5:width * 4 // 5]
    if not crop.size:
        crop = gray
    return cv2.resize(crop, (32, 32),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def _closest_idle_index(frame, idle_thumbs):
    target = _frame_thumb(frame)
    return min(range(len(idle_thumbs)),
               key=lambda i: float(np.mean(np.abs(target - idle_thumbs[i]))))


def _blend_to_idle(frame, idle_frames):
    if not idle_frames or any(frame.shape != target.shape for target in idle_frames):
        return []
    blended = []
    count = len(idle_frames)
    for step, target in enumerate(idle_frames, 1):
        progress = step / (count + 1)
        alpha = progress * progress * (3.0 - 2.0 * progress)
        blended.append(cv2.addWeighted(frame, 1.0 - alpha, target, alpha, 0))
    return blended


def _resize_idle_frame(frame, target_shape):
    """Match idle video dimensions to Ditto's WebRTC output dimensions."""
    target_height, target_width = target_shape[:2]
    if frame.shape[:2] == (target_height, target_width):
        return frame
    shrinking = frame.shape[0] > target_height or frame.shape[1] > target_width
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_width, target_height),
                      interpolation=interpolation)


def _normalize_source_lips(source_infos, neutral_exp):
    """Replace only source lip motion; preserve its head, eyes and body motion."""
    neutral_lips = np.asarray(neutral_exp).reshape(-1, 3)[list(_LIP_KEYPOINTS)]
    for info in source_infos:
        exp = np.array(info["exp"], copy=True)
        exp.reshape(-1, 3)[list(_LIP_KEYPOINTS)] = neutral_lips
        info["exp"] = exp
    return len(source_infos)


def _normalize_condition_lips(condition, neutral_exp):
    """Replace lip rows in Ditto's existing 265-d source condition."""
    result = np.array(condition, copy=True)
    exp = result[..., -63:].reshape(*result.shape[:-1], 21, 3)
    neutral_lips = np.asarray(neutral_exp).reshape(21, 3)[list(_LIP_KEYPOINTS)]
    exp[..., list(_LIP_KEYPOINTS), :] = neutral_lips
    return result


def _sharpen_lip_sequence(sequence, response, previous_lips=None):
    """Increase lip velocity while preserving every frame's timestamp.

    Ditto stores expression as the final 63 values (21 keypoints x 3). The
    original conversion still clips the adjusted sequence to the model's
    learned range after this function runs.
    """
    result = np.array(sequence, copy=True)
    if result.ndim != 3 or result.shape[0] != 1 or result.shape[1] == 0:
        return result, previous_lips
    if result.shape[-1] < 63 or response <= 1.0:
        return result, previous_lips

    exp = result[..., -63:].reshape(1, result.shape[1], 21, 3)
    previous = None if previous_lips is None else np.asarray(previous_lips).copy()
    lip_indices = list(_LIP_KEYPOINTS)
    for frame_index in range(result.shape[1]):
        current = exp[0, frame_index, lip_indices, :].copy()
        if previous is not None:
            exp[0, frame_index, lip_indices, :] = previous + response * (current - previous)
        previous = current
    return result, previous


def install_lip_response(sdk, response):
    """Wrap Ditto's motion conversion with lip-only temporal sharpening."""
    response = max(1.0, min(float(response), 1.5))
    if response <= 1.0:
        logger.info("ditto lip response disabled")
        return

    original_cvt_fmt = sdk.audio2motion.cvt_fmt
    previous_lips = None
    priming_call = True

    def cvt_fmt_with_lip_response(sequence):
        nonlocal previous_lips, priming_call
        # Online Ditto's first conversion establishes motion_stitch.d0 and is
        # never rendered. Do not let that hidden context affect visible lips.
        if priming_call:
            priming_call = False
            return original_cvt_fmt(sequence)
        sharpened, previous_lips = _sharpen_lip_sequence(
            sequence, response, previous_lips)
        return original_cvt_fmt(sharpened)

    sdk.audio2motion.cvt_fmt = cvt_fmt_with_lip_response
    logger.info("ditto lip response: %.2fx", response)


class _SemanticPauseMotion:
    """Apply phoneme/pause lip closure immediately before motion stitching."""

    def __init__(self, motion_stitch, next_pause, neutral_lips, close_frames,
                 delay_frames=0):
        object.__setattr__(self, "_obj", motion_stitch)
        object.__setattr__(self, "_next_pause", next_pause)
        object.__setattr__(self, "_neutral_lips", np.asarray(neutral_lips).copy())
        object.__setattr__(self, "_close_frames", max(1, int(close_frames)))
        object.__setattr__(self, "_delay_frames", max(0, int(delay_frames)))
        object.__setattr__(self, "_delay", deque())
        self.reset()

    def reset(self):
        delay = object.__getattribute__(self, "_delay")
        delay.clear()
        delay.extend([(False, 0.0)] * object.__getattribute__(self, "_delay_frames"))
        object.__setattr__(self, "_pause_run", 0)

    def __call__(self, x_s_info, x_d_info, **kwargs):
        marker = self._next_pause()
        delay = object.__getattribute__(self, "_delay")
        if object.__getattribute__(self, "_delay_frames"):
            delay.append(marker)
            marker = delay.popleft()
        if isinstance(marker, tuple):
            semantic_pause, close_strength = marker
        else:
            semantic_pause = bool(marker)
            close_strength = 1.0 if semantic_pause else 0.0
        close_strength = max(0.0, min(float(close_strength), 1.0))
        if semantic_pause:
            run = object.__getattribute__(self, "_pause_run") + 1
            object.__setattr__(self, "_pause_run", run)
            alpha = min(1.0, run / object.__getattribute__(self, "_close_frames"))
        elif close_strength:
            object.__setattr__(self, "_pause_run", 0)
            alpha = close_strength
        else:
            object.__setattr__(self, "_pause_run", 0)
            alpha = 0.0
        if alpha:
            result = dict(x_d_info)
            exp = np.array(x_d_info["exp"], copy=True)
            lips = exp.reshape(-1, 3)
            neutral = object.__getattribute__(self, "_neutral_lips")
            lip_indices = list(_LIP_KEYPOINTS)
            lips[lip_indices] = (1.0 - alpha) * lips[lip_indices] + alpha * neutral
            result["exp"] = exp
            x_d_info = result
        return self._obj(x_s_info, x_d_info, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_obj"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_obj"), name, value)


def install_semantic_pause_closure(sdk, next_pause, close_ms=120, offset_ms=0):
    neutral_lips = getattr(sdk, "_livetalking_neutral_lips", None)
    if neutral_lips is None:
        logger.warning("ditto semantic pause closure disabled: neutral lips unavailable")
        return
    close_frames = max(1, int(round(float(close_ms) / 40.0)))
    delay_frames = max(0, int(float(offset_ms) / 40.0))
    sdk.motion_stitch = _SemanticPauseMotion(
        sdk.motion_stitch, next_pause, neutral_lips, close_frames,
        delay_frames=delay_frames)
    logger.info(
        "ditto semantic pause closure: %d frames (%dms), audio-aligned delay=%d frames (%dms)",
        close_frames, close_frames * 40, delay_frames, delay_frames * 40)
    return sdk.motion_stitch


def neutralize_sdk_source_lips(sdk, idle_path):
    """Apply the same closed-mouth source baseline used by the live server."""
    cap = cv2.VideoCapture(idle_path)
    ok, idle_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {idle_path}")

    neutral = sdk.avatar_registrar.source2info(
        cv2.cvtColor(idle_bgr, cv2.COLOR_BGR2RGB),
        crop_scale=sdk.crop_scale,
        crop_vx_ratio=sdk.crop_vx_ratio,
        crop_vy_ratio=sdk.crop_vy_ratio,
        crop_flag_do_rot=sdk.crop_flag_do_rot,
    )
    neutral_exp = neutral["x_s_info"]["exp"]
    sdk._livetalking_neutral_lips = np.asarray(neutral_exp).reshape(21, 3)[
        list(_LIP_KEYPOINTS)].copy()
    count = _normalize_source_lips(sdk.source_info["x_s_info_lst"], neutral_exp)
    for name in ("s_kp_cond", "kp_cond"):
        setattr(sdk.audio2motion, name, _normalize_condition_lips(
            getattr(sdk.audio2motion, name), neutral_exp))
    return count


def _offset_delays(offset_ms):
    """Return (20ms audio chunks, 40ms video frames) to delay."""
    return (
        max(0, int(round(float(offset_ms) / 20.0))),
        max(0, int(round(-float(offset_ms) / 40.0))),
    )


def _is_final_audio_event(metadata):
    """True only for the final marker, never for ordinary sentence pauses."""
    return (isinstance(metadata, dict)
            and metadata.get("status") == "end"
            and metadata.get("final") is not False)


def setup_kwargs_from_env():
    """Ditto setup() kwargs for the current env — the single source of truth.

    scripts/ditto_make_idle.py calls this too: an idle clip rendered with even
    slightly different kwargs will not match generated speech pixel for pixel,
    and the mismatch is exactly what shows up as a seam at the switch to idle.
    """
    # Base call = the working "大牙" baseline: Ditto's DEFAULT motion keys.
    # Do NOT add use_d_keys here — it restricts the applied keys and flattens
    # the mouth so the output looks like raw source playback (reverted).
    setup_kwargs = dict(
        sampling_timesteps=int(os.environ.get("DITTO_STEPS", "5")),
        max_size=int(os.environ.get("DITTO_MAX_SIZE", "1024")),
        emo=int(os.environ.get("DITTO_EMO", "4")),
        online_mode=os.environ.get("DITTO_ONLINE", "0") == "1",
    )
    # Official mouth/expression-shaping knobs — the correct way to tame the
    # mouth (never use_d_keys). Forwarded ONLY when the env var is set, so the
    # default call stays identical to the baseline. If a name doesn't match the
    # installed Ditto build, setup() errors only when you opt into that var.
    #   DITTO_FADE_TYPE  crossfade style between online chunks (e.g. d0)
    #   DITTO_OVERLAP    online chunk overlap (overlap_v2, frames)
    #   DITTO_SMO_K_D    smoothing kernel over driving motion
    #   DITTO_SMO_K_S    smoothing kernel over source motion
    if os.environ.get("DITTO_FADE_TYPE"):
        setup_kwargs["fade_type"] = os.environ["DITTO_FADE_TYPE"]
    if os.environ.get("DITTO_OVERLAP"):
        setup_kwargs["overlap_v2"] = int(os.environ["DITTO_OVERLAP"])
    if os.environ.get("DITTO_SMO_K_D"):
        setup_kwargs["smo_k_d"] = int(os.environ["DITTO_SMO_K_D"])
    if os.environ.get("DITTO_SMO_K_S"):
        setup_kwargs["smo_k_s"] = int(os.environ["DITTO_SMO_K_S"])
    # DITTO_EXP: scale ONLY the expression/mouth amplitude → smaller mouth
    # WITHOUT blurring lip-sync (unlike smo_k_d, which smears the shapes).
    # Keep the head keys (pitch/yaw/roll/t) at full: the old bug passed a bare
    # {"exp": x}, dropping them, so the head froze and it looked like raw
    # source playback. motion_stitch applies (v - d0[k]) * use_d_keys[k] per key.
    if os.environ.get("DITTO_EXP"):
        _e = float(os.environ["DITTO_EXP"])
        setup_kwargs["use_d_keys"] = {"exp": _e, "pitch": 1.0, "yaw": 1.0, "roll": 1.0, "t": 1.0}
    return setup_kwargs


def load_model():
    return {
        "cfg_pkl": os.environ.get(
            "DITTO_CFG",
            f"{_DITTO_REPO}/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"),
        "data_root": os.environ.get(
            "DITTO_DATA_ROOT",
            f"{_DITTO_REPO}/checkpoints/ditto_pytorch"),
    }


def load_avatar(avatar_id):
    """For Ditto the 'avatar' is a source portrait image or video."""
    for ext in ("mp4", "png", "jpg", "jpeg"):
        p = f"./data/avatars/{avatar_id}/source.{ext}"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"ditto avatar '{avatar_id}' needs data/avatars/{avatar_id}/source.mp4 (or .png)")


def warm_up(batch_size, model, *args):
    return


def _drain_queue(q):
    """Empty a Queue without blocking (used to cut buffered speech on interrupt)."""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


class _FrameSink:
    """Stands in for Ditto's VideoWriterByImageIO. writer_worker calls this per
    finished RGB frame; we hand it to the avatar instead of writing a file."""
    def __init__(self, on_frame):
        self._on_frame = on_frame

    def __call__(self, frame_rgb, fmt="rgb"):
        self._on_frame(frame_rgb)

    def close(self):
        pass


class _Prof:
    """Transparent proxy that times __call__ on a Ditto pipeline stage so we can
    see which stage caps throughput. All other attribute access/set proxies to
    the wrapped object, so worker code using .d0, .cvt_fmt, .seq_frames still works."""
    def __init__(self, name, obj):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_n", 0)
        object.__setattr__(self, "_t", 0.0)
        object.__setattr__(self, "_sync", None)
        if os.environ.get("DITTO_PROF_SYNC", "1") != "0":
            try:
                import torch
                if torch.cuda.is_available():
                    object.__setattr__(self, "_sync", torch.cuda.synchronize)
            except Exception:
                pass

    def __call__(self, *a, **k):
        if self._sync:
            self._sync()
        s = time.perf_counter()
        r = self._obj(*a, **k)
        if self._sync:
            self._sync()
        object.__setattr__(self, "_t", self._t + (time.perf_counter() - s))
        object.__setattr__(self, "_n", self._n + 1)
        if self._n % 10 == 0:
            ms = self._t / self._n * 1000.0
            logger.info(f"[prof] {self._name}: {ms:.1f} ms/call → {1000.0/ms:.1f} fps cap ({self._n})")
        return r

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_obj"), k)

    def __setattr__(self, k, v):
        setattr(object.__getattribute__(self, "_obj"), k, v)


@register("avatar", "ditto")
class DittoReal(BaseAvatar):
    def __init__(self, opt, model, avatar):
        super().__init__(opt)               # wires self.tts and self.output
        self.cfg = model
        self.source_path = avatar

        self._t_build = time.perf_counter()   # [timing] session build start (= right after /offer)
        from stream_pipeline_online import StreamSDK
        _t = time.perf_counter()
        self.sdk = StreamSDK(self.cfg["cfg_pkl"], self.cfg["data_root"])
        logger.info("[ditto-timing] StreamSDK engine load: %.2fs", time.perf_counter() - _t)

        # Sliding-window buffer for hubert (see constants above). Starts with the
        # prepad so the first window's valid region lines up, exactly as inference.py.
        self._feat_buf = np.full(_PREPAD, 0.0, dtype=np.float32)
        self._feat_pos = 0

        self._ditto_frames: "Queue" = Queue()  # (BGR frame, [(pcm, userdata) x2]) ready to play
        self._audio_out: "Queue" = Queue()      # (float32[320], userdata) awaiting its frame
        self._frame_keep: "Queue" = Queue()     # real audio frame=True, padded tail=False
        self._pause_frames: "Queue" = Queue()   # semantic-pause flag per 40ms motion frame
        self._pause_packets = []                 # pair 2 x 20ms TTS packets per frame
        self._pause_motion = None
        self._vad_silent_frames = 0
        self._vad_rms = max(0.0, float(os.environ.get("DITTO_VAD_RMS", "0.004")))
        self._prof = bool(os.environ.get("DITTO_PROF"))
        self._prof_t0 = self._prof_last = time.perf_counter()
        self._prof_audio_chunks = 0
        self._prof_audio_samples = 0
        self._prof_run_chunks = 0
        self._prof_expected_frames = 0
        self._prof_frames_out = 0
        self._prof_frames_used = 0
        self._prof_holds = 0
        self._prof_idle = 0
        self._prof_frames_drop = 0
        self._drop_ditto_frames = 0

        # Backpressure: cap how many frames the SDK may run ahead of playback.
        # TTS delivers a whole answer's audio far faster than real time; without a
        # cap the SDK queues thousands of frames, so an interrupt has to grind the
        # old answer out before the next one starts (~10s). 20 frames ≈ 0.8s lead.
        self._feed_cap = int(os.environ.get("DITTO_FEED_CAP", "20"))
        self._feed_epoch = 0   # bumped on flush_talk to abort in-flight feeding
        self._muted = False    # set on flush; drop audio until the next utterance ('start')
        self._utt_t0 = 0.0             # [timing] utterance audio-in time
        self._utt_gen_pending = False  # log audio-in → first frame GENERATED
        self._utt_show_pending = False # log audio-in → first frame SHOWN (speak start)
        self._tts_start_seq = 0
        self._avatar_start_seq = 0
        self._utt_active = False
        self._utt_audio_chunks = 0
        self._utt_frames_scheduled = 0
        self._final_pending = False
        self._last_ditto_frame_at = time.perf_counter()
        self._tail_audio_fallback = False

        # ── diagnostics (DITTO_DEBUG) — prove writer frames ≠ source frames ──
        self._dbg = bool(os.environ.get("DITTO_DEBUG"))
        self._dbg_dir = os.environ.get("DITTO_DEBUG_DIR", "/tmp/ditto_debug")
        self._dbg_n = int(os.environ.get("DITTO_DEBUG_N", "20"))
        self._dbg_writer_saved = 0
        self._dbg_prev_writer = None
        self._dbg_src_bgr = []
        self._dbg_src_small = []

        # DITTO_SYNC_CSV=/path.csv → per shown frame: how loud the audio bound to
        # it was, and how open the mouth in it was. Cross-correlating the two
        # columns gives the real A/V offset in frames (scripts/ditto_sync_report.py).
        self._sync_csv = os.environ.get("DITTO_SYNC_CSV")
        self._sync_fh = None
        self._sync_n = 0
        self._sync_warned = False

    def _sdk_queue_sizes(self):
        parts = []
        for name, obj in vars(self.sdk).items():
            if not hasattr(obj, "qsize"):
                continue
            try:
                parts.append(f"{name}={obj.qsize()}")
            except Exception:
                pass
        return " ".join(parts)

    def _speech_pending(self):
        # Real playback state, not the profiling counters (tail padding leaves
        # those nonzero forever): audio still waiting for its frame means the
        # SDK owes us output, and queued frames still have to be shown.
        return not self._audio_out.empty() or not self._ditto_frames.empty()

    def _prof_log(self, force=False):
        if not self._prof:
            return
        now = time.perf_counter()
        if not force and now - self._prof_last < 5.0:
            return
        self._prof_last = now
        elapsed = max(now - self._prof_t0, 0.001)
        audio_s = self._prof_audio_samples / 16000.0
        logger.info(
            "[ditto-prof] %.1fs audio=%.2fs chunks=%d run_chunk=%d expected=%d "
            "out=%d used=%d drop=%d hold=%d idle=%d out_fps=%.1f local_q ditto=%d audio=%d %s",
            elapsed, audio_s, self._prof_audio_chunks, self._prof_run_chunks,
            self._prof_expected_frames, self._prof_frames_out,
            self._prof_frames_used, self._prof_frames_drop, self._prof_holds,
            self._prof_idle, self._prof_frames_out / elapsed,
            self._ditto_frames.qsize(), self._audio_out.qsize(),
            self._sdk_queue_sizes())

    def _run_chunk(self, audio, chunksize, keep_frames=None, count_expected=True):
        self._prof_run_chunks += 1
        if count_expected:
            self._prof_expected_frames += chunksize[1]
        if self._utt_active:
            self._utt_frames_scheduled += chunksize[1]
            keep_frames = chunksize[1] if keep_frames is None else keep_frames
            for i in range(chunksize[1]):
                self._frame_keep.put(i < keep_frames)
        self.sdk.run_chunk(audio, chunksize)
        self._prof_log()

    def _flush_pause_packets(self):
        if not self._pause_packets:
            return
        semantic = all(packet[0] for packet in self._pause_packets)
        strength = sum(packet[1] for packet in self._pause_packets) / len(
            self._pause_packets)
        rms = float(np.sqrt(sum(packet[2] for packet in self._pause_packets)
                            / len(self._pause_packets)))
        if semantic:
            self._vad_silent_frames = 2
        elif rms <= self._vad_rms:
            self._vad_silent_frames += 1
        else:
            self._vad_silent_frames = 0
        self._pause_frames.put((
            semantic or self._vad_silent_frames >= 2,
            strength,
        ))
        self._pause_packets.clear()

    def _queue_pause_packet(self, audio, datainfo):
        audio = np.asarray(audio, dtype=np.float32)
        power = float(np.mean(audio * audio)) if audio.size else 0.0
        self._pause_packets.append((
            bool(datainfo.get("semantic_pause")),
            max(0.0, min(float(datainfo.get("lip_close_strength", 0.0)), 1.0)),
            power,
        ))
        if len(self._pause_packets) == _AUDIO_CHUNKS_PER_FRAME:
            self._flush_pause_packets()

    def _next_pause_frame(self):
        try:
            return self._pause_frames.get_nowait()
        except queue.Empty:
            return False, 0.0

    def _clear_pause_state(self):
        pause_frames = getattr(self, "_pause_frames", None)
        if pause_frames is not None:
            _drain_queue(pause_frames)
        pause_packets = getattr(self, "_pause_packets", None)
        if pause_packets is not None:
            pause_packets.clear()
        pause_motion = getattr(self, "_pause_motion", None)
        if pause_motion is not None:
            pause_motion.reset()
        self._vad_silent_frames = 0

    # TTS pushes 20ms float32 chunks here (override base, which routes to asr).
    def put_audio_frame(self, audio_chunk, datainfo: dict = {}):
        # After an interrupt we stay muted until the NEXT utterance begins, so the
        # tail of the sentence that was mid-synthesis when Stop/mic was pressed is
        # dropped instead of finishing. Each new utterance's first chunk carries
        # status='start' → unmute.
        epoch = self._feed_epoch
        if datainfo.get('status') == 'start' and not self._utt_active:
            self._muted = False
            self._tail_audio_fallback = False
            self._utt_active = True
            self._utt_audio_chunks = 0
            self._utt_frames_scheduled = 0
            self._tts_start_seq += 1
            self._utt_t0 = time.perf_counter()   # [timing] this utterance's audio arrived
            self._utt_gen_pending = True
            self._utt_show_pending = True
        if self._muted:
            return
        a = np.asarray(audio_chunk, dtype=np.float32)
        # Backpressure: block the TTS feed while the SDK is >_feed_cap frames ahead,
        # so the SDK backlog (and thus interrupt latency) stays bounded. Bails if a
        # flush bumps the epoch or we're shutting down. Silence must pass through:
        # it completes Ditto's final model batch, so blocking it here deadlocks the
        # final marker behind the frames that only that silence can generate.
        while (np.any(a) and self._prof_expected_frames - self._prof_frames_out
               - self._prof_frames_drop >= self._feed_cap):
            if self._feed_epoch != epoch or (getattr(self, 'quit_event', None) is not None
                                             and self.quit_event.is_set()):
                return
            time.sleep(0.008)
        if self._feed_epoch != epoch:
            return
        if self._utt_active:
            self._utt_audio_chunks += 1
        self._prof_audio_chunks += 1
        self._prof_audio_samples += len(a)
        # queued for the speaker, in the same order it drives the mouth
        self._audio_out.put((a, datainfo))
        self._queue_pause_packet(a, datainfo)
        # accumulate and drive Ditto's mouth with a sliding 6480-sample window
        self._feat_buf = np.concatenate([self._feat_buf, a])
        while self._feat_pos + _SPLIT_LEN <= len(self._feat_buf):
            self._run_chunk(self._feat_buf[self._feat_pos:self._feat_pos + _SPLIT_LEN], _CHUNKSIZE)
            self._feat_pos += _HOP
        # drop consumed history; nothing before the next window start is needed
        if self._feat_pos:
            self._feat_buf = self._feat_buf[self._feat_pos:]
            self._feat_pos = 0
        # end of an utterance: pad-and-flush the tail so all speech gets frames,
        # then reset — otherwise leftover audio drifts into the next utterance.
        if datainfo.get('status') == 'end':
            self._flush_pause_packets()
            logger.info("ditto final audio received: flushing tail")
            self._final_pending = True
            self._flush_tail()
            self._utt_active = False

    def _flush_tail(self):
        pos = self._feat_pos
        for keep_frames in _tail_frame_counts(
                self._utt_audio_chunks, self._utt_frames_scheduled, _CHUNKSIZE[1]):
            window = self._feat_buf[pos:pos + _SPLIT_LEN]
            if len(window) < _SPLIT_LEN:
                window = np.pad(window, (0, _SPLIT_LEN - len(window)))
            self._run_chunk(window, _CHUNKSIZE, keep_frames=keep_frames)
            pos += _HOP
        # Flush the half batch the SDK would otherwise strand (see
        # _alignment_flush_chunks). keep_frames=0 marks every frame it generates
        # as padding, so _on_frame drops them and playback keeps its real pairing.
        for _ in range(_alignment_flush_chunks(
                self._prof_run_chunks,
                getattr(self.sdk.audio2motion, "valid_clip_len", 10),
                _CHUNKSIZE[1])):
            self._run_chunk(np.zeros(_SPLIT_LEN, dtype=np.float32), _CHUNKSIZE,
                            keep_frames=0)
        self._feat_buf = np.full(_PREPAD, 0.0, dtype=np.float32)
        self._feat_pos = 0
        self._prof_log(force=True)

    def flush_talk(self):
        """Stop talking NOW (the Stop/interrupt button). The base only stops NEW
        TTS; the real speech lives in our buffers + the SDK's backlog, so we also:
          - reset the sliding-window buffer (no new chunks from buffered audio),
          - drop the frames the SDK still owes for audio already fed to run_chunk,
          - empty the generated-frame and audio queues the pump is draining.
        Without this, /interrupt_talk leaves the buffered answer playing to the end."""
        self._feed_epoch += 1                      # abort any backpressured put_audio_frame
        self._muted = True                         # drop the interrupted utterance's tail until next 'start'
        self.speaking = False
        self._tts_start_seq = 0
        self._avatar_start_seq = 0
        self._utt_active = False
        self._utt_audio_chunks = 0
        self._utt_frames_scheduled = 0
        self._final_pending = False
        self._tail_audio_fallback = False
        super().flush_talk()                       # stop TTS feeding new text
        self._feat_buf = np.full(_PREPAD, 0.0, dtype=np.float32)
        self._feat_pos = 0
        pending = self._prof_expected_frames - self._prof_frames_out - self._prof_frames_drop
        if pending > 0:
            self._drop_ditto_frames = _reserve_drop_frames(
                self._drop_ditto_frames, pending)
        _drain_queue(self._audio_out)
        _drain_queue(self._ditto_frames)
        _drain_queue(self._frame_keep)
        self._clear_pause_state()
        logger.info("ditto flush_talk: cleared buffered speech, swallowing %d in-flight frames",
                    max(0, pending))

    def _on_frame(self, frame_rgb):
        frame = np.asarray(frame_rgb)
        if frame.ndim != 3:
            return
        self._last_ditto_frame_at = time.perf_counter()
        if self._drop_ditto_frames:
            self._drop_ditto_frames -= 1
            self._prof_frames_drop += 1
            self._prof_log()
            return
        try:
            keep_frame = self._frame_keep.get_nowait()
        except queue.Empty:
            keep_frame = True
        if not keep_frame:
            self._prof_frames_drop += 1
            self._prof_log()
            return
        # The final-audio fallback owns the tail once the SDK has stopped
        # producing. A late callback must not steal those audio packets again.
        if self._tail_audio_fallback:
            self._prof_frames_drop += 1
            self._prof_log()
            return
        self._prof_frames_out += 1
        if self._utt_gen_pending:
            self._utt_gen_pending = False
            logger.info("[ditto-timing] VIDEO first frame GENERATED: %.2fs after audio in (window + diffusion)",
                        time.perf_counter() - self._utt_t0)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self._dbg and self._dbg_writer_saved < self._dbg_n:
            self._dbg_dump_writer(frame_bgr)
        # Bind this frame to the audio that drove it, so playback can never
        # slip the two apart (see _take_audio_pair).
        self._ditto_frames.put((frame_bgr, _take_audio_pair(self._audio_out)))
        self._prof_log()

    def _sync_log(self, frame_bgr, frame_audio):
        """One row per shown frame: audio loudness vs mouth openness.

        DITTO_SYNC_CSV is diagnostic-only: use Ditto's already-loaded face mesh
        and normalized inner-lip distance instead of a framing-dependent ROI."""
        if self._sync_fh is None:
            self._sync_fh = open(self._sync_csv, "w", buffering=1)
            self._sync_fh.write("frame,audio_rms,mouth_open\n")
        rms = 0.0
        for a, _ in (frame_audio or []):
            if a is not None:
                rms = max(rms, float(np.sqrt(np.mean(np.square(a)))))
        mouth_open = float("nan")
        try:
            mesh = self.sdk.avatar_registrar.source2info.landmark478(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).reshape(-1, 3)
            h, w = frame_bgr.shape[:2]
            xy = mesh[:, :2] * np.array([w, h], dtype=np.float32)
            mouth_width = float(np.linalg.norm(xy[61] - xy[291]))
            if mouth_width > 1e-6:
                mouth_open = float(
                    np.linalg.norm(xy[13] - xy[14]) / mouth_width)
        except Exception as exc:
            if not self._sync_warned:
                self._sync_warned = True
                logger.warning("ditto sync mouth landmark failed: %s", exc)
        self._sync_fh.write(f"{self._sync_n},{rms:.6f},{mouth_open:.6f}\n")
        self._sync_n += 1

    # ── Diagnostics (DITTO_DEBUG=1) ─────────────────────────────────────────
    # Proves whether Ditto is GENERATING or just replaying the source video by
    # dumping the first DITTO_DEBUG_N writer + source frames and logging diffs.
    # Off unless enabled; only those first N writer frames pay any cost.
    @staticmethod
    def _dbg_small(img_bgr):
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _dbg_setup_sources(self):
        os.makedirs(self._dbg_dir, exist_ok=True)
        src_rgb = self.sdk.source_info.get("img_rgb_lst") or []
        self._dbg_src_bgr = [cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR) for f in src_rgb]
        self._dbg_src_small = [self._dbg_small(f) for f in self._dbg_src_bgr]
        for i, f in enumerate(self._dbg_src_bgr[:self._dbg_n]):
            cv2.imwrite(os.path.join(self._dbg_dir, f"source_{i:04d}.jpg"), f)
        logger.info("[ditto-dbg] dumped %d/%d source frames to %s",
                    min(self._dbg_n, len(self._dbg_src_bgr)),
                    len(self._dbg_src_bgr), self._dbg_dir)

    def _dbg_dump_writer(self, writer_bgr):
        i = self._dbg_writer_saved
        self._dbg_writer_saved += 1
        cv2.imwrite(os.path.join(self._dbg_dir, f"frame_{i:04d}.jpg"), writer_bgr)
        w = self._dbg_small(writer_bgr)
        # best-matching source frame — alignment-independent. If the writer is
        # merely replaying the source, SOME source frame is near-identical (~0).
        best_i, best_d = -1, 1e9
        for si, s in enumerate(self._dbg_src_small):
            d = float(np.mean(np.abs(w - s)))
            if d < best_d:
                best_d, best_i = d, si
        # same-index diff, for reference (fuzzy: the warm-up drop offsets it)
        aligned = (float(np.mean(np.abs(w - self._dbg_src_small[i])))
                   if i < len(self._dbg_src_small) else float("nan"))

        # mouth = lower-centre band, where Ditto's generated motion shows up
        def mouth(x):
            h, ww = x.shape
            return x[int(h * 0.55):int(h * 0.95), int(ww * 0.30):int(ww * 0.70)]
        mouth_d = (float(np.mean(np.abs(mouth(w) - mouth(self._dbg_src_small[best_i]))))
                   if best_i >= 0 else float("nan"))
        if best_i >= 0:
            cv2.imwrite(os.path.join(self._dbg_dir, f"match_{i:04d}_src{best_i:04d}.jpg"),
                        self._dbg_src_bgr[best_i])
        prev_d = (float(np.mean(np.abs(w - self._dbg_prev_writer)))
                  if self._dbg_prev_writer is not None else float("nan"))
        self._dbg_prev_writer = w
        logger.info("[ditto-dbg] writer#%02d best_src=%d full=%.2f mouth=%.2f "
                    "aligned=%.2f prev=%.2f (0-255 mean-abs)",
                    i, best_i, best_d, mouth_d, aligned, prev_d)
        if i == self._dbg_n - 1:
            logger.info("[ditto-dbg] DONE → %s. Read as: full≈0 every frame = writer "
                        "is replaying source (no generation). prev≈0 = static frames. "
                        "Real generation = small full/aligned but clear mouth AND "
                        "nonzero prev.", self._dbg_dir)

    def _load_idle_bgr(self):
        avatar_dir = os.path.dirname(self.source_path)
        # Generation and playback are separate decisions. A generated clip may
        # exist for comparison, but normal idle must remain the recorded,
        # closed-mouth idle.mp4 unless explicitly opted in.
        use_generated = os.environ.get("DITTO_USE_GENERATED_IDLE", "0") == "1"
        names = ("idle.generated.mp4", "idle.mp4") if use_generated else ("idle.mp4",)
        target_shape = self.sdk.source_info["img_rgb_lst"][0].shape
        for name in names:
            idle_path = os.path.join(avatar_dir, name)
            if not os.path.exists(idle_path):
                continue
            if name == "idle.generated.mp4" and not os.path.exists(idle_path + ".json"):
                logger.warning("ditto generated idle ignored: missing fingerprint metadata")
                continue
            cap = cv2.VideoCapture(idle_path)
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(_resize_idle_frame(frame, target_shape))
            cap.release()
            if frames:
                logger.info("ditto idle video loaded: %s frames=%d", idle_path, len(frames))
                return frames
        return [cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR)
                for f in self.sdk.source_info["img_rgb_lst"]]

    def _neutralize_source_lips(self):
        """Use idle.mp4's closed mouth as the source-video lip baseline."""
        if os.environ.get("DITTO_NEUTRAL_LIPS", "1").lower() in {"0", "false", "no"}:
            return
        idle_path = os.path.join(os.path.dirname(self.source_path), "idle.mp4")
        try:
            count = neutralize_sdk_source_lips(self.sdk, idle_path)
            logger.info("ditto neutral lips: applied idle mouth baseline to %d source frames",
                        count)
        except Exception:
            logger.exception("ditto neutral lips skipped")

    def _pump(self, quit_event: Event):
        # idle:   cycle source frames (smooth animation before/after speech)
        # speech: show Ditto frames; hold last when queue briefly empty (no flicker)
        # Keep draining audio through short writer gaps and the terminal silence
        # tail; waiting for another generated frame can deadlock the final chunk.
        ii = 0
        current_frame = self._idle_bgr[0]
        last_ditto_t = 0.0
        in_speech = False
        _HOLD = float(os.environ.get("DITTO_HOLD", "0.10"))
        _START_BUFFER = int(os.environ.get("DITTO_START_BUFFER", "6"))
        # Positive delays audio; negative delays video. Pairing remains exact at
        # zero, while a measured fixed display/model offset can be compensated.
        _OFFSET_MS = float(os.environ.get("DITTO_AV_OFFSET_MS", "260"))
        _AUDIO_DELAY_CHUNKS, _VIDEO_DELAY_FRAMES = _offset_delays(_OFFSET_MS)
        _END_HOLD = max(_HOLD, _AUDIO_DELAY_CHUNKS * 0.02)
        _FINAL_HOLD = max(0.0, float(os.environ.get("DITTO_FINAL_HOLD_MS", "370")) / 1000.0)
        _IDLE_BLEND_FRAMES = 4
        _IDLE_BLEND_SECONDS = _IDLE_BLEND_FRAMES * 0.04
        _FINAL_STATIC_HOLD = max(0.0, _FINAL_HOLD - _IDLE_BLEND_SECONDS)
        audio_delay = deque()
        video_delay = deque()
        idle_blend = deque()
        idle_thumbs = [_frame_thumb(frame) for frame in self._idle_bgr]
        final_idle_at = None
        dbg_pump_saved = 0

        target = time.perf_counter()
        while not quit_event.is_set():
            now = time.perf_counter()
            got_ditto = False
            frame_audio = None
            force_idle_tick = False
            if final_idle_at is not None and now >= final_idle_at:
                _drain_queue(self._audio_out)
                _drain_queue(self._ditto_frames)
                audio_delay.clear()
                video_delay.clear()
                final_idle_at = None
                self._final_pending = False
                in_speech = False
                self.speaking = False
                ii = _closest_idle_index(current_frame, idle_thumbs)
                blend_targets = [
                    self._idle_bgr[(ii + offset) % len(self._idle_bgr)]
                    for offset in range(_IDLE_BLEND_FRAMES)
                ]
                transition = _blend_to_idle(current_frame, blend_targets)
                if transition:
                    idle_blend.extend(transition)
                    ii = (ii + _IDLE_BLEND_FRAMES) % len(self._idle_bgr)
                force_idle_tick = True
                logger.info("ditto pump: final hold complete -> idle transition")
            try:
                if force_idle_tick or final_idle_at is not None:
                    raise queue.Empty
                # Only wait for the cushion while the SDK still has audio to turn
                # into frames. Once it owes nothing, a short tail would never
                # reach _START_BUFFER, and holding it back deadlocks the return
                # to idle: the frames stay queued, so _speech_pending() is
                # permanently true and the pump never leaves speech.
                if (not in_speech and self._ditto_frames.qsize() < _START_BUFFER
                        and not self._audio_out.empty()):
                    raise queue.Empty
                generated_frame, frame_audio = self._ditto_frames.get_nowait()
                got_ditto = True
                if not in_speech:
                    audio_delay.clear()
                    audio_delay.extend([(None, {})] * _AUDIO_DELAY_CHUNKS)
                    video_delay.clear()
                    idle_blend.clear()
                in_speech = True
                self.speaking = True
                last_ditto_t = now
                self._prof_frames_used += 1
                video_delay.append(generated_frame)
                if len(video_delay) > _VIDEO_DELAY_FRAMES:
                    current_frame = video_delay.popleft()
                if self._utt_show_pending:
                    self._utt_show_pending = False
                    self._avatar_start_seq += 1
                    logger.info("[ditto-timing] SPEAK START (audio+video out to WebRTC): %.2fs after audio in",
                                time.perf_counter() - self._utt_t0)
                # confirm what's shown DURING speech is a writer frame (not idle):
                # these should visually equal frame_*.jpg from _dbg_dump_writer.
                if self._dbg and dbg_pump_saved < self._dbg_n:
                    cv2.imwrite(os.path.join(self._dbg_dir,
                                f"pump_ditto_{dbg_pump_saved:04d}.jpg"), current_frame)
                    dbg_pump_saved += 1
            except queue.Empty:
                if final_idle_at is not None:
                    pass  # hold the last generated frame until the blend begins
                elif in_speech and video_delay:
                    current_frame = video_delay.popleft()
                    got_ditto = True
                    last_ditto_t = now
                elif (in_speech and self._final_pending and not self._utt_active
                      and not self._audio_out.empty()
                      and self._ditto_frames.empty()
                      and (now - self._last_ditto_frame_at) >= 0.5):
                    if not self._tail_audio_fallback:
                        self._tail_audio_fallback = True
                        logger.info("ditto pump: SDK tail stalled; draining final audio")
                    frame_audio = _take_audio_pair(self._audio_out)
                elif (in_speech and not self._speech_pending()
                      and not self._final_pending
                      and (now - last_ditto_t) > _END_HOLD):
                    in_speech = False  # speech done, resume idle
                    self.speaking = False
                    audio_delay.clear()
                    video_delay.clear()
                    logger.info("ditto pump: speech drained -> idle")
                if not in_speech:
                    self.speaking = False
                    if idle_blend:
                        current_frame = idle_blend.popleft()
                    else:
                        current_frame = self._idle_bgr[ii % len(self._idle_bgr)]
                        ii += 1
                    self._prof_idle += 1
                else:
                    self._prof_holds += 1
                # else: mid-speech gap — hold last Ditto frame (no flicker)

            self.output.push_video_frame(current_frame)
            self.record_video_data(current_frame)

            # Measure what was actually shown against what was actually heard,
            # so lip-sync is a number instead of an opinion (DITTO_SYNC_CSV).
            if self._sync_csv and in_speech:
                self._sync_log(current_frame, frame_audio)

            # Play exactly the audio bound to this frame. Holding a frame or
            # sitting idle emits silence instead of stealing the next frame's
            # audio, so the two can never drift apart.
            if frame_audio:
                audio_delay.extend(frame_audio)
            final_audio_played = False
            for _ in range(_AUDIO_CHUNKS_PER_FRAME):
                a, ud = audio_delay.popleft() if audio_delay else (None, {})
                pcm = _SILENCE if a is None else (a * 32767).astype(np.int16)
                self.output.push_audio_frame(pcm, ud)
                self.record_audio_data(pcm)
                final_audio_played = final_audio_played or _is_final_audio_event(ud)
            if final_audio_played:
                # The SDK may strand final silence and let the playback fallback
                # drain it without producing motion. Never carry those pause
                # masks into the next utterance.
                self._clear_pause_state()
                self._final_pending = False
                final_idle_at = time.perf_counter() + _FINAL_STATIC_HOLD
                logger.info(
                    "ditto pump: final audio played; holding %.0fms then blending %.0fms to idle",
                    _FINAL_STATIC_HOLD * 1000.0, _IDLE_BLEND_SECONDS * 1000.0)
            elif (self._final_pending and not self._utt_active
                  and self._audio_out.empty() and self._ditto_frames.empty()
                  and not audio_delay and not video_delay):
                self._clear_pause_state()
                self._final_pending = False
                final_idle_at = time.perf_counter() + _FINAL_STATIC_HOLD
                logger.info(
                    "ditto pump: final queues drained; holding %.0fms then blending %.0fms to idle",
                    _FINAL_STATIC_HOLD * 1000.0, _IDLE_BLEND_SECONDS * 1000.0)

            target += 0.04
            dt = target - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                target = time.perf_counter()
            self._prof_log()
        logger.info('ditto pump stop')
        self.speaking = False
        self._prof_log(force=True)

    def render(self, quit_event):
        self.quit_event = quit_event
        self.init_customindex()

        # Register source; output_path is a dummy — we replace the writer below.
        # Two speed knobs (env-tunable) — the pytorch backend can't hit 25fps at
        # full res/steps, and below-real-time output makes the mouth stutter:
        #   DITTO_STEPS     LMDM diffusion denoise steps (default 50). Biggest
        #                   lever; 15 is ~3x faster with little visible loss.
        #   DITTO_MAX_SIZE  longest-edge the pipeline processes/outputs at
        #                   (default 1920). 640 is plenty for a talking head.
        setup_kwargs = setup_kwargs_from_env()
        logger.info("ditto setup kwargs: %s", setup_kwargs)
        _t = time.perf_counter()
        self.sdk.setup(self.source_path, f"/tmp/ditto_{self.opt.sessionid}.mp4",
                       **setup_kwargs)
        logger.info("[ditto-timing] sdk.setup (source processing): %.2fs", time.perf_counter() - _t)
        self._neutralize_source_lips()
        install_lip_response(
            self.sdk, os.environ.get("DITTO_LIP_RESPONSE", "1.2"))
        self._pause_motion = install_semantic_pause_closure(
            self.sdk, self._next_pause_frame,
            os.environ.get("DITTO_PAUSE_CLOSE_MS", "120"),
            os.environ.get("DITTO_AV_OFFSET_MS", "260"))
        # Hijack Ditto's file writer → frames flow to WebRTC (no queue race).
        self.sdk.writer = _FrameSink(self._on_frame)
        if self._dbg:
            self._dbg_setup_sources()
        # DITTO_PROF=1 → time each per-frame stage to find the throughput cap.
        if os.environ.get("DITTO_PROF"):
            for _s in ("audio2motion", "motion_stitch", "warp_f3d", "decode_f3d", "putback"):
                setattr(self.sdk, _s, _Prof(_s, getattr(self.sdk, _s)))
            logger.info("ditto profiling ON")
        # Idle frames = the full source frames Ditto composites onto (RGB→BGR).
        self._idle_bgr = self._load_idle_bgr()

        # Online Ditto's first valid_clip_len frames establish d0 and produce no
        # output (stream_pipeline_online.py:432-440). Prime the complete batch
        # with silence; otherwise the first real speech frames are swallowed by
        # d0 and every audio/pause marker is shifted away from its mouth frame.
        valid_clip_len = getattr(self.sdk.audio2motion, "valid_clip_len", 10)
        priming_chunks = _priming_chunk_count(valid_clip_len, _CHUNKSIZE[1])
        for _ in range(priming_chunks):
            self._run_chunk(np.zeros(_SPLIT_LEN, dtype=np.float32), _CHUNKSIZE,
                            count_expected=False)
        logger.info("ditto JIT/d0 warm-up queued: %d chunks (%d frames)",
                    priming_chunks, priming_chunks * _CHUNKSIZE[1])

        self.tts.render(quit_event)          # TTS → put_audio_frame → run_chunk
        self.output.start()

        pump_quit = Event()
        pump = Thread(target=self._pump, args=(pump_quit,))
        pump.start()

        logger.info("[ditto-timing] total build→ready: %.2fs (engine load + setup + warmup queued)",
                    time.perf_counter() - self._t_build)
        logger.info('ditto render start')
        while not quit_event.is_set():
            time.sleep(0.1)
        logger.info('ditto render stop')

        pump_quit.set()
        pump.join()
        try:
            self.sdk.close()
        except Exception:
            logger.exception("ditto sdk close error")
        self.output.stop()

# ponytail: _ditto_frames / _audio_out are unbounded. If TTS outruns real-time
# on a long utterance they grow (latency creeps up, sync holds). Cap with a
# bounded Queue + drop-oldest only if that actually bites in a demo.
