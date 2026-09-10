import os
import re
import time
import numpy as np
from elevenlabs.client import ElevenLabs

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

# Loudness matching across segments. Each semantic segment is its own ElevenLabs
# request, and ElevenLabs picks the emphasis per request: "Certainly!" comes back
# far hotter than a long sentence, so the avatar's volume jumps between them.
_TARGET_RMS = float(os.environ.get("TTS_TARGET_RMS", "0.06"))   # ~ -24 dBFS
_MAX_GAIN = float(os.environ.get("TTS_MAX_GAIN", "8.0"))
_CEILING = float(os.environ.get("TTS_CEILING", "0.95"))
_ESTIMATE_FRAMES = int(os.environ.get("TTS_ESTIMATE_FRAMES", "8"))   # 8 x 20ms
_SPEECH_GATE_RMS = float(os.environ.get("TTS_SPEECH_GATE_RMS", "0.003"))
_PHONEME_TAIL_FRAMES = max(
    1, int(round(float(os.environ.get("TTS_PHONEME_CLOSE_MS", "80")) / 20.0)))
_N_CLOSE_STRENGTH = float(os.environ.get("TTS_N_CLOSE_STRENGTH", "0.45"))


def _terminal_lip_close_strength(text):
    """Return a conservative visual closure hint for the final English sound.

    This is deliberately a small spelling heuristic, not a phoneme recognizer.
    It covers the visible closures requested for common segment endings without
    adding a pronunciation model to the streaming path.
    """
    words = re.findall(r"[A-Za-z]+", text or "")
    if not words:
        return 0.0
    word = words[-1].lower()
    if re.search(r"(?:[mbp]|[mbp]e)$", word):
        return 1.0
    if re.search(r"(?:n|ne)$", word):
        return max(0.0, min(_N_CLOSE_STRENGTH, 1.0))
    return 0.0


def _segment_gain(frames):
    """One gain for a whole segment, from its first frames.

    The jumps are BETWEEN segments, not inside one, so a single frozen gain
    removes them while leaving the segment's own dynamics untouched — unlike a
    continuously adapting AGC, which would pump inside a sentence.

    ponytail: estimated from a 160ms prefix rather than the whole segment.
    Buffering a whole segment would add its generation tail to every gap between
    segments, and TTS loudness is near-stationary within one sentence anyway.
    Bounded by both an RMS target and the prefix's peak headroom, so a hot
    segment is pulled down and a quiet one lifted without clipping.
    """
    if not frames:
        return 1.0
    audio = np.concatenate(frames)
    # ElevenLabs frequently starts a request with quiet consonants or padding.
    # Measuring that padding as speech can select the maximum boost and make the
    # following vowel jump in volume. Estimate loudness only from active 20ms
    # frames and use their median, which is stable without buffering the segment.
    frame_rms = np.asarray([
        float(np.sqrt(np.mean(np.square(frame)))) for frame in frames if len(frame)
    ])
    active_rms = frame_rms[frame_rms >= _SPEECH_GATE_RMS]
    if not len(active_rms):
        return 1.0
    rms = float(np.median(active_rms))
    peak = float(np.max(np.abs(audio)))
    if rms < 1e-4 or peak < 1e-4:
        return 1.0                      # silence carries no level to match
    # Only the boost is capped, so near-silence is not amplified into noise.
    # Attenuation is unbounded: a hot segment needs whatever cut it takes.
    return float(min(_TARGET_RMS / rms, _CEILING / peak, _MAX_GAIN))


def _level_frame(frame, gain):
    """Apply one stable segment gain and a non-clipping emergency peak limit."""
    output = np.asarray(frame, dtype=np.float32) * float(gain)
    if not output.size:
        return output

    peak = float(np.max(np.abs(output)))
    if peak > _CEILING:
        # Scale the complete frame instead of clipping individual samples. This
        # guarantees headroom without flat-topped clipping distortion.
        output *= _CEILING / peak
    return output


@register("tts", "elevenlabs")
class ElevenLabsTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self._client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
        self._voice_id = opt.REF_FILE or "SEWXl8lPSO01tdGbWECX"
        self._previous_text = ""
        self._first_frame = True
        self._level_energy = 0.0
        self._level_samples = 0
        self._level_peak = 0.0

    def flush_talk(self):
        super().flush_talk()
        self._previous_text = ""

    def _emit(self, frames, text, textevent, gain):
        for frame in frames:
            eventpoint = {}
            if self._first_frame:
                eventpoint = {"status": "start", "text": text}
                self._first_frame = False
            eventpoint.update(**textevent)
            output = _level_frame(frame, gain)
            self._level_energy += float(np.sum(np.square(output)))
            self._level_samples += output.size
            if output.size:
                self._level_peak = max(
                    self._level_peak, float(np.max(np.abs(output))))
            self.parent.put_audio_frame(output, eventpoint)

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        text = text.strip()
        final = bool(textevent.get("final"))
        if not text:
            if final:
                self._send_silence_tail(text, textevent, True)
            return

        self._first_frame = True
        self._level_energy = 0.0
        self._level_samples = 0
        self._level_peak = 0.0
        started = time.perf_counter()
        # Frames held back while the segment's gain is still being estimated.
        held = []
        phoneme_tail = []
        gain = None

        def queue_frame(frame):
            phoneme_tail.append(frame)
            if len(phoneme_tail) > _PHONEME_TAIL_FRAMES:
                self._emit([phoneme_tail.pop(0)], text, textevent, gain)

        try:
            chunks = self._client.text_to_speech.stream(
                voice_id=self._voice_id,
                text=text,
                model_id="eleven_flash_v2_5",
                output_format="pcm_16000",
                previous_text=self._previous_text or None,
            )
            pcm_buffer = bytearray()
            frame_bytes = self.chunk * np.dtype(np.int16).itemsize
            got_audio = False
            for pcm_chunk in chunks:
                if self.state != State.RUNNING:
                    return
                if not got_audio:
                    got_audio = True
                    logger.info("elevenlabs first audio: %.4fs", time.perf_counter() - started)
                pcm_buffer.extend(pcm_chunk)
                while len(pcm_buffer) >= frame_bytes:
                    raw_frame = bytes(pcm_buffer[:frame_bytes])
                    del pcm_buffer[:frame_bytes]
                    frame = np.frombuffer(raw_frame, dtype=np.int16).astype(np.float32) / 32768.0
                    if gain is not None:
                        queue_frame(frame)
                        continue
                    held.append(frame)
                    if len(held) >= _ESTIMATE_FRAMES:
                        gain = _segment_gain(held)
                        logger.info("elevenlabs segment gain: %.2fx", gain)
                        for buffered_frame in held:
                            queue_frame(buffered_frame)
                        held = []
        except Exception:
            logger.exception("elevenlabs tts error")
            return

        logger.info("elevenlabs stream/feed complete: %.4fs", time.perf_counter() - started)
        usable_bytes = len(pcm_buffer) - (len(pcm_buffer) % 2)
        if usable_bytes:
            frame = np.frombuffer(bytes(pcm_buffer[:usable_bytes]), dtype=np.int16).astype(np.float32) / 32768.0
            frame = np.pad(frame, (0, self.chunk - len(frame)))
            if gain is None:
                held.append(frame)
            else:
                queue_frame(frame)
        if held:
            # Segment shorter than the estimate window: gain it on what we got.
            if gain is None:
                gain = _segment_gain(held)
                logger.info("elevenlabs segment gain: %.2fx (short segment)", gain)
            for buffered_frame in held:
                queue_frame(buffered_frame)

        close_strength = _terminal_lip_close_strength(text)
        tail_count = len(phoneme_tail)
        for index, frame in enumerate(phoneme_tail, 1):
            tail_event = dict(textevent)
            if close_strength:
                tail_event["lip_close_strength"] = close_strength * index / tail_count
            self._emit([frame], text, tail_event, gain)

        if self._level_samples:
            output_rms = (self._level_energy / self._level_samples) ** 0.5
            logger.info(
                "elevenlabs output level: rms=%.4f peak=%.4f target=%.4f",
                output_rms, self._level_peak, _TARGET_RMS)

        self._send_silence_tail(text, textevent, final)
        self._previous_text = text

    def _send_silence_tail(self, text, textevent, final):
        # Sentence pauses keep the mouth closed without resetting Ditto's audio
        # window. Only the final marker flushes the window and returns to idle.
        pause_ms = max(20, int(textevent.get("pause_ms", os.environ.get("DITTO_TAIL_MS", "500"))))
        for index in range((pause_ms + 19) // 20):
            eventpoint = dict(textevent)
            # Ditto receives PCM, not punctuation. Mark these deliberately
            # inserted zero-audio packets so its motion layer can close only
            # the lips during a semantic pause without touching head motion or
            # the final return-to-idle state machine.
            eventpoint["semantic_pause"] = True
            if index * 20 + 20 >= pause_ms:
                eventpoint.update(
                    status="end" if final else "segment_end",
                    text=text,
                    final=final,
                )
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
