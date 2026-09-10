import os
import base64
import time
import threading
import numpy as np
import resampy

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

try:
    import dashscope
    from dashscope.audio.qwen_tts_realtime import (
        QwenTtsRealtime,
        QwenTtsRealtimeCallback,
        AudioFormat,
    )
except ImportError:
    logger.error("QwenTTS requires the dashscope SDK: pip install dashscope>=1.25.11")
    raise


SRC_SR = 24000   # Qwen TTS only supports 24kHz output
DST_SR = 16000   # Project standard sample rate


@register("tts", "qwentts")
class QwenTTS(BaseTTS):
    """
    Alibaba Cloud Tongyi Qianwen real-time text-to-speech (Qwen TTS Realtime)
    Based on the DashScope Python SDK (dashscope >= 1.25.11)
    Uses commit mode: establishes the WebSocket connection only once, and each synthesis is triggered via append_text + commit.

    Requires the environment variable DASHSCOPE_API_KEY to be set.
    Usage:
        python app.py --tts qwentts --REF_FILE Cherry
    Here REF_FILE is used as the voice name, e.g. system voices such as Cherry / Ethan.
    """

    def __init__(self, opt, parent):
        super().__init__(opt, parent)

        # Voice name, reuses the REF_FILE parameter
        self.voice = opt.REF_FILE if opt.REF_FILE else 'Cherry'
        # Model name
        self.model = getattr(opt, 'qwen_tts_model', 'qwen3-tts-flash-realtime')
        # WebSocket URL
        self.ws_url = getattr(opt, 'qwen_tts_url',
                              'wss://dashscope.aliyuncs.com/api-ws/v1/realtime')

        # Set the DashScope API Key
        api_key = getattr(opt, 'dashscope_api_key', None) or os.environ.get('DASHSCOPE_API_KEY')
        if api_key:
            dashscope.api_key = api_key
        else:
            logger.warning("QwenTTS: DASHSCOPE_API_KEY is not set, please set the environment variable or pass it via a parameter")

        # ---------- Internal state ----------
        self._remainder = np.array([], dtype=np.float32)  # 16kHz samples left over from the last resample that were less than one chunk
        self._response_event = threading.Event()
        self._first_chunk = True          # The first audio packet in the sentence currently being synthesized
        self._current_text = ''
        self._current_textevent = {}

        # ---------- Callback class ----------
        tts_ref = self

        class _Callback(QwenTtsRealtimeCallback):
            def on_open(self) -> None:
                logger.info("QwenTTS WebSocket connection established")

            def on_close(self, close_status_code, close_msg) -> None:
                logger.info(f"QwenTTS WebSocket closed: code={close_status_code}, msg={close_msg}")
                tts_ref._response_event.set()

            def on_event(self, response: dict) -> None:
                try:
                    event_type = response.get('type', '')

                    if event_type == 'session.created':
                        logger.info(f"QwenTTS session: {response.get('session', {}).get('id', '')}")

                    elif event_type == 'response.audio.delta':
                        audio_b64 = response.get('delta', '')
                        if audio_b64:
                            pcm_data = base64.b64decode(audio_b64)
                            tts_ref._on_audio_data(pcm_data)

                    elif event_type == 'response.done':
                        logger.info("QwenTTS response done")
                        tts_ref._flush_remainder()
                        tts_ref._response_event.set()

                    elif event_type == 'error':
                        logger.error(f"QwenTTS error: {response}")
                        tts_ref._response_event.set()

                except Exception as e:
                    logger.exception(f"QwenTTS callback handling exception: {e}")

        # ---------- Establish the single connection ----------
        self._callback = _Callback()
        self._tts_client = QwenTtsRealtime(
            model=self.model,
            callback=self._callback,
            url=self.ws_url,
        )
        self._tts_client.connect()
        self._tts_client.update_session(
            voice=self.voice,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,  # Qwen TTS only supports 24kHz output
            sample_rate=16000,
            mode='commit',
        )
        logger.info(f"QwenTTS initialization complete: model={self.model}, voice={self.voice}")

    # ========================== Core methods ==========================

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        t_start = time.perf_counter()

        ref_file = textevent.get('tts', {}).get('ref_file',self.voice)

        # Reset state
        self._remainder = np.array([], dtype=np.float32)
        self._first_chunk = True
        self._current_text = text
        self._current_textevent = textevent
        self._response_event.clear()

        try:
            #logger.info(f"QwenTTS sending text: {text[:80]}...")
            if ref_file != self.voice:
                logger.info(f'ref_file:{ref_file},self.voice:{self.voice}')
                self.voice=ref_file
                self._tts_client.close()
                self._tts_client.connect()
                self._tts_client.update_session(
                    voice=self.voice,
                    response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,  # Qwen TTS only supports 24kHz output
                    sample_rate=16000,
                    mode='commit',
                )
            self._tts_client.append_text(text)
            self._tts_client.commit()

            # Wait for response.done (audio is processed as a stream in the callback)
            self._response_event.wait(timeout=60)

            t_end = time.perf_counter()
            logger.info(f"QwenTTS synthesis complete, elapsed: {t_end - t_start:.2f}s")

        except Exception as e:
            logger.exception(f"QwenTTS txt_to_audio exception: {e}")

    # ========================== Streaming audio processing (called within the callback) ==========================

    def _on_audio_data(self, pcm_data: bytes):
        """Received PCM 24kHz 16bit mono audio; resample it to 16kHz in one pass, then push it out in chunks"""
        if self.state != State.RUNNING:
            self._remainder = np.array([], dtype=np.float32)
            return

        # Whole 24kHz PCM segment -> float32 -> resample to 16kHz in one pass
        samples_16k = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        #samples_16k = resampy.resample(x=samples_24k, sr_orig=SRC_SR, sr_new=DST_SR)

        # Prepend the leftover from last time
        if self._remainder.shape[0] > 0:
            samples_16k = np.concatenate([self._remainder, samples_16k])

        # Push in chunks of self.chunk (320 samples = 20ms @16kHz)
        idx = 0
        total = samples_16k.shape[0]
        while total - idx >= self.chunk and self.state == State.RUNNING:
            frame = samples_16k[idx:idx + self.chunk]
            eventpoint = {}
            if self._first_chunk:
                eventpoint = {'status': 'start', 'text': self._current_text}
                self._first_chunk = False
            eventpoint.update(**self._current_textevent)
            self.parent.put_audio_frame(frame, eventpoint)
            idx += self.chunk

        # Keep what is less than one chunk for next time
        self._remainder = samples_16k[idx:] if idx < total else np.array([], dtype=np.float32)

    def _flush_remainder(self):
        """Synthesis finished; push the remaining samples and send the end event"""
        if self.state != State.RUNNING:
            self._remainder = np.array([], dtype=np.float32)
            return

        # Push the remaining complete chunks
        if self._remainder.shape[0] >= self.chunk:
            idx = 0
            total = self._remainder.shape[0]
            while total - idx >= self.chunk and self.state == State.RUNNING:
                frame = self._remainder[idx:idx + self.chunk]
                eventpoint = {}
                if self._first_chunk:
                    eventpoint = {'status': 'start', 'text': self._current_text}
                    self._first_chunk = False
                eventpoint.update(**self._current_textevent)
                self.parent.put_audio_frame(frame, eventpoint)
                idx += self.chunk

        self._remainder = np.array([], dtype=np.float32)

        # Send the end event
        eventpoint = {'status': 'end', 'text': self._current_text}
        eventpoint.update(**self._current_textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

    def stop_tts(self):
        self._tts_client.close()
        logger.info("QwenTTS closed")