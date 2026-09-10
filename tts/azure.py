import os
import numpy as np
import azure.cognitiveservices.speech as speechsdk

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "azuretts")
class AzureTTS(BaseTTS):
    CHUNK_SIZE = 640  # 16kHz, 20ms, 16-bit Mono PCM size
    def __init__(self, opt, parent):
        super().__init__(opt,parent)
        self.audio_buffer = b''
        self.voice = opt.REF_FILE or "zh-CN-XiaoxiaoMultilingualNeural"   # e.g. "zh-CN-XiaoxiaoMultilingualNeural"
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        tts_region = os.getenv("AZURE_TTS_REGION")
        speech_endpoint = f"wss://{tts_region}.tts.speech.microsoft.com/cognitiveservices/websocket/v2"
        self.speech_config = speechsdk.SpeechConfig(subscription=speech_key,endpoint=speech_endpoint)
        self.speech_config.speech_synthesis_voice_name = self.voice
        self.speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm)
        
        # Get the result as an in-memory stream
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
        self.speech_synthesizer.synthesizing.connect(self._on_synthesizing)
        
    def txt_to_audio(self,msg:tuple[str, dict]):
        msg_text, textevent = msg
        ref_file = textevent.get('tts', {}).get('ref_file',self.voice)
        self.speech_config.speech_synthesis_voice_name = ref_file
        # Creating a new synthesizer on the fly for each new speaker configuration can be slow, but we avoid major changes here to keep the code compatible
        # self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
        # self.speech_synthesizer.synthesizing.connect(self._on_synthesizing)

        result=self.speech_synthesizer.speak_text(msg_text)

        # Latency metrics
        fb_latency = int(result.properties.get_property(
            speechsdk.PropertyId.SpeechServiceResponse_SynthesisFirstByteLatencyMs
        ))
        fin_latency = int(result.properties.get_property(
            speechsdk.PropertyId.SpeechServiceResponse_SynthesisFinishLatencyMs
        ))
        logger.info(f"azure audio generation: first-byte latency: {fb_latency} ms, finish latency: {fin_latency} ms, result_id: {result.result_id}")

    # === Callback ===
    def _on_synthesizing(self, evt: speechsdk.SpeechSynthesisEventArgs):
        if evt.result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info("SynthesizingAudioCompleted")
        elif evt.result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = evt.result.cancellation_details
            logger.info(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                if cancellation_details.error_details:
                    logger.info(f"Error details: {cancellation_details.error_details}")        
        if self.state != State.RUNNING:
            self.audio_buffer = b''
            return

        # evt.result.audio_data is the small chunk of raw PCM that just arrived
        self.audio_buffer += evt.result.audio_data
        while len(self.audio_buffer) >= self.CHUNK_SIZE:
            chunk = self.audio_buffer[:self.CHUNK_SIZE]
            self.audio_buffer = self.audio_buffer[self.CHUNK_SIZE:]

            frame = (np.frombuffer(chunk, dtype=np.int16)
                       .astype(np.float32) / 32767.0)
            self.parent.put_audio_frame(frame)
