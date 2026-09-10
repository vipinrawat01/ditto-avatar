import os
import time
import numpy as np
import soundfile as sf
import resampy

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

class IndexTTS2(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        # IndexTTS2 configuration parameters
        self.server_url = opt.TTS_SERVER  # Gradio server address, e.g. "http://127.0.0.1:7860/"
        self.ref_audio_path = opt.REF_FILE  # Reference audio file path
        self.max_tokens = getattr(opt, 'MAX_TOKENS', 120)  # Maximum number of tokens

        # Initialize the Gradio client
        try:
            from gradio_client import Client, handle_file
            self.client = Client(self.server_url)
            self.handle_file = handle_file
            logger.info(f"IndexTTS2 Gradio client initialized successfully: {self.server_url}")
        except ImportError:
            logger.error("IndexTTS2 requires gradio_client: pip install gradio_client")
            raise
        except Exception as e:
            logger.error(f"IndexTTS2 Gradio client initialization failed: {e}")
            raise
        
    def txt_to_audio(self, msg):
        text, textevent = msg
        try:
            # First split the text
            segments = self.split_text(text)
            if not segments:
                logger.error("IndexTTS2 text splitting failed")
                return

            logger.info(f"IndexTTS2 split text into {len(segments)} segments")

            # Generate audio for each segment in a loop
            for i, segment_text in enumerate(segments):
                if self.state != State.RUNNING:
                    break

                logger.info(f"IndexTTS2 generating audio segment {i+1}/{len(segments)}...")
                audio_file = self.indextts2_generate(segment_text)

                if audio_file:
                    # Create event information for each segment
                    segment_msg = (segment_text, textevent)
                    self.file_to_stream(audio_file, segment_msg, is_first=(i==0), is_last=(i==len(segments)-1))
                else:
                    logger.error(f"IndexTTS2 audio generation failed for segment {i+1}")

        except Exception as e:
            logger.exception(f"IndexTTS2 txt_to_audio error: {e}")

    def split_text(self, text):
        """Split text using the IndexTTS2 API"""
        try:
            logger.info(f"IndexTTS2 starting text splitting, length: {len(text)}")

            # Call the text-splitting API
            result = self.client.predict(
                text=text,
                max_text_tokens_per_segment=self.max_tokens,
                api_name="/on_input_text_change"
            )
            
            # Parse the split result
            if 'value' in result and 'data' in result['value']:
                data = result['value']['data']
                logger.info(f"IndexTTS2 split into {len(data)} segments total")

                segments = []
                for i, item in enumerate(data):
                    index = item[0] + 1
                    segment_content = item[1]
                    token_count = item[2]
                    logger.info(f"Segment {index}: {len(segment_content)} characters, {token_count} tokens")
                    segments.append(segment_content)

                return segments
            else:
                logger.error(f"IndexTTS2 text split result has an unexpected format: {result}")
                return [text]  # If splitting fails, return the original text

        except Exception as e:
            logger.exception(f"IndexTTS2 text splitting failed: {e}")
            return [text]  # If splitting fails, return the original text

    def indextts2_generate(self, text):
        """Call the IndexTTS2 Gradio API to generate speech"""
        start = time.perf_counter()
        
        try:
            # Call the gen_single API
            result = self.client.predict(
                emo_control_method="Same as the voice reference",
                prompt=self.handle_file(self.ref_audio_path),
                text=text,
                emo_ref_path=self.handle_file(self.ref_audio_path),
                emo_weight=0.8,
                vec1=0.5,
                vec2=0,
                vec3=0,
                vec4=0,
                vec5=0,
                vec6=0,
                vec7=0,
                vec8=0,
                emo_text="",
                emo_random=False,
                max_text_tokens_per_segment=self.max_tokens,
                param_16=True,
                param_17=0.8,
                param_18=30,
                param_19=0.8,
                param_20=0,
                param_21=3,
                param_22=10,
                param_23=1500,
                api_name="/gen_single"
            )
            
            end = time.perf_counter()
            logger.info(f"IndexTTS2 segment generation complete, elapsed: {end-start:.2f}s")

            # Return the path to the generated audio file
            if 'value' in result:
                audio_file = result['value']
                return audio_file
            else:
                logger.error(f"IndexTTS2 result has an unexpected format: {result}")
                return None

        except Exception as e:
            logger.exception(f"IndexTTS2 API call failed: {e}")
            return None

    def file_to_stream(self, audio_file, msg, is_first=False, is_last=False):
        """Convert an audio file into an audio stream"""
        text, textevent = msg
        
        try:
            # Read the audio file
            stream, sample_rate = sf.read(audio_file)
            logger.info(f'IndexTTS2 audio file {sample_rate}Hz: {stream.shape}')

            # Convert to float32
            stream = stream.astype(np.float32)

            # If multi-channel, use only the first channel
            if stream.ndim > 1:
                logger.info(f'IndexTTS2 audio has {stream.shape[1]} channels, using only the first')
                stream = stream[:, 0]

            # Resample to the target sample rate
            if sample_rate != self.sample_rate and stream.shape[0] > 0:
                logger.info(f'IndexTTS2 resampling: {sample_rate}Hz -> {self.sample_rate}Hz')
                stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

            # Send the audio stream in chunks
            streamlen = stream.shape[0]
            idx = 0
            first_chunk = True

            while streamlen >= self.chunk and self.state == State.RUNNING:
                eventpoint = None

                # Only send the start event on the first chunk of the first segment
                if is_first and first_chunk:
                    eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
                    first_chunk = False

                self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                idx += self.chunk
                streamlen -= self.chunk

            # Only send the end event on the last segment
            if is_last:
                eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
                self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

            # Clean up the temporary file
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    logger.info(f"IndexTTS2 deleted temporary file: {audio_file}")
            except Exception as e:
                logger.warning(f"IndexTTS2 failed to delete temporary file: {e}")

        except Exception as e:
            logger.exception(f"IndexTTS2 audio stream processing failed: {e}")
