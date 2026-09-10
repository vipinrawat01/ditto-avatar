###############################################################################
#  Audio processing utility functions
###############################################################################

import io
import numpy as np


def pcm_to_float32(pcm_bytes: bytes, sample_width: int = 2) -> np.ndarray:
    """Convert PCM bytes to a float32 numpy array in the range [-1.0, 1.0]."""
    if sample_width == 2:
        data = np.frombuffer(pcm_bytes, dtype=np.int16)
        return data.astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(pcm_bytes, dtype=np.int32)
        return data.astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")


def float32_to_pcm(audio: np.ndarray, sample_width: int = 2) -> bytes:
    """Convert a float32 numpy array to PCM bytes."""
    if sample_width == 2:
        data = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
        return data.tobytes()
    elif sample_width == 4:
        data = (audio * 2147483648.0).clip(-2147483648, 2147483647).astype(np.int32)
        return data.tobytes()
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")


def resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio from one sample rate to another."""
    if from_rate == to_rate:
        return audio
    try:
        import resampy
        return resampy.resample(audio, from_rate, to_rate)
    except ImportError:
        # fallback: linear interpolation
        ratio = to_rate / from_rate
        n_samples = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, n_samples)
        return np.interp(indices, np.arange(len(audio)), audio)
