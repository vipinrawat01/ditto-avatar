import soundfile as sf
import resampy
import numpy as np

def read_audio_file(file_path):
    audio, sample_rate = sf.read(file_path)
    return audio, sample_rate

def change_sample_rate(audio, current_rate, target_rate):
    new_audio = resampy.resample(audio, current_rate, target_rate)
    return new_audio, target_rate

def change_channels(audio, current_channels, target_channels):
    new_audio = audio.reshape(-1, current_channels)
    new_audio = np.tile(new_audio, (1, target_channels // current_channels))
    return new_audio, target_channels

def change_bit_depth(audio, current_depth, target_depth):
    new_audio = audio.astype(np.int16)
    new_audio *= 2 ** (target_depth - current_depth)
    return new_audio, target_depth

def save_audio_file(audio, sample_rate, file_path):
    sf.write(file_path, audio, sample_rate)