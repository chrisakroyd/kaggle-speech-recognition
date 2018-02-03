import librosa
import numpy as np

SAMPLE_RATE = 16000
TARGET_DURATION = 16000


# AUDIO PREPROCESSING FUNCTIONS
def shift_audio(audio, ms_shift=100):
    ms = 16
    time_shift_dist = int(np.random.uniform(-(ms_shift * ms), (ms_shift * ms)))
    audio = np.roll(audio, time_shift_dist)
    return audio


def get_silence_slice(audio):
    silence_start_idx = np.random.randint(audio.shape[0] - TARGET_DURATION)
    silence_slice = audio[silence_start_idx: silence_start_idx + TARGET_DURATION]
    return silence_slice


# Add in random background noise
def add_background_noises(audio, background_wav, volume_range=0.1):
    bg_audio = background_wav
    bg_audio_start_idx = np.random.randint(bg_audio.shape[0] - TARGET_DURATION)
    bg_audio_slice = bg_audio[bg_audio_start_idx: bg_audio_start_idx + 16000] * np.random.uniform(0, volume_range)
    return audio + bg_audio_slice


# adjust volume(amplitude)
def amplitude_scaling(audio, multiplier=0.2):
    return audio * np.random.uniform(1.0 - multiplier, 1.0 + multiplier)


# Adjust the length of the signal
def time_strech(audio, multiplier=0.2):
    rate = np.random.uniform(1.0 - multiplier, 1.0 + multiplier)
    return librosa.effects.time_stretch(audio, rate)


# Adjust the pitch of the signal
def pitch_scaling(audio, max_step_shift=2.0):
    steps = np.random.uniform(-max_step_shift, max_step_shift)
    return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=steps)


def trim_pad_audio(audio):
    duration = len(audio)
    if duration < TARGET_DURATION:
        audio = np.pad(audio, (TARGET_DURATION - audio.size, 0), mode='constant')
    elif duration > TARGET_DURATION:
        audio = audio[0:TARGET_DURATION]
    return audio
