import librosa
import numpy as np

SAMPLE_RATE = 16000
TARGET_DURATION = 16000


# AUDIO PREPROCESSING FUNCTIONS
def shift_audio(audio, ms_shift=100):
    """
    This function shifts the audio by a value between 0 and ms_shift with any cutoff being wrapped around to the other
    side of the list. This is augmentation is performed to make a NN more invariant to over-fitting the audio start
    position.
    :param audio: A list of floating point values representing a wav file.
    :param ms_shift: The maximum shift to be performed
    :return: audio: A list of floating point values randomly shifted.
    """
    ms = 16
    time_shift_dist = int(np.random.uniform(-(ms_shift * ms), (ms_shift * ms)))
    audio = np.roll(audio, time_shift_dist)
    return audio


def get_silence_slice(audio):
    """
    Given the audio for a 'silence' file, returns a random one second long slice from it.
    :param audio: A list of floating point values representing a silence wav file.
    :return: silence_slice: A random slice of silence audio.
    """
    silence_start_idx = np.random.randint(audio.shape[0] - TARGET_DURATION)
    silence_slice = audio[silence_start_idx: silence_start_idx + TARGET_DURATION]
    return silence_slice


# Add in random background noise
def add_background_noises(audio, background_wav, volume_range=0.1):
    """
    Given audio and background noise, mixes the two together giving the background noise a
    volume between 0 and volume_range
    :param audio: A list of floating point values representing a wav file.
    :param background_wav: A list of floating point values representing a silence wav file.
    :param volume_range: A floating point value for the max volume of the background noise
    :return: audio: An audio representation with the background noise mixed in.
    """
    bg_audio = background_wav
    bg_audio_start_idx = np.random.randint(bg_audio.shape[0] - TARGET_DURATION)
    bg_audio_slice = bg_audio[bg_audio_start_idx: bg_audio_start_idx + 16000] * np.random.uniform(0, volume_range)
    return audio + bg_audio_slice


# adjust volume(amplitude)
def amplitude_scaling(audio, multiplier=0.2):
    """
    Scales the amplitude(volume) of an audio clip between 0 and a multiplier to create an audio file that is either
    louder or quieter than the original.
    :param audio: A list of floating point values representing a wav file.
    :param multiplier: A floating point value signifying the max/min the volume should be raised or lowered by.
    :return: audio: The audio array with the amplitude scaled.
    """
    return audio * np.random.uniform(1.0 - multiplier, 1.0 + multiplier)


# Adjust the length of the signal
def time_strech(audio, multiplier=0.2):
    """
    Stretches or shrinks the audio, speeding it up or slowing it down based on a multiplier.
    :param audio: A list of floating point values representing a wav file.
    :param multiplier: A floating point value signifying the max/min the volume should be stretched/shrunk by.
    :return: audio: Time stretched audio.
    """
    rate = np.random.uniform(1.0 - multiplier, 1.0 + multiplier)
    return librosa.effects.time_stretch(audio, rate)


# Adjust the pitch of the signal
def pitch_scaling(audio, max_step_shift=2.0):
    """
    Increase/decreases the pitch of the audio between 0 and max_step_shift steps.
    :param audio: A list of floating point values representing a wav file.
    :param max_step_shift: A floating point value for the steps the pitch should be shifted by.
    :return: audio: Time stretched audio.
    """
    steps = np.random.uniform(-max_step_shift, max_step_shift)
    return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=steps)


def trim_pad_audio(audio):
    """
    Trims or pads the audio to conform to one standard size of one second.
    :param audio: A list of floating point values representing a wav file.
    :return: audio of size 16000
    """
    duration = len(audio)
    if duration < TARGET_DURATION:
        audio = np.pad(audio, (TARGET_DURATION - audio.size, 0), mode='constant')
    elif duration > TARGET_DURATION:
        audio = audio[0:TARGET_DURATION]
    return audio
