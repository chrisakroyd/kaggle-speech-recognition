import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile


SAMPLE_RATE = 16000
TARGET_DURATION = 16000
# MFCC method required DCT filters, define these up here for later reuse.
dct_filters = librosa.filters.dct(40, 40)


def log_spectrogram_signal(wav, sample_rate=SAMPLE_RATE, window_size=20, step_size=10, eps=1e-10):
    """
    A function that loads a wav file and converts it into a log spectrogram representation using
    scipy's signal.
    :param wav: File path to a wav file.
    :param sample_rate: The sample rate of the original file.
    :param window_size:
    :param step_size:
    :param eps: Small value to avoid NaN issues.
    :return: A Log Spectrogram.
    """
    audio = wavfile.read(wav)[1]

    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))

    if audio.size < TARGET_DURATION:
        audio = np.pad(audio, (TARGET_DURATION - audio.size, 0), mode='constant')
    elif audio.size > TARGET_DURATION:
        audio = audio[0:TARGET_DURATION]

    _, _, spec = signal.spectrogram(audio, fs=sample_rate,
                                    window='hann', nperseg=nperseg,
                                    noverlap=noverlap, detrend=False)

    log_spec = np.log(spec.T.astype(np.float32) + eps)
    log_spec = log_spec.reshape(log_spec.shape[0], log_spec.shape[1], 1)
    return log_spec


def log_spectrogram(audio, sample_rate=SAMPLE_RATE, window_size=20, step_size=10, eps=1e-6):
    """
    Creates a log spectrogram using librosa rather than signal.
    :param audio: A list of floating point values representing a wav file.
    :param sample_rate: The sample rate of the original file.
    :param window_size:
    :param step_size:
    :param eps: Small value to avoid NaN issues.
    :return: A log Spectrogram
    """
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))

    D = librosa.stft(audio, n_fft=nperseg, hop_length=noverlap, win_length=nperseg, window='hann')
    spectrogram = librosa.magphase(D)[0]
    log_spectrogram = np.log(spectrogram.astype(np.float32) + eps)
    return log_spectrogram.reshape(log_spectrogram.shape[0], log_spectrogram.shape[1], 1)


def log_mel_spectrogram(audio, sample_rate=SAMPLE_RATE, n_mels=40, n_fft=480, normalize=False):
    """
    Creates a log mel spectrogram representation of the audio.
    :param audio: A list of floating point values representing a wav file.
    :param sample_rate: The sample rate of the original file.
    :param n_mels:
    :param n_fft:
    :param normalize:
    :return: A log mel spectrogram
    """
    mel_spec = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=n_mels, hop_length=160, n_fft=n_fft, fmin=20, fmax=4000)
    mel_spec[mel_spec > 0] = np.log(mel_spec[mel_spec > 0])
    if normalize:
        mel_spec -= (np.mean(mel_spec, axis=0) + 1e-8)

    return mel_spec.reshape(mel_spec.shape[0], mel_spec.shape[1], 1)


def load_audio(audio):
    """
    Takes raw audio and performs normalisation.
    :param audio: A list of floating point values representing a wav file.
    :return: audio: Raw audio array.
    """
    # Normalize the audio.
    audio = (audio - np.mean(audio)) / np.std(audio)
    return audio.reshape(-1, 1)


def mfcc(audio, sample_rate=SAMPLE_RATE, n_mels=40, n_fft=400, normalize=True):
    """
    Creates an MFCC (Mel-frequency cepstral coefficients) representation of an audio file.
    :param audio:
    :param sample_rate:
    :param n_mels:
    :param n_fft:
    :param normalize:
    :return: A MFCC representation of the audio.
    """
    mfcc = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=n_mels, hop_length=160, n_fft=n_fft, fmin=20, fmax=4000)
    mfcc[mfcc > 0] = np.log(mfcc[mfcc > 0])
    mfcc = [np.matmul(dct_filters, x) for x in np.split(mfcc, mfcc.shape[1], axis=1)]
    mfcc = np.array(mfcc, order="F").squeeze(2).astype(np.float32)

    if normalize:
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return mfcc.reshape(mfcc.shape[0], mfcc.shape[1], 1)


def log_mel_filterbanks(audio, sample_rate=SAMPLE_RATE, n_mels=40, n_fft=480, normalize=True):
    """
    Creates a log mel filter bank representation of an audio file, log mel filter banks is an MFCC
    without the Discrete fourier transform being applied.
    :param audio:
    :param sample_rate:
    :param n_mels:
    :param n_fft:
    :param normalize:
    :return: A log mel filter bank representation of the audio.
    """
    filter_bank = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=n_mels, hop_length=160, n_fft=n_fft,
                                                 fmin=20, fmax=4000)
    filter_bank[filter_bank > 0] = np.log(filter_bank[filter_bank > 0])

    if normalize:
        filter_bank -= (np.mean(filter_bank, axis=0) + 1e-8)

    return filter_bank.reshape(filter_bank.shape[0], filter_bank.shape[1], 1)
