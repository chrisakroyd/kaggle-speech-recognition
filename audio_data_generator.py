from scipy import signal
from scipy.io import wavfile
import librosa
import numpy as np
from src.load_data import get_background_noise

SAMPLE_RATE = 16000
TARGET_DURATION = 16000

background_noise_path = './input/train/audio/_background_noise_'
background_noise_mixing_probability = 0.8


def log_spectogram_signal(wav, sample_rate=SAMPLE_RATE, window_size=20, step_size=10, eps=1e-10):
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


def log_spectogram(audio, sample_rate=SAMPLE_RATE, window_size=20, step_size=10, eps=1e-6):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))

    D = librosa.stft(audio, n_fft=nperseg, hop_length=noverlap, win_length=nperseg, window='hann')
    spectrogram = librosa.magphase(D)[0]
    log_spectrogram = np.log(spectrogram.astype(np.float32) + eps)
    return log_spectrogram.reshape(log_spectrogram.shape[0], log_spectrogram.shape[1], 1)


def load_audio(audio):
    # Normalize the audio.
    audio = (audio - np.mean(audio)) / np.std(audio)
    return audio.reshape(-1, 1)


def load_mfcc(audio, sample_rate=SAMPLE_RATE, window_size_ms=30, window_stride_ms=10, n_mels=128):
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, hop_length=window_stride_samples,
                                       n_fft=window_size_samples)
    mfcc = librosa.feature.mfcc(S=S, n_mfcc=40)
    return mfcc.reshape(mfcc.shape[0], mfcc.shape[1], 1)

def log_mel_filterbanks(audio, sample_rate=SAMPLE_RATE):
    return

# AUDIO PREPROCESSING FUNCTIONS
# Shift the start/end of audio by -n to n milliseconds
def shift_audio(audio, ms_shift=100):
    ms = 16
    time_shift_dist = int(np.random.uniform(-(ms_shift * ms), (ms_shift * ms)))
    audio = np.roll(audio, time_shift_dist)
    return audio


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


class AudioDataGenerator(object):
    def __init__(self, generator_method='log_spectogram'):
        self.generator_method = generator_method
        self.background_noises = get_background_noise(background_noise_path)
        self.background_noises = [librosa.load(bg_wav, sr=SAMPLE_RATE)[0] for bg_wav in self.background_noises]
        if generator_method == 'log_spectogram':
            self.spec_func = log_spectogram
        elif generator_method == 'raw_audio':
            self.spec_func = load_audio
        elif generator_method == 'mel_cepstrum':
            self.spec_func = load_mfcc
        else:
            print('INVALID DATA GENERATOR SPECIFIED')

    def get_data_shape(self, wav_path):
        spec = self.spec_func(self.preprocess(wav_path, train=False))
        print(spec.shape)
        return spec.shape

    def preprocess(self, wav, train=True):
        audio = librosa.load(wav, sr=SAMPLE_RATE, mono=True)[0]

        # Audio padding/chopping to standardise the length of samples.
        audio = trim_pad_audio(audio)

        # Perform some pre-processing steps.
        if train:
            # Time shift start/end of audio of between -100ms and 100ms.
            audio = shift_audio(audio)
            # Time Stretch the audio
            # audio = time_strech(audio)
            # Some pre-processing steps can adjust length of audio, therefore trim again.
            # audio = trim_pad_audio(audio)

            # Pitch shift the audio
            # audio = pitch_scaling(audio)
            # Adjust the volume of the recording (Amplitude scaling)
            # audio = amplitude_scaling(audio)
            # Mix in background noise
            if np.random.uniform(0.0, 1.0) < background_noise_mixing_probability:
                background_noise_file = self.background_noises[np.random.randint(len(self.background_noises))]
                audio = add_background_noises(audio, background_noise_file)

        # Small check to make sure I didn't mess up.
        assert len(audio) == TARGET_DURATION

        return audio

    def flow(self, input_x, labels, batch_size=32, train=True):
        while True:
            idx = np.random.randint(0, input_x.shape[0], batch_size)
            im = input_x[idx]
            label = labels[idx]
            specgram = [self.spec_func(self.preprocess(x, train=train)) for x in im]

            yield np.concatenate([specgram]), label

    def flow_in_mem(self, input_x, labels, batch_size=32, train=True):
        input_preprocessed = np.array([self.spec_func(self.preprocess(x, train=train)) for x in input_x])

        while True:
            idx = np.random.randint(0, input_x.shape[0], batch_size)
            label = labels[idx]

            yield np.concatenate([input_preprocessed[idx]]), label

    def flow_test(self, test_files, batch_size=128):
        counter = 0

        while True:
            start = counter * batch_size
            end = (counter + 1) * batch_size

            if end > len(test_files):
                end = start + (len(test_files) - start)

            idx = np.arange(start, end)

            im = test_files.path[idx]
            raw_audio = [self.spec_func(self.preprocess(x, train=False)) for x in im]

            yield np.concatenate([raw_audio])

            counter += 1
