from scipy import signal
from scipy.io import wavfile
import librosa
import numpy as np

SAMPLE_RATE = 16000
TARGET_DURATION = 16000


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


def log_spectogram(wav, sample_rate=SAMPLE_RATE, window_size=20, step_size=10, eps=1e-10):
    audio = librosa.load(wav, sr=SAMPLE_RATE, mono=True)[0]

    duration = len(audio)
    if duration < TARGET_DURATION:
        audio = np.pad(audio, (TARGET_DURATION - audio.size, 0), mode='constant')
    elif duration > TARGET_DURATION:
        audio = audio[0:TARGET_DURATION]

    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))

    D = librosa.stft(audio, n_fft=nperseg, hop_length=nperseg - noverlap, win_length=nperseg, window='hann')
    spectrogram = librosa.magphase(D)[0]
    log_spectrogram = np.log(spectrogram.astype(np.float32) + eps)
    # Keep things the same orientation as the previous scipy signal based spectrogram
    log_spectrogram = np.swapaxes(log_spectrogram, 1, 0)
    return log_spectrogram.reshape(log_spectrogram.shape[0], log_spectrogram.shape[1], 1)


def load_audio(wav):
    audio = librosa.load(wav, sr=SAMPLE_RATE, mono=True)[0].reshape(-1, 1)

    # Normalize the audio.
    audio = (audio - np.mean(audio)) / np.std(audio)
    duration = len(audio)

    # Crude method for padding/trimming all audio to be one second long
    if duration < TARGET_DURATION:
        audio = np.concatenate((audio, np.zeros(shape=(TARGET_DURATION - duration, 1))))
    elif duration > TARGET_DURATION:
        audio = audio[0:TARGET_DURATION]

    # Small check to make sure I didn't mess up.
    assert len(audio) == TARGET_DURATION

    return audio


def load_mfcc(wav):
    audio = librosa.load(wav, sr=SAMPLE_RATE, mono=True)[0]
    # audio = (audio - np.mean(audio)) / np.std(audio)
    duration = len(audio)
    if duration < TARGET_DURATION:
        audio = np.pad(audio, (TARGET_DURATION - audio.size, 0), mode='constant')
    elif duration > TARGET_DURATION:
        audio = audio[0:TARGET_DURATION]

    window_size_ms = 30
    window_stride_ms = 10
    window_size_samples = int(SAMPLE_RATE * window_size_ms / 1000)
    window_stride_samples = int(SAMPLE_RATE * window_stride_ms / 1000)

    S = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128, hop_length=window_stride_samples,
                                       n_fft=window_size_samples)
    mfcc = librosa.feature.mfcc(S=S, n_mfcc=40)
    return mfcc.reshape(mfcc.shape[0], mfcc.shape[1], 1)


class AudioDataGenerator(object):
    def __init__(self, generator_method='log_spectogram'):
        self.generator_method = generator_method
        if generator_method == 'log_spectogram':
            self.spec_func = log_spectogram
        elif generator_method == 'raw_audio':
            self.spec_func = load_audio
        elif generator_method == 'mel_cepstrum':
            self.spec_func = load_mfcc
        else:
            print('INVALID DATA GENERATOR SPECIFIED')

    def get_data_shape(self, wav_path):
        spec = self.spec_func(wav_path)
        print(spec.shape)
        return spec.shape

    def flow(self, input_x, labels, batch_size=32):
        while True:
            idx = np.random.randint(0, input_x.shape[0], batch_size)
            im = input_x[idx]
            label = labels[idx]
            specgram = [self.spec_func(x) for x in im]

            yield np.concatenate([specgram]), label

    def flow_test(self, test_files, batch_size=128):
        counter = 0

        while True:
            start = counter * batch_size
            end = (counter + 1) * batch_size

            if end > len(test_files):
                end = start + (len(test_files) - start)

            idx = np.arange(start, end)

            im = test_files.path[idx]
            raw_audio = [self.spec_func(x) for x in im]

            yield np.concatenate([raw_audio])

            counter += 1
