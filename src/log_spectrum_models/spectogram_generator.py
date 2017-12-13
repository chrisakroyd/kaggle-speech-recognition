from scipy import signal
from scipy.io import wavfile
import numpy as np

SAMPLE_RATE = 16000
TARGET_DURATION = 16000


def get_data_shape(wav_path):
    spec = log_spectograms([wav_path])
    print(spec[0].shape)
    return spec[0].shape


def log_spectograms(paths):
    # read the wav files
    specgram = [log_spectogram(wavfile.read(x)[1]) for x in paths]
    return specgram


def log_spectogram(audio, sample_rate=SAMPLE_RATE, window_size=20, step_size=10, eps=1e-10):
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


def batch_generator(input_x, labels, batch_size=32, shuffle=True):
    counter = 0

    while True:
        if shuffle:
            idx = np.random.randint(0, input_x.shape[0], batch_size)
        else:
            start = counter * batch_size
            end = (counter + 1) * batch_size

            if end > len(input_x):
                end = start + (len(input_x) - start)
            idx = np.arange(start, end)

        im = input_x[idx]
        label = labels[idx]
        specgram = log_spectograms(im)

        yield np.concatenate([specgram]), label

        counter += 1


def test_batch_generator(test_files, batch_size=32):
    counter = 0

    while True:
        start = counter * batch_size
        end = (counter + 1) * batch_size

        if end > len(test_files):
            end = start + (len(test_files) - start)

        idx = np.arange(start, end)

        im = test_files.path[idx]
        specgram = log_spectograms(im)

        yield np.concatenate([specgram])

        counter += 1

