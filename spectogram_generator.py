from scipy import signal
from scipy.io import wavfile
import numpy as np


def log_spectograms(paths, nsamples=16000):
    # read the wav files
    wavs = [wavfile.read(x)[1] for x in paths]

    # zero pad the shorter samples and cut off the long ones.
    data = []
    for wav in wavs:
        if wav.size < 16000:
            d = np.pad(wav, (nsamples - wav.size, 0), mode='constant')
        else:
            d = wav[0:nsamples]
        data.append(d)

    # get the specgram
    specgram = [signal.spectrogram(d, nperseg=256, noverlap=128)[2] for d in data]
    specgram = [s.reshape(s.shape[0], s.shape[1], -1) for s in specgram]

    return specgram


# def log_specgram(audio, sample_rate=16000, window_size=20, step_size=10, eps=1e-10):
#     nperseg = int(round(window_size * sample_rate / 1e3))
#     noverlap = int(round(step_size * sample_rate / 1e3))
#     _, _, spec = signal.spectrogram(audio, fs=sample_rate,
#                                     window='hann', nperseg=nperseg,
#                                     noverlap=noverlap, detrend=False)
#     return np.log(spec.T.astype(np.float32) + eps)
#

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

