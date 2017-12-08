from scipy import signal
from scipy.io import wavfile
import numpy as np


def get_specgrams(paths, nsamples=16000):
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
    specgram = [s.reshape(129, 124, -1) for s in specgram]

    return specgram


def spectogram_batch_generator(x, y, batch_size=16):
    while True:
        # choose batch_size random images / labels from the data
        idx = np.random.randint(0, x.shape[0], batch_size)
        im = x[idx]
        label = y[idx]

        specgram = get_specgrams(im)

        yield np.concatenate([specgram]), label
