import numpy as np
import librosa


def raw_audio_batch_generator(x, y, batch_size=16):
    while True:
        # choose batch_size random images / labels from the data
        idx = np.random.randint(0, x.shape[0], batch_size)
        im = x[idx]
        label = y[idx]

        raw_audio = librosa.load(im)
        raw_audio = raw_audio.reshape(-1, 1)

        yield np.concatenate([raw_audio]), label
