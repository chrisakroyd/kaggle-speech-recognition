import numpy as np
import librosa


def batch_generator(input_x, labels, batch_size=32, shuffle=True):
    samples_per_epoch = input_x.shape[0]
    number_of_batches = samples_per_epoch / batch_size
    counter = 0

    while True:
        if shuffle:
            idx = np.random.randint(0, input_x.shape[0], batch_size)
        else:
            idx = np.arange(counter * batch_size, (counter + 1) * batch_size)

        im = input_x[idx]
        label = labels[idx]
        raw_audio = [librosa.load(x).reshape(-1, 1) for x in im]

        yield np.concatenate([raw_audio]), label

        if counter <= number_of_batches:
            counter = 0
