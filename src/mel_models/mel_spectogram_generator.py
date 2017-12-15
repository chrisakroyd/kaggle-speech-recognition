import numpy as np
import librosa
from python_speech_features import mfcc

from src.load_data import load_data

(x_train, y_train), (x_val, y_val), label_binarizer = load_data(path='./input/train/audio/')


SAMPLE_RATE = 16000
TARGET_DURATION = 16000

mels = 40
dct_filters = 40
n_fft = 480

filters = librosa.filters.dct(40, 40)


def get_data_shape(wav_path):
    spec = load_mel_spec(wav_path)
    return spec.shape


def load_mel_spec(wav):
    audio = librosa.load(wav, sr=SAMPLE_RATE, mono=True)[0]
    duration = len(audio)

    if duration < TARGET_DURATION:
        audio = np.concatenate((audio, np.zeros(shape=(TARGET_DURATION - duration, 1))))
    elif duration > TARGET_DURATION:
        audio = audio[0:TARGET_DURATION]

    audio = librosa.feature.melspectrogram(audio, sr=SAMPLE_RATE, n_mels=mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    audio[audio > 0] = np.log(audio[audio > 0])
    # audio = [np.matmul(dct_filters, x) for x in np.split(audio, audio.shape[1], axis=1)]
    audio = [dct_filters * x for x in np.split(audio, audio.shape[1], axis=1)]
    audio = np.array(audio, order="F").squeeze(2).astype(np.float32)

    return audio


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
        raw_audio = [load_mel_spec(x) for x in im]

        yield np.concatenate([raw_audio]), label

        if counter <= number_of_batches:
            counter = 0


def test_batch_generator(test_files, batch_size=32):
    counter = 0

    while True:
        start = counter * batch_size
        end = (counter + 1) * batch_size

        if end > len(test_files):
            end = start + (len(test_files) - start)

        idx = np.arange(start, end)

        im = test_files.path[idx]
        raw_audio = [load_mel_spec(x) for x in im]

        yield np.concatenate([raw_audio])

        counter += 1

print(get_data_shape(x_train.iloc[0]))
