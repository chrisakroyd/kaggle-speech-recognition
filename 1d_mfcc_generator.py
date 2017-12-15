import numpy as np
import librosa

from src.load_data import load_data

(x_train, y_train), (x_val, y_val), label_binarizer = load_data(path='./input/train/audio/')


SAMPLE_RATE = 16000
TARGET_DURATION = 16000


def get_data_shape(wav_path):
    spec = load_mfcc(wav_path)
    return spec.shape


def load_mfcc(wav):
    audio = librosa.load(wav, sr=SAMPLE_RATE, mono=True)[0]

    if np.std(audio) == 0:
        print(wav)

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

    return librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE)


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
        raw_audio = [load_mfcc(x) for x in im]

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
        raw_audio = [load_mfcc(x) for x in im]

        yield np.concatenate([raw_audio])

        counter += 1

print(get_data_shape(x_train.iloc[0]))
