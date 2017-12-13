import numpy as np
import librosa

SAMPLE_RATE = 16000
TARGET_DURATION = 16000


def get_data_shape(wav_path):
    spec = load_audio(wav_path)
    print(spec[0].shape)
    return spec[0].shape


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
    assert duration == TARGET_DURATION

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
        raw_audio = [load_audio(x) for x in im]

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
        raw_audio = [load_audio(x) for x in im]

        yield np.concatenate([raw_audio])

        counter += 1
