from pathlib import Path
import pandas as pd
import numpy as np
from math import ceil
from sklearn.preprocessing import LabelBinarizer

TRAIN_WORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
BACKGROUND_NOISE = '_background_noise_'
SILENCE_LABEL = 'silence'
UNKNOWN_LABEL = 'unknown'

RANDOM_SEED = 59185
CONTROL_BALANCE = True

silence_percentage = 10
unknown_percentage = 10


def get_data(path):
    datadir = Path(path)
    files = [(str(f), f.parts[-2], '/'.join(str(i) for i in f.parts[-2:])) for f in datadir.glob('**/*.wav') if f]
    df = pd.DataFrame(files, columns=['path', 'word', 'hash_path'])

    return df


def get_test_data(path):
    datadir = Path(path)
    files = [(str(f), f.parts[-1]) for f in datadir.glob('**/*.wav') if f]
    df = pd.DataFrame(files, columns=['path', 'fname'])

    return df


def get_background_noise(path):
    datadir = Path(path)
    files = [str(f) for f in datadir.glob('**/*.wav') if f]
    return files


def prepare_data(df):
    words = df.word.unique().tolist()
    silence = [BACKGROUND_NOISE]
    unknown = [w for w in words if w not in silence + TRAIN_WORDS]

    # there are only 6 silence files. Mark them as unknown too.
    df.loc[df.word.isin(silence), 'word'] = SILENCE_LABEL
    df.loc[df.word.isin(unknown), 'word'] = UNKNOWN_LABEL

    return df


def load_data(path, val_path, control_balance=CONTROL_BALANCE):
    data_set = prepare_data(get_data(path))
    x_train, x_val, y_train, y_val = [], [], [], []
    silence_x, silence_y, unknown_x, unknown_y = [], [], [], []
    unknown_val, unknown_val_y = [], []

    train_word_set = set(TRAIN_WORDS)

    label_binarizer = LabelBinarizer().fit(data_set.word)

    with open(val_path) as fin:
        validation_files = set(fin.read().splitlines())

    for i in range(len(data_set)):
        data_item = data_set.iloc[i]
        label = label_binarizer.transform([data_item.word])[0]

        if data_item.hash_path in validation_files:
            if data_item.word == UNKNOWN_LABEL and CONTROL_BALANCE:
                unknown_val.append(data_item.path)
                unknown_val_y.append(label)
            else:
                x_val.append(data_item.path)
                y_val.append(label)
        elif data_item.word in train_word_set:
            x_train.append(data_item.path)
            y_train.append(label)
        elif data_item.word == SILENCE_LABEL and CONTROL_BALANCE:
            silence_x.append(data_item.path)
            silence_y.append(label)
        elif data_item.word == UNKNOWN_LABEL and CONTROL_BALANCE:
            unknown_x.append(data_item.path)
            unknown_y.append(label)
        else:
            x_train.append(data_item.path)
            y_train.append(label)

    if control_balance:
        set_size = len(x_train)
        silence_size = int(ceil(set_size * silence_percentage / 100))
        unknown_size = int(ceil(set_size * unknown_percentage / 100))
        val_set_size = len(x_val)
        val_silence_size = int(ceil(val_set_size * silence_percentage / 100))
        val_unknown_size = int(ceil(val_set_size * unknown_percentage / 100))

        np.random.seed(RANDOM_SEED)
        for _ in range(silence_size):
            index = np.random.randint(0, len(silence_x))
            x_train.append(silence_x[index])
            y_train.append(silence_y[index])

        for _ in range(val_silence_size):
            index = np.random.randint(0, len(silence_x))
            x_val.append(silence_x[index])
            y_val.append(silence_y[index])

        unknown_indexes = np.random.randint(0, len(unknown_x), unknown_size)
        unknown_indexes_val = np.random.randint(0, len(unknown_val), val_unknown_size)

        rand_unknown_x = np.array(unknown_x)[unknown_indexes].tolist()
        rand_unknown_y = np.array(unknown_y)[unknown_indexes].tolist()

        rand_unknown_x_val = np.array(unknown_val)[unknown_indexes_val].tolist()
        rand_unknown_y_val = np.array(unknown_val_y)[unknown_indexes_val].tolist()

        x_train.extend(rand_unknown_x)
        y_train.extend(rand_unknown_y)
        x_val.extend(rand_unknown_x_val)
        y_val.extend(rand_unknown_y_val)

    return (np.array(x_train), np.array(y_train)), (np.array(x_val), np.array(y_val)), label_binarizer
