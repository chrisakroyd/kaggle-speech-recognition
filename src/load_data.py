from pathlib import Path
import pandas as pd
import numpy as np

TRAIN_WORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
BACKGROUND_NOISE = '_background_noise_'
SILENCE_CLASS = 'silence'
UNKNOWN_CLASS = 'unknown'

RANDOM_SEED = 59185
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
    df.loc[df.word.isin(silence), 'word'] = SILENCE_CLASS
    df.loc[df.word.isin(unknown), 'word'] = UNKNOWN_CLASS

    return df


def load_data(path, val_path):
    data_set = prepare_data(get_data(path))
    x_train, x_val, y_train, y_val = [], [], [], []

    labels = data_set.word.unique().tolist()
    label_index = pd.get_dummies(pd.Series(labels))

    with open(val_path) as fin:
        validation_files = set(fin.read().splitlines())

    for i in range(len(data_set)):
        data_item = data_set.iloc[i]
        label = label_index[data_item.word].values.tolist()

        if data_item.hash_path in validation_files:
            x_val.append(data_item.path)
            y_val.append(label)
        else:
            x_train.append(data_item.path)
            y_train.append(label)

    return (np.array(x_train), np.array(y_train)), (np.array(x_val), np.array(y_val)), label_index
