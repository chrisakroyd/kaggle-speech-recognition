from pathlib import Path
import pandas as pd
import numpy as np
from math import ceil
from sklearn.preprocessing import LabelBinarizer

# Train words are the words we want to keep.
TRAIN_WORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
# These words are used for data relabelling.
BACKGROUND_NOISE = '_background_noise_'
SILENCE_LABEL = 'silence'
UNKNOWN_LABEL = 'unknown'
# Same Seed = Same Distribution.
RANDOM_SEED = 59185
# Due to there being a large amount of unknown words, this variable allows us to control the balance between
# our train words and our unknown words.
CONTROL_BALANCE = True
# When CONTROL_BALANCE is true, these variables control what percentage of data is either silent or unknown.
silence_percentage = 10
unknown_percentage = 10


def get_data(path):
    """
    Loads the file paths for the audio files, the word being spoken and its stable hash code.
    :param path: A file path to the directory containing data.
    :return: df: A pandas data frame with the columns path, word and hash_path.
    """
    datadir = Path(path)
    files = [(str(f), f.parts[-2], '/'.join(str(i) for i in f.parts[-2:])) for f in datadir.glob('**/*.wav') if f]
    df = pd.DataFrame(files, columns=['path', 'word', 'hash_path'])

    return df


def get_test_data(path):
    """
    Loads the file paths for the test audio files as well as the fname, the fname is what we 'predict'.
    :param path: A file path to the directory containing test data.
    :return: df: A pandas data frame with the columns path and fname
    """
    datadir = Path(path)
    files = [(str(f), f.parts[-1]) for f in datadir.glob('**/*.wav') if f]
    df = pd.DataFrame(files, columns=['path', 'fname'])

    return df


def get_background_noise(path):
    """
    Loads the background noise data, data that consists of environmental sounds or white noise.
    :param path: A file path to the background noise directory.
    :return: file: A list of file paths to background noise files.
    """
    datadir = Path(path)
    files = [str(f) for f in datadir.glob('**/*.wav') if f]
    return files


def prepare_data(df):
    """
    Renames the word column based on whether it is one of our training words, a silent word(Background noise). Any
    words that do not fit within these two categories are termed 'unknown' words and are relabelled as such.
    :param df: A data frame with the column word
    :return:
    """
    words = df.word.unique().tolist()
    silence = [BACKGROUND_NOISE]
    unknown = [w for w in words if w not in silence + TRAIN_WORDS]

    # there are only 6 silence files. Mark them as unknown too.
    df.loc[df.word.isin(silence), 'word'] = SILENCE_LABEL
    df.loc[df.word.isin(unknown), 'word'] = UNKNOWN_LABEL

    return df


# @TODO revist and rework this function to be cleaner.
def load_data(path, val_path, control_balance=CONTROL_BALANCE):
    """
    Loads the file paths for all the files in the data set, optionally controlling the balance so that no one
    class dominates.
    The validation set consists of all Id's in the validation.txt file passed in through the val_path. This is so
    that there is no speaker overlap.
    :param path: File path to the train directory for data loading.
    :param val_path: The file path of the validation.txt file. This is used to stably assign the train/val set so
    no speaker appears in both.
    :param control_balance: Whether we should contrl the balance of classes within the dataset.
    :return: (x_train, y_train), (x_val, y_val), label_binarizer:
    """
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
