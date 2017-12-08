from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def get_data(path):
    datadir = Path(path)
    files = [(str(f), f.parts[-2]) for f in datadir.glob('**/*.wav') if f]
    df = pd.DataFrame(files, columns=['path', 'word'])

    return df


def prepare_data(df):
    train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    words = df.word.unique().tolist()
    silence = ['_background_noise_']
    unknown = [w for w in words if w not in silence + train_words]

    # there are only 6 silence files. Mark them as unknown too.
    df.loc[df.word.isin(silence), 'word'] = 'unknown'
    df.loc[df.word.isin(unknown), 'word'] = 'unknown'

    return df


def load_data():
    train = prepare_data(get_data('./input/train/audio/'))

    label_binarizer = LabelBinarizer()
    X = train.path
    y = label_binarizer.fit_transform(train.word)

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y)

    return (x_train, y_train), (x_val, y_val)

