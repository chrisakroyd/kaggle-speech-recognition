import pandas as pd
from load_data import get_test_data
from spectogram_generator import test_batch_generator
from math import ceil

test_set = get_test_data('./input/test/audio')

BATCH_SIZE = 32


def write_results(model, label_binarizer):
    index = []
    results = []

    predictions = model.predict_generator(test_batch_generator(test_set, batch_size=BATCH_SIZE),
                                          steps=ceil(test_set.shape[0] / BATCH_SIZE))

    assert len(predictions) == len(test_set)

    for i in range(len(predictions)):
        prediction = label_binarizer.inverse_transform(predictions[i].reshape(1, -1))[0]
        index.append(test_set.iloc[i].fname)
        results.append(prediction)

    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = index
    df['label'] = results
    df.to_csv('sub.csv', index=False)

