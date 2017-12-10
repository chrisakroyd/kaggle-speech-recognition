import numpy as np
import pandas as pd
from load_data import prepare_data, get_data
from spectogram_generator import log_spectogram


def write_results(model, label_binarizer):
    test = prepare_data(get_data('../input/test/'))

    predictions = []
    paths = test.path.tolist()

    for path in paths:
        specgram = log_spectogram([path])
        pred = model.predict(np.array(specgram))
        predictions.extend(pred)

    labels = [label_binarizer.inverse_transform(p.reshape(1, -1), threshold=0.5)[0] for p in predictions]
    test['labels'] = labels
    test.path = test.path.apply(lambda x: str(x).split('/')[-1])
    submission = pd.DataFrame({'fname': test.path.tolist(), 'label': labels})
    submission.to_csv('submission.csv', index=False)
