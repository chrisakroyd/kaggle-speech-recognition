import pandas as pd
from math import ceil


BATCH_SIZE = 32


def write_results(model, label, test_batch_generator, test_set):
    """
    Runs the predications and then writes all the results to a csv file.
    :param model: Keras model to run the predictions on.
    :param label: The label binarizer that was used for training - this is to go from predictions to labels.
    :param test_batch_generator: A Generator for test examples.
    :param test_set: The full test set.
    """
    index = []
    results = []

    print('Running ' + str(len(test_set)) + ' predictions...')

    predictions = model.predict_generator(test_batch_generator(test_set, batch_size=BATCH_SIZE),
                                          steps=ceil(test_set.shape[0] / BATCH_SIZE))

    assert len(predictions) == len(test_set)

    print('Writing ' + str(len(predictions)) + ' predictions...')

    for i in range(len(predictions)):
        prediction = label.inverse_transform(predictions[i].reshape(1, -1))[0]
        index.append(test_set.iloc[i].fname)
        results.append(prediction)

    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = index
    df['label'] = results
    df.to_csv('sub3.csv', index=False)

