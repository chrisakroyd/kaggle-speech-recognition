from keras.backend import ctc_batch_cost
from keras.layers import BatchNormalization, Dense, Conv1D, GRU, TimeDistributed, Bidirectional, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 100
EPOCHS = 100
LEARN_RATE = 0.001


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)


class DeepSpeechRNN:
    def __init__(self, num_clases=12):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.checkpoint_path = 'deep_rnn_mel.hdf5'

        self.gru_thickness = 1024
        self.gru_layers = 3
        self.num_classes = num_clases

    def create_model(self, shape):
        input = Input(shape=shape, dtype='float32')
        conv1 = Conv1D(self.gru_thickness, 11, border_mode='valid', strides=2, activation='relu', input_shape=shape)(input)
        batch1 = BatchNormalization()(conv1)

        gru_layers = batch1

        for i in range(self.gru_layers):
            gru_layers = Bidirectional(
                GRU(self.gru_thickness, activation='relu', return_sequences=True),
                merge_mode='sum')(gru_layers)
            gru_layers = BatchNormalization()(gru_layers)

        # Relying on the CTC loss function to perform softmax for me ( as in the docs )
        # y_pred = TimeDistributed(Dense(self.num_classes, activation='linear'))(gru_layers)
        y_pred = TimeDistributed(Dense(self.num_classes, activation="softmax"))(gru_layers)

        labels = Input(name='the_labels', shape=[None, ], dtype='int32')
        input_length = Input(name='input_length', shape=[1], dtype='int32')
        label_length = Input(name='label_length', shape=[1], dtype='int32')

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                           labels,
                                                                           input_length,
                                                                           label_length])

        model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)

        model.summary()

        # model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(), metrics=['accuracy'])

        return model
