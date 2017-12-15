from keras.layers import BatchNormalization, Dense, Conv1D, GRU, TimeDistributed, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 100
EPOCHS = 100
LEARN_RATE = 0.001


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
        model = Sequential()

        model.add(Conv1D(self.gru_thickness, 11, border_mode='valid', strides=2, activation='relu', input_shape=shape))
        model.add(BatchNormalization())

        for i in range(self.gru_layers):
            model.add(Bidirectional(GRU(self.gru_thickness, activation='relu', return_sequences=True), merge_mode='sum'))
            model.add(BatchNormalization())

        model.add(TimeDistributed(Dense(self.num_classes, activation='softmax')))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        return model