from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 16
EPOCHS = 20
LEARN_RATE = 0.001
NUM_CLASSES = 12


class ConvAudioModel:
    def __init__(self, num_clases=12):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_clases
        self.checkpoint_path = './conv5_dense3.hdf5'

    def create_model(self, shape):
        model = Sequential()
        model.add(Conv1D(8, kernel_size=3, padding='same', input_shape=shape))
        for i in range(1, 6):
            model.add(Conv1D(8 * (2 ** i), 3, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling1D(2, padding='same'))

        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        return model



