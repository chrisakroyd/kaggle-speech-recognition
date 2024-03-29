from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dense, Activation, Dropout, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 100
EPOCHS = 20
LEARN_RATE = 0.001
NUM_CLASSES = 12


class ConvAudioModel:
    """
        Simple Convolutional Neural Network working on raw audio, roughly equivalent to VGG-11
    """
    def __init__(self, num_clases=12):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_clases
        self.checkpoint_path = './ConvAudioModel.hdf5'

    def create_model(self, shape):
        model = Sequential()
        model.add(Conv1D(16, kernel_size=3, padding='same', input_shape=shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2, padding='same'))
        for i in range(1, 8):
            model.add(Conv1D(16 * (2 ** i), kernel_size=3, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling1D(2, padding='same'))

        # model.add(GlobalAveragePooling1D())
        model.add(GlobalMaxPooling1D())

        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.LEARN_RATE), metrics=['accuracy'])

        return model



