from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

# HPARAMs
BATCH_SIZE = 100
EPOCHS = 50
LEARN_RATE = 0.001
NUM_CLASSES = 12


class ConvMelModel:
    def __init__(self, num_clases=12):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_clases
        self.checkpoint_path = './conv_mel_tutorial.hdf5'

    def create_model(self, shape):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(8, 20), padding='same', input_shape=shape, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(2))
        model.add(Conv2D(64, kernel_size=(4, 10), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=self.LEARN_RATE, momentum=0.9, decay=0.00001), metrics=['accuracy'])

        return model
