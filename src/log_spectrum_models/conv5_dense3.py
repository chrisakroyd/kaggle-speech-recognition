from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 12


class Conv5Dense3Model:
    def __init__(self, num_clases=12):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.num_classes = num_clases
        self.checkpoint_path = './conv5_dense3.hdf5'

    def create_model(self, shape):
        model = Sequential()

        model.add(BatchNormalization(input_shape=shape))
        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.2))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.2))
        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        return model

