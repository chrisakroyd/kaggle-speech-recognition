from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Dropout, GlobalMaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 64
EPOCHS = 25
LEARN_RATE = 0.005
NUM_CLASSES = 12


class CNNModel:
    def __init__(self, num_clases=12):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_clases
        self.checkpoint_path = './best_model.hdf5'

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

        model.add(GlobalMaxPool2D())

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        return model

