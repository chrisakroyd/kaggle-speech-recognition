from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Dropout, GlobalMaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 64
EPOCHS = 50
LEARN_RATE = 0.0001
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
        model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
        model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
        model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
        model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
        model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.2))

        model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
        model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
        model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.2))

        model.add(GlobalMaxPool2D())

        model.add(BatchNormalization())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        return model

