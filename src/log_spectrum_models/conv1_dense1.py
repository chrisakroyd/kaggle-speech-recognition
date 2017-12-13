from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

# HPARAMS
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 12


class Conv1Dense1Model:
    def __init__(self, num_clases=12):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.num_classes = num_clases
        self.checkpoint_path = './conv1_dense1.hdf5'

    def create_model(self, shape):
        model = Sequential()

        model.add(BatchNormalization(input_shape=shape))
        model.add(Conv2D(16, (3, 3), activation='elu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(32, activation='elu'))
        model.add(Dropout(0.25))

        # 11 because background noise has been taken out
        model.add(Dense(self.num_classes, activation='sigmoid'))

        model.summary()

        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        return model
