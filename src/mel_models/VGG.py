from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, AveragePooling2D, GlobalMaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 100
EPOCHS = 20
LEARN_RATE = 0.001


class VGG:
    def __init__(self, num_clases=12):
        config_key = 'VGG5'
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_clases
        self.checkpoint_path = config_key + '_mel.hdf5'

    def create_model(self, shape):
        model = Sequential()

        model.add(BatchNormalization(input_shape=shape))

        # # Block 1
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv1'))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Classification block
        model.add(GlobalMaxPool2D())
        model.add(Dense(512, activation='relu', name='fc1'))
        model.add(Dense(256, activation='relu', name='fc2'))

        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        return model
