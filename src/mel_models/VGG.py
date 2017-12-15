from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 100
EPOCHS = 100
LEARN_RATE = 0.001

vgg_config = {
    'VGG11': [(64,), (128,), (256, 256), (512, 512), (512, 512)],
    'VGG13': [(64, 64), (128, 128), (256, 256), (512, 512), (512, 512)]
}


class VGG:
    def __init__(self, num_clases=12):
        config_key = 'VGG11'
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.config = vgg_config[config_key]
        self.num_classes = num_clases
        self.checkpoint_path = config_key + '_mel.hdf5'

    def create_model(self, shape):
        model = Sequential()

        model.add(BatchNormalization(input_shape=shape))

        for block in self.config:
            for filters in block:
                model.add(Conv2D(filters, (3, 3)))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
            model.add(MaxPooling2D(2))

        model.add(Flatten())
        model.add(Dense(7680, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        return model
