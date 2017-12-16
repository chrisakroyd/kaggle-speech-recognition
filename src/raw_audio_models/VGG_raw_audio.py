import keras.backend as K
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dense, Activation, Lambda
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2

"""
    VGG based approach to classifying raw-audio
    based on this paper: https://arxiv.org/pdf/1610.00087.pdf
"""

vgg_config = {
    'raw_audio_11': [(64, 64), (128, 128), (256, 256, 256), (512, 512)],
    'raw_audio_18': [(64, 64, 64, 64), (128, 128, 128, 128), (256, 256, 256, 256), (512, 512, 512, 512)]
}

# HPARAMs
BATCH_SIZE = 64
EPOCHS = 100
LEARN_RATE = 0.001
REG_STRENGTH = 0.0001


class VGGRawAudio:
    def __init__(self, num_clases=12):
        config_key = 'raw_audio_18'
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_clases
        self.config = vgg_config[config_key]
        self.checkpoint_path = config_key + '.hdf5'

    def create_model(self, shape):
        model = Sequential()

        model.add(Conv1D(64,
                         input_shape=shape,
                         kernel_size=80,
                         strides=4,
                         padding='same',
                         kernel_initializer='glorot_uniform',
                         kernel_regularizer=l2(REG_STRENGTH)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4, strides=None))

        model.add(BatchNormalization())

        for block in self.config:
            for filters in block:
                model.add(Conv1D(filters,
                                 kernel_size=3,
                                 kernel_regularizer=l2(REG_STRENGTH)
                                 ))
                model.add(BatchNormalization())
                model.add(Activation('relu'))

            model.add(MaxPooling1D(pool_size=4, strides=None))

        model.add((Lambda(lambda x: K.mean(x, axis=1))))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.LEARN_RATE), metrics=['accuracy'])

        return model
