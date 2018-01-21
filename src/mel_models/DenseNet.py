from keras.layers import Conv2D, BatchNormalization, Dense, Activation, AveragePooling2D, GlobalAveragePooling2D, concatenate
from keras.regularizers import l2
from keras.models import Input, Model
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 64
EPOCHS = 50
LEARN_RATE = 0.001
NUM_CLASSES = 12


def conv_block(ip, nb_filter, weight_decay=1E-4):
    x = Activation('relu')(ip)
    x = Conv2D(nb_filter, kernel_size=3, kernel_initializer="he_uniform", padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    return x


def transition_block(ip, nb_filter, weight_decay=1E-4):
    x = Conv2D(nb_filter, kernel_size=1, kernel_initializer="he_uniform", padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(ip)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)

    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, weight_decay=1E-4):
    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, weight_decay)
        feature_list.append(x)
        x = concatenate(feature_list)
        nb_filter += growth_rate

    return x, nb_filter


class DenseNetModel:
    def __init__(self, num_clases=12):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_clases
        self.checkpoint_path = './best_model.hdf5'

    def create_model(self, shape, depth=58, nb_dense_block=5, growth_rate=32, nb_filter=16, weight_decay=1E-4):
        model_input = Input(shape=shape)

        assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

        # layers in each dense block
        nb_layers = int((depth - 4) / 3)

        # Initial convolution
        x = Conv2D(nb_filter, kernel_size=3, kernel_initializer="he_uniform", padding="same", name="initial_conv2D",
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay))(model_input)

        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate,
                                       weight_decay=weight_decay)
            # add transition_block
            x = transition_block(x, nb_filter, weight_decay=weight_decay)

        # The last dense_block does not have a transition_block
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate,
                                   weight_decay=weight_decay)

        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

        model = Model(input=model_input, output=x, name="create_dense_net")

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        return model

