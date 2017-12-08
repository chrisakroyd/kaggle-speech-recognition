import time

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from load_data import load_data

from data_prep_spectograms import spectogram_batch_generator

(x_train, y_train), (x_val, y_val) = load_data()


def get_model(shape):
    model = Sequential()

    model.add(BatchNormalization(input_shape=shape))
    model.add(Conv2D(16, (3, 3), activation='elu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(32, activation='elu'))
    model.add(Dropout(0.25))

    # 11 because background noise has been taken out
    model.add(Dense(11, activation='sigmoid'))

    return model


shape = (129, 124, 1)
model = get_model(shape)
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.summary()
# create training and test data.

tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=32)

train_gen = spectogram_batch_generator(x_train.values, y_train, batch_size=32)
valid_gen = spectogram_batch_generator(x_val.values, y_val, batch_size=32)

model.fit_generator(
    generator=train_gen,
    epochs=1,
    steps_per_epoch=x_train.shape[0] // 32,
    validation_data=valid_gen,
    validation_steps=x_val.shape[0] // 32,
    callbacks=[tensorboard])
