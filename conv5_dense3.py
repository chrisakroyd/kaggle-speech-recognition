import time

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

from load_data import load_data
from results import write_results

from spectogram_generator import batch_generator, log_spectograms

(x_train, y_train), (x_val, y_val), label_binarizer = load_data()

WRITE_RESULTS = True
# HPARAMs
BATCH_SIZE = 16
EPOCHS = 1
NUM_CLASSES = 12


def get_data_shape():
    wav_path = x_train[0]
    spec = log_spectograms([wav_path])
    print(spec[0].shape)
    return spec[0].shape


def get_model(shape):
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
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model


shape = get_data_shape()
model = get_model(shape)
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.summary()

# create training and test data.
tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=BATCH_SIZE)
checkpoint = ModelCheckpoint('./conv5_dense3.hdf5')

train_gen = batch_generator(x_train.values, y_train, batch_size=BATCH_SIZE)
valid_gen = batch_generator(x_val.values, y_val, batch_size=BATCH_SIZE, shuffle=False)

model.fit_generator(
    generator=train_gen,
    epochs=EPOCHS,
    steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
    validation_data=valid_gen,
    validation_steps=x_val.shape[0] // BATCH_SIZE,
    callbacks=[tensorboard, checkpoint])

if WRITE_RESULTS:
    write_results(model, label_binarizer)
