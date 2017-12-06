from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, Dense, GRU
from keras.models import Sequential
from keras.optimizers import Adam

LEARN_RATE = 0.03
MIN_LEARN_RATE = 0.00001
BATCH_SIZE = 32
EPOCHS = 500


def build_model():
    model = Sequential()

    model.add(Conv1D(32, kernal_size=(20, 5), strides=(8, 2)))

    model.add(GRU(32, activation='relu', recurrent_activation='relu'))
    model.add(GRU(32, activation='relu', recurrent_activation='relu'))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LEARN_RATE), metrics=['accuracy'])

    return model


model = build_model()


early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)


history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=[learning_rate_reduction, early_stop],
                    validation_data=(x_val, y_val))

