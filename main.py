import time
from math import ceil
import keras.backend as K

# Only use the amount of memory we require rather than the maximum
if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model

# Utility code.
from src.load_data import load_data, get_test_data
from src.results import write_results
# Model
from src.models.general_CNN import CNNModel
# Generator and pre-processing.
from src.audio_data_generator import AudioDataGenerator


TRAIN = True
WRITE_RESULTS = True
# File paths to data
TRAIN_PATH = './input/train/audio/'
TEST_PATH = './input/test/audio'
VAL_FILE_PATH = './input/train/validation_list.txt'
# What feature representation we use.
FEATURE_REP = 'log_mel_spectrogram'

(x_train, y_train), (x_val, y_val), label_binarizer = load_data(path=TRAIN_PATH, val_path=VAL_FILE_PATH)

model_instance = CNNModel()

audio_preprocessor = AudioDataGenerator(generator_method=FEATURE_REP)

if TRAIN:
    model = model_instance.create_model(audio_preprocessor.get_data_shape(x_train[0]))

    tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=model_instance.BATCH_SIZE)
    checkpoint = ModelCheckpoint(model_instance.checkpoint_path, monitor='val_loss')

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=7,
                               verbose=1,
                               min_delta=0.00001)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=2,
                                  verbose=1,
                                  epsilon=0.0001,
                                  mode='min', min_lr=0.0001)

    train_gen = audio_preprocessor.flow(x_train, y_train, batch_size=model_instance.BATCH_SIZE)
    valid_gen = audio_preprocessor.flow_in_mem(x_val, y_val, batch_size=model_instance.BATCH_SIZE, train=False)

    model.fit_generator(
        generator=train_gen,
        epochs=model_instance.EPOCHS,
        steps_per_epoch=ceil(x_train.shape[0] / model_instance.BATCH_SIZE),
        validation_data=valid_gen,
        validation_steps=ceil(x_val.shape[0] / model_instance.BATCH_SIZE),
        callbacks=[tensorboard, checkpoint, early_stop]
    )

if WRITE_RESULTS:
    test_set = get_test_data(path=TEST_PATH)
    model = load_model(model_instance.checkpoint_path)
    write_results(model, label_binarizer, audio_preprocessor.flow_test, test_set)
