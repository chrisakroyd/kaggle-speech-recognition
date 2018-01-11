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

# Models
# Log spectrum based models
from src.general_CNN import CNNModel
# Raw Audio based models
from src.raw_audio_models.CNN import ConvAudioModel
# Mel cepstrum coefficient based models.
from src.mel_models.VGG import VGG
from src.mel_models.DenseNet import DenseNetModel
from src.mel_models.ResNet import ResNet

from audio_data_generator import AudioDataGenerator

(x_train, y_train), (x_val, y_val), label_binarizer = load_data(path='./input/train/audio/',
                                                                val_path='./input/train/validation_list.txt')
test_set = get_test_data(path='./input/test/audio')

TRAIN = True
WRITE_RESULTS = True

# MODEL_TYPE = 'log_spectogram'
# MODEL_TYPE = 'raw_audio'
# MODEL_TYPE = 'mfcc'
MODEL_TYPE = 'log_mel_spectrogram'
# MODEL_TYPE = 'log_mel_filterbanks'

# model_instance = CNNModel()
# model_instance = ConvAudioModel()
# model_instance = DenseNetModel()
model_instance = VGG()
# model_instance = ResNet()


audio_preprocessor = AudioDataGenerator(generator_method=MODEL_TYPE)

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

model = load_model(model_instance.checkpoint_path)

if WRITE_RESULTS:
    write_results(model, label_binarizer, audio_preprocessor.flow_test, test_set)
