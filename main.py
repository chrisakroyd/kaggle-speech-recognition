import time
from math import ceil

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model

# Utility code.
from src.load_data import load_data, get_test_data
from src.results import write_results

# Models
# Log spectrum based models
from src.log_spectrum_models.conv5_dense3 import Conv5Dense3Model
# Raw Audio based models
from src.raw_audio_models.VGG_raw_audio import VGGRawAudio
# Mel cepstrum coefficient based models.
from src.mel_models.CNN import ConvMelModel
from src.mel_models.VGG import VGG

from audio_data_generator import AudioDataGenerator

(x_train, y_train), (x_val, y_val), label_binarizer = load_data(path='./input/train/audio/')
test_set = get_test_data(path='./input/test/audio')

TRAIN = True
WRITE_RESULTS = True

MODEL_TYPE = 'log_spectogram'
# MODEL_TYPE = 'raw_audio'
# MODEL_TYPE = 'mel_cepstrum'

model_instance = Conv5Dense3Model()
# model_instance = VGGRawAudio()
# model_instance = ConvAudioModel()
# model_instance = Conv1Dense1Model()
# model_instance = ConvMelModel()
# model_instance = VGG()

audio_preprocessor = AudioDataGenerator(generator_method=MODEL_TYPE)

if TRAIN:
    model = model_instance.create_model(audio_preprocessor.get_data_shape(x_train.iloc[0]))

    tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=model_instance.BATCH_SIZE)
    checkpoint = ModelCheckpoint(model_instance.checkpoint_path, monitor='val_loss')
    early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=0.0001)

    train_gen = audio_preprocessor.flow(x_train.values, y_train, batch_size=model_instance.BATCH_SIZE)
    valid_gen = audio_preprocessor.flow(x_val.values, y_val, batch_size=model_instance.BATCH_SIZE)

    model.fit_generator(
        generator=train_gen,
        epochs=model_instance.EPOCHS,
        steps_per_epoch=ceil(x_train.shape[0] / model_instance.BATCH_SIZE),
        validation_data=valid_gen,
        validation_steps=ceil(x_val.shape[0] / model_instance.BATCH_SIZE),
        callbacks=[tensorboard, checkpoint, early_stop]
    )
else:
    model = load_model(model_instance.checkpoint_path)

if WRITE_RESULTS:
    write_results(model, label_binarizer, audio_preprocessor.flow_test, test_set)
