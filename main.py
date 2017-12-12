import time

from keras.callbacks import TensorBoard, ModelCheckpoint
# Utility code.
from src.load_data import load_data
from src.results import write_results
# Data Generators
from src.log_spectrum_models.spectogram_generator import batch_generator as spectogram_batch_generator,\
    get_data_shape as get_spectogram_data_shape, test_batch_generator as spectogram_test_batch_generator
from src.raw_audio_models.raw_audio_generator import batch_generator as audio_batch_generator,\
    get_data_shape as get_audio_data_shape, test_batch_generator as audio_test_batch_generator
# Models
from src.log_spectrum_models.conv1_dense1 import Conv1Dense1Model
from src.log_spectrum_models.conv5_dense3 import Conv5Dense3Model
from src.mel_cepstrum_models.VGG import VGG

(x_train, y_train), (x_val, y_val), label_binarizer = load_data(path='./input/train/audio/')

WRITE_RESULTS = False
MODEL_TYPE = 'log_spectogram'

batch_generator = 0
test_batch_generator = 0
get_data_shape = 0

if MODEL_TYPE == 'log_spectogram':
    batch_generator = spectogram_batch_generator
    test_batch_generator = spectogram_test_batch_generator
    get_data_shape = get_spectogram_data_shape
elif MODEL_TYPE == 'raw_audio':
    batch_generator = audio_batch_generator
    test_batch_generator = audio_test_batch_generator
    get_data_shape = get_audio_data_shape
elif MODEL_TYPE == 'mel_cepstrum':
    batch_generator = 0
    test_batch_generator = 0
    get_data_shape = 0
else:
    print('INVALID DATA GENERATOR SPECIFIED')

# model_instance = Conv1Dense1Model()
# model_instance = Conv5Dense3Model()
model_instance = VGG()

model = model_instance.create_model(get_data_shape(x_train[0]))

tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=model_instance.BATCH_SIZE)
checkpoint = ModelCheckpoint('./test.hdf5', monitor='val_loss')

train_gen = batch_generator(x_train.values, y_train, batch_size=model_instance.BATCH_SIZE)
valid_gen = batch_generator(x_val.values, y_val, batch_size=model_instance.BATCH_SIZE, shuffle=False)

model.fit_generator(
    generator=train_gen,
    epochs=model_instance.EPOCHS,
    steps_per_epoch=x_train.shape[0] // model_instance.BATCH_SIZE,
    validation_data=valid_gen,
    validation_steps=x_val.shape[0] // model_instance.BATCH_SIZE,
    callbacks=[tensorboard, checkpoint])

if WRITE_RESULTS:
    write_results(model, label_binarizer, test_batch_generator)
