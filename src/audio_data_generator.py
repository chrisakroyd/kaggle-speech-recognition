import librosa
import numpy as np
from src.load_data import get_background_noise
from src.preprocessing.feature_representations import load_audio, log_spectrogram, log_mel_spectrogram, \
    log_mel_filterbanks, mfcc
from src.preprocessing.preprocessing import trim_pad_audio, amplitude_scaling, shift_audio, add_background_noises, \
    get_silence_slice

# constants to keep track of the sample rate/how long audio files should be.
SAMPLE_RATE = 16000
TARGET_DURATION = 16000
SILENCE_LABEL = 'silence'
# Where we keep the background noise and how often we should mix it in.
background_noise_path = './input/train/audio/_background_noise_'
background_noise_mixing_probability = 0.8


class AudioDataGenerator(object):
    def __init__(self, generator_method='log_mel_spectrogram'):
        """
        The Audio Generator class provides several functions that create generators for use with
        Keras. Different feature representations can be used e.g. log spectrograms, MFCC, raw audio
        depending upon the value of the generator_method parameter.
        :param generator_method: A string that corresponds to a feature function.
        """
        self.generator_method = generator_method
        self.background_noises = get_background_noise(background_noise_path)
        self.background_noises = [librosa.load(bg_wav, sr=SAMPLE_RATE)[0] for bg_wav in self.background_noises]
        # Work through the possible options for the generator method and set the feature function based on
        # what the given generator_method string is.
        if generator_method == 'log_spectrogram':
            self.feature_func = log_spectrogram
        elif generator_method == 'log_mel_spectrogram':
            self.feature_func = log_mel_spectrogram
        elif generator_method == 'raw_audio':
            self.feature_func = load_audio
        elif generator_method == 'mfcc':
            self.feature_func = mfcc
        elif generator_method == 'log_mel_filterbanks':
            self.feature_func = log_mel_filterbanks
        else:
            print('INVALID DATA GENERATOR SPECIFIED')

    def get_data_shape(self, wav_path):
        """
        Due to there being many possible methods to generate the features returned by the generator,
        this function returns the expected output shape dynamically for the input shape to a Keras network.
        :param wav_path: A filepath to a .wav file.
        :return: shape: A tuple representing the shape of the data returned by the generator functions.
        """
        spec = self.feature_func(self.preprocess(wav_path, train=False))
        return spec.shape

    def preprocess(self, wav, label=None, train=True):
        """
        A function that loads a .wav file and optionally performs various pre-processing functions upon it.
        e.g. Amplitude Scaling, Time Shifting, Time Stretching.
        :param wav: A file path to a .wav file.
        :param label: The label for this wave file.
        :param train: A True/False value that indicates if we are in Train mode. If True, runs pre-processing.
        :return: audio: A numpy array representing a loaded wav file.
        """
        audio = librosa.load(wav, sr=SAMPLE_RATE, mono=True)[0]

        # Perform some pre-processing steps.
        if train:
            if label != SILENCE_LABEL:
                # Audio padding/chopping to standardise the length of samples.
                audio = trim_pad_audio(audio)

                # Adjust the volume of the recording (Amplitude scaling)
                audio = amplitude_scaling(audio)
                # Time shift start/end of audio of between -100ms and 100ms.
                audio = shift_audio(audio)
                # Time Stretch the audio
                # audio = time_strech(audio)
                # Some pre-processing steps can adjust length of audio, therefore trim again.
                # audio = trim_pad_audio(audio)

                # Pitch shift the audio
                # audio = pitch_scaling(audio)
                # Mix in background noise
                if np.random.uniform(0.0, 1.0) < background_noise_mixing_probability:
                    background_noise_file = self.background_noises[np.random.randint(len(self.background_noises))]
                    audio = add_background_noises(audio, background_noise_file)
            else:
                audio = get_silence_slice(audio)

        audio = trim_pad_audio(audio)

        # Small check to make sure I didn't mess up.
        assert len(audio) == TARGET_DURATION

        return audio

    def flow(self, input_x, labels, batch_size=32, train=True):
        """
        Creates a generator that returns samples of batch_size with the designated feature representation.
        :param input_x: Pandas dataframe for the
        :param labels: Pandas dataframe for the labels.
        :param batch_size: size of the batches.
        :param train: Whether we are in train mode and need to perform data augmentation steps.
        :return: An array of batch size.
        """
        while True:
            idx = np.random.randint(0, input_x.shape[0], batch_size)
            im = input_x[idx]
            label = labels[idx]
            specgram = [self.feature_func(self.preprocess(im[i], label[i], train=train)) for i in range(len(im))]

            yield np.concatenate([specgram]), label

    def flow_in_mem(self, input_x, labels, batch_size=32, train=True):
        """
        Same as the flow generator except all data is loaded and persisted in memory to speed up train
        times. Requires a relatively large amount of RAM but cuts train time in half so it is preferable.
        :param input_x: The input data.
        :param labels: The input labels.
        :param batch_size: size of the batch we return
        :param train: True/False whether to run the pre processing steps designed to combat overfitting.
        :return: A generator that yields a numpy array of batch_size.
        """
        input_preprocessed = np.array([self.feature_func(
            self.preprocess(input_x[i], labels[i], train=train)
        ) for i in range(len(input_x))])

        while True:
            idx = np.random.randint(0, input_x.shape[0], batch_size)
            label = labels[idx]

            yield np.concatenate([input_preprocessed[idx]]), label

    def flow_test(self, test_files, batch_size=128):
        """
        Generator for the testing mode.
        :param test_files: pandas dataframe of the test files
        :param batch_size: size of batches
        :return: batches of batch_size preprocessed test examples.
        """
        counter = 0

        while True:
            start = counter * batch_size
            end = (counter + 1) * batch_size

            if end > len(test_files):
                end = start + (len(test_files) - start)

            idx = np.arange(start, end)

            im = test_files.path[idx]
            raw_audio = [self.feature_func(self.preprocess(x, train=False)) for x in im]

            yield np.concatenate([raw_audio])

            counter += 1
