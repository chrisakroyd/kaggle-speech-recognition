import librosa
import numpy as np
from src.load_data import get_background_noise
from src.preprocessing.feature_representations import load_audio, log_spectrogram, log_mel_spectrogram, \
    log_mel_filterbanks, mfcc
from src.preprocessing.preprocessing import trim_pad_audio, amplitude_scaling, shift_audio, add_background_noises, \
    get_silence_slice

SAMPLE_RATE = 16000
TARGET_DURATION = 16000
SILENCE_LABEL = 'silence'

background_noise_path = './input/train/audio/_background_noise_'
background_noise_mixing_probability = 0.8


class AudioDataGenerator(object):
    def __init__(self, generator_method='log_mel_spectrogram'):
        self.generator_method = generator_method
        self.background_noises = get_background_noise(background_noise_path)
        self.background_noises = [librosa.load(bg_wav, sr=SAMPLE_RATE)[0] for bg_wav in self.background_noises]
        if generator_method == 'log_spectrogram':
            self.spec_func = log_spectrogram
        elif generator_method == 'log_mel_spectrogram':
            self.spec_func = log_mel_spectrogram
        elif generator_method == 'raw_audio':
            self.spec_func = load_audio
        elif generator_method == 'mfcc':
            self.spec_func = mfcc
        elif generator_method == 'log_mel_filterbanks':
            self.spec_func = log_mel_filterbanks
        else:
            print('INVALID DATA GENERATOR SPECIFIED')

    def get_data_shape(self, wav_path):
        spec = self.spec_func(self.preprocess(wav_path, train=False))
        print(spec.shape)
        return spec.shape

    def preprocess(self, wav, label=None, train=True):
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
        while True:
            idx = np.random.randint(0, input_x.shape[0], batch_size)
            im = input_x[idx]
            label = labels[idx]
            specgram = [self.spec_func(self.preprocess(im[i], label[i], train=train)) for i in range(len(im))]

            yield np.concatenate([specgram]), label

    def flow_in_mem(self, input_x, labels, batch_size=32, train=True):
        input_preprocessed = np.array([self.spec_func(
            self.preprocess(input_x[i], labels[i], train=train)
        ) for i in range(len(input_x))])

        while True:
            idx = np.random.randint(0, input_x.shape[0], batch_size)
            label = labels[idx]

            yield np.concatenate([input_preprocessed[idx]]), label

    def flow_test(self, test_files, batch_size=128):
        counter = 0

        while True:
            start = counter * batch_size
            end = (counter + 1) * batch_size

            if end > len(test_files):
                end = start + (len(test_files) - start)

            idx = np.arange(start, end)

            im = test_files.path[idx]
            raw_audio = [self.spec_func(self.preprocess(x, train=False)) for x in im]

            yield np.concatenate([raw_audio])

            counter += 1
