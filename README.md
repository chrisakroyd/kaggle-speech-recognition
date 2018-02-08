# kaggle-speech-recognition

This repository represents my entry into the recent [Kaggle Speech Recognition Competition](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge), this was my first ever Kaggle competition and I went into it with no domain knowledge so achieving a Leaderboard score of 0.852 (top 24%) was a unexpected surprise. This was quite challenging due to two factors 1. I had not applied neural networks to anything but toy examples before. 2. I'd never needed to create my own data preprocessing pipeline before. The work paid off and I learned a lot, ultimately though due to time constraints I couldn't spend much time finalising my solution but never the less I'm happy with my result.

### Requirements
    - Python 3.6
    - TensorFlow 1.4.1
    - Keras 2.1.1
    - Librosa 0.5.1
    - Scipy 1.0.0
    - Sklearn 0.19.1
    - Pandas 0.21.0

### Running the Code

1. Download the competition data from [The Competition website](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data).
2. Extract both the training and test data into the /input/ folder.
3. run `Python main.py` from the root folder.

### Structure
##### Folder Layout
* input
    * test - Testing data
    * train - Training data
* logs - Tensorboard Logs
* notebooks - Jupyter notebooks
* src
    * models - Stores all the implemented models.
        * mel_models - Models developed to be used solely on MFCC's.
        * raw_audio_models - Models that can only be used on raw audio or other 1-d inputs.
    * preprocessing - Preprocessing and feature representation related code.
* main.py - Main entrypoint to running the code.
##### Code Flow

The entrypoint, main.py, has the below overall code flow:
Load data -> Create data generators -> Create model -> Attach Callbacks -> Train model for n epochs -> Run predictions over test set. -> Write predictions.

### Experimentation notebooks

To run the notebooks:

run `docker build kaggle-speech-recognition -t kaggle-speech` to generate the docker image

run `docker run -it --rm -v ${pwd}:/home/jovyan/work -p 8888:8888 kaggle-speech` to run docker.

### Data Pre-Processing
Data pre-processing was a key part of this challenge and raised my score by 2-3%, demonstrations of these techniques can be found within the analytics notebook but below is a short list of techniques.

* Time Shifting - Shifting the start/end positions of the audio.
* Silence Slice - Gets a random slice of background noise (silence) rather than a fixed slice..
* Add Background Noise - Randomly splice in low volume background noise into audio examples.
* Amplitude Scaling - Randomly Adjust the volume of an audio example.
* Time Stretching - Stretch/Shrink the audio, allows for a clear view at each audio 'character'
* Pitch Scaling - Adjust the pitch of a speakers voice.

### Feature Representations
While experimenting I used and developed several different audio feature representations, all found within the 
src/preprocessing/feature_representations.py file. These representations are:

* Normalized 1-d Audio.
* Log Spectrogram (Generated with scipy Signal).
* Log Spectrogram (Generated with librosa).
* Log Mel Spectrogram
* Log Mel Filter Banks
* Mel-frequency cepstral coefficients

### Models
Overall I trained 6 different models over 6 different feature representations which have made it into this repository, several attempts were made at creating a CRNN model utilising CTC loss but these attempts were incomplete and therefore omitted from this repository. While all models attained a local validation accuracy of between 88-95% generally the smaller models performed better on the leader board set. Below is a list of the models within this repository.

* general_cnn - Best model that was used to set a baseline over all feature representations.
* mel_models/CNN - Small CNN similar to the TensorFlow speech recognition tutorial.
* mel_models/DenseNet - DenseNet based approach.
* mel_models/ResNet - ResNet based approach.
* mel_models/VGG - VGG based approach.
* raw_audio_models/CNN - Small CNN similar to the tensorflow speech recognition tutorial performed on 1d audio data.