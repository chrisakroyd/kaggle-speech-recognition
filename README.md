# kaggle-speech-recognition

### Experimentation notebooks

run `docker build kaggle-speech-recognition -t kaggle-speech` to generate the docker image

run `docker run -it --rm -v ${pwd}:/home/jovyan/work -p 8888:8888 kaggle-speech` to run docker.

### load_data explanation

Loads a list of filepaths that belong to train/validation sets.
