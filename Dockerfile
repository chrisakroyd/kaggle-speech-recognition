FROM jupyter/scipy-notebook

LABEL maintainer="Chris Akroyd <chris@chrisakroyd.com>"

# Install Tensorflow
RUN conda install --quiet --yes \
    'tensorflow=1.3*' \
    'librosa=0.5*' \
    'keras=2.0*' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR