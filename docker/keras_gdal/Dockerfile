FROM tensorflow/tensorflow:nightly-gpu-py3

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable

RUN apt-get update \
    && apt-get install -y libgdal-dev \
       git \
       gdal-bin \
       python3-gdal \
    && apt-get autoremove \
    && rm -rf /var/lib/apt /var/cache/apt 

RUN pip install keras \
    scikit-image \
    tensorboard \
    rasterio \
    mapboxcli \ 
    && rm -rf /.cache/pip

RUN mkdir -p $(jupyter --data-dir)/nbextensions 
RUN cd $(jupyter --data-dir)/nbextensions && \
    git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding && \
    jupyter nbextension enable vim_binding/vim_binding
