FROM smizy/keras:2.1-cpu-alpine

RUN apk add --no-cache \
    --repository http://dl-cdn.alpinelinux.org/alpine/edge/testing \
    autoconf \
    automake \
    bash \
    g++ \
    gcc \
    gdal-dev \
    geos \
    git \
    make \
    musl-dev \
    nasm \
    nodejs \
    proj4-dev \
    python3-dev \
    py3-pillow

RUN pip3 install GDAL

# link geo libraries so Django can find them
RUN ln -s /usr/lib/libgeos_c.so.1 /usr/local/lib/libgeos_c.so
RUN ln -s /usr/lib/libgdal.so.20.1.0 /usr/local/lib/libgdal.so

# jupyter vim bindings, can comment out 
RUN mkdir -p $(jupyter --data-dir)/nbextensions 
RUN cd $(jupyter --data-dir)/nbextensions && \
    git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding && \
    jupyter nbextension enable vim_binding/vim_binding

