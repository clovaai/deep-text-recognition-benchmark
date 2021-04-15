FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
ENV FORCE_CUDA="1"

# ENV DEBIAN_FRONTEND noninteractive
# RUN apt-get update && \
#      apt-get -y install gcc mono-mcs && \
#      rm -rf /var/lib/apt/lists/*

# RUN  apt-get update -y && \
#      apt-get upgrade -y && \
#      apt-get dist-upgrade -y && \
#      apt-get -y autoremove && \
#      apt-get clean
# FROM python:3.8.8-slim
# FROM python:3.8.8



# FROM continuumio/miniconda3
# RUN python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

ENV DEBIAN_FRONTEND=noninteractive

ENV LANG C.UTF-8
ENV PYTHON_VERSION="3.8.5"
ENV OPENCV_VERSION="4.5.0"
ENV NUMPY_VERSION="1.17.4"
ENV PYTHON_PIP_VERSION="19.3.1"

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    nasm \
    yasm \
    openssl \
    libssl-dev \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk2.0-dev \
    libavformat-dev \
    libpq-dev \
    libffi-dev \
    lsb-release \
    libreadline-dev \
    libsqlite3-dev \
    apt-transport-https \
    && rm -rf /var/lib/apt/lists/*
    
RUN set -ex \
	&& buildDeps=' \
		dpkg-dev \
		tcl-dev \
		tk-dev \
	' \
	&& apt-get update && apt-get install -y $buildDeps --no-install-recommends \
	\
	&& wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz" \
	&& wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc" \
	&& export GNUPGHOME="$(mktemp -d)" \
	&& rm -rf "$GNUPGHOME" python.tar.xz.asc \
	&& mkdir -p /usr/src/python \
	&& tar -xJC /usr/src/python --strip-components=1 -f python.tar.xz \
	&& rm python.tar.xz \
	\
	&& cd /usr/src/python \
	&& gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)" \
	&& ./configure \
		--build="$gnuArch" \
		--enable-loadable-sqlite-extensions \
		--enable-shared \
		--with-system-expat \
		--with-system-ffi \
		--without-ensurepip \
	&& make -j "$(nproc)" \
	&& make install \
	&& ldconfig

# make some useful symlinks that are expected to exist
RUN cd /usr/local/bin \
	&& ln -s idle3 idle \
	&& ln -s pydoc3 pydoc \
	&& ln -s python3 python \
	&& ln -s python3-config python-config

RUN set -ex; \
	\
	wget -O get-pip.py 'https://bootstrap.pypa.io/get-pip.py'; \
	\
	python get-pip.py \
		--disable-pip-version-check \
		--no-cache-dir \
		"pip==$PYTHON_PIP_VERSION" \
	; \
	pip --version; \
	\
	find /usr/local -depth \
		\( \
			\( -type d -a \( -name test -o -name tests \) \) \
			-o \
			\( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
		\) -exec rm -rf '{}' +; \
	rm -f get-pip.py



RUN pip3 install --upgrade pip

# # If you have cuda9 or cuda10 installed on your machine, please run the following command to install
# # RUN python3 -m pip install paddlepaddle-gpu==2.0.0 -i https://mirror.baidu.com/pypi/simple
RUN python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

WORKDIR /usr/src/app
COPY ./ ./

RUN pip install -r requirements.txt

ENTRYPOINT python3 -m paddle.distributed.launch --gpus '0'  tools/train.py -c configs/rec/srcn_ic.yml