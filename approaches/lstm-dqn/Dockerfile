FROM phusion/baseimage:0.11

RUN locale-gen ru_RU.UTF-8
ENV LANG ru_RU.UTF-8

MAINTAINER m.domrachev.scientist@gmail.com

ENV BUILD_THREADS 4

RUN apt-get update
RUN apt-get --no-install-recommends --no-upgrade -y --force-yes install \
    libtool \
    mc \
    python3.6 \
    python3.6-dev \
    libpython3.6 \
    python3.6-distutils

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6
RUN apt-get -y --force-yes install build-essential uuid-dev libffi-dev curl git docker.io wget graphviz

RUN pip3 install spacy
RUN pip3 install gym==0.10.4
RUN python3.6 -m spacy download en

RUN pip3 install docker
RUN pip3 install textworld==1.0.0
RUN pip3 install torchvision
