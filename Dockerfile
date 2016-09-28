FROM ubuntu:16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y --no-install-recommends \
    build-essential \
    libimage-exiftool-perl \
    python-dev \
    python-numpy \
    python-opencv \
    python-pip \
    python-setuptools \
    python-wheel

COPY . /app

VOLUME /app/config
VOLUME /app/images
WORKDIR /app

ENTRYPOINT ["python", "undistort.py", "--matrix", "/app/config/matrix.txt", "--distortion", "/app/config/distortion.txt", "/app/images/*"]
