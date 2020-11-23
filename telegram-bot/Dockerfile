FROM tensorflow/tensorflow:2.3.0

# install debian packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install essentials
    build-essential \
    wget \
    git \
    # requirements for opencv
    ffmpeg \
    libsm6 \
    libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
    # pip install
 && pip3 --no-cache-dir install \
    opencv-python \
    uvicorn \
    python-multipart \
    fastapi \
    fastapi-utils \
    pydantic\
    typing\
    pathlib\
    pytelegrambotapi \
    # make dirs
 && mkdir /app 
 # && mkdir /app/weights && mkdir /app/tmp

WORKDIR /app
COPY . .

EXPOSE 80/tcp
EXPOSE 80/udp

ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]