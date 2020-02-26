FROM tensorflow/tensorflow:1.15.0-gpu-py3

ARG user
ARG uid
RUN useradd -ms /bin/bash -u $uid $user

RUN mkdir /home/$user/TfOptFlow
COPY requirements.txt /home/$user/TfOptFlow

RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y ffmpeg && \
    apt-get install -y libsm6 libxext6 libxrender-dev

RUN pip3 install --upgrade pip

WORKDIR /home/$user/TfOptFlow

RUN pip3 install -r requirements.txt
RUN rm -rf requirements.txt

USER $user