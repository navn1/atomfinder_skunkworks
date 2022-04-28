FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# RUN mkdir /app

# WORKDIR /app

# ADD modelTrainingScript.py /
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb

COPY ./cuda-keyring_1.0-1_all.deb cuda-keyring_1.0-1_all.deb 
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

RUN pip install atomai

CMD ["python3", "data/modelTrainingScript.py", "data/Trainingdata.npz"]