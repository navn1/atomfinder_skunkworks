FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# RUN mkdir /app

# WORKDIR /app

# ADD modelTrainingScript.py /

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

RUN pip install atomai

CMD ["python3", "data/modelTrainingScript.py", "data/Trainingdata.npz"]