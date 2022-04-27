FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# RUN mkdir /app

# WORKDIR /app

ADD modelTestScript.py /

RUN pip install atomai

CMD ["python3", "/modelTestScript.py"]