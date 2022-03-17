FROM python:3.8

RUN mkdir /app

WORKDIR /app

ADD modelTestScript.py /

RUN pip install atomai

CMD ["python3", "/modelTestScript.py"]