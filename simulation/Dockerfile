FROM python:3.8

RUN mkdir /app

WORKDIR /app

ADD simulationScript.py /

RUN pip install abtem

RUN pip install cupy-cuda112

CMD ["python3", "/simulationScript.py $value"]



