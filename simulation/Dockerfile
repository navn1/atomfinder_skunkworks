FROM python:3.8

RUN mkdir /app

WORKDIR /app

ADD simulationTestScript.py /

RUN pip install abtem

RUN pip install cupy-cuda112

CMD ["python3", "/simulationTestScript.py"]



