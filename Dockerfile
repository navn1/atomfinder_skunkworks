FROM python:3.8

LABEL author="naveen balu" 

RUN mkdir /app

WORKDIR /app

ADD pyTest.py /

RUN pip install abtem

RUN pip install cupy-cuda112

CMD ["python3", "/pyTest.py"]



