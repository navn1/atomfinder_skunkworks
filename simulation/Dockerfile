# FROM nvidia/cuda:11.1.1-runtime-ubuntu20.04
FROM cupy/cupy:latest
# RUN mkdir /app

# WORKDIR /app

ADD simulationScript.py /


# RUN apt-get update
# RUN apt-get upgrade -y
# RUN apt-get install -y python3-pip
# RUN python3 --version 

# RUN apt-get install -y python3-pip

RUN pip install abtem
# RUN pip uninstall -y numpy
# RUN pip install numpy==1.21 
# RUN pip install cupy-cuda111

 CMD ["python3", "/simulationScript.py", "data/crystal.cif", "data/miller_indices.txt"]
# COPY commandsToRun.sh /commands.sh
# RUN ["chmod", "+x", "/commands.sh"]
# ENTRYPOINT ["/commands.sh"]


