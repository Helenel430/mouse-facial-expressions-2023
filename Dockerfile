FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt update 
RUN apt install -y build-essential wget

# Install miniconda to manage python 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /root/miniconda
ENV PATH $PATH:/root/miniconda/bin
RUN conda init bash

# Install python dependencies
COPY . /tmp 
RUN cd /tmp && pip install -r requirements.txt
RUN pip install "deeplabcut[tf]"

# Setup the ipykernel to run notebooks
RUN pip install ipykernel
RUN conda run -n base python -m ipykernel install --user