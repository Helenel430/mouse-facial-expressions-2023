FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt update 
RUN apt install -y build-essential wget

