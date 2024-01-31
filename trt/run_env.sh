#!/bin/bash

docker run -it --rm -e NVIDIA_VISIBLE_DEVICES=0 --gpus "device=0" --name trt-cookbook --shm-size 16G --ulimit memlock=-1 --ulimit stack=67108864 -v ~/workspace/learn_cuda/trt/work:/work nvcr.io/nvidia/pytorch:23.04-py3 /bin/bash
