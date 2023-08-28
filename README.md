# SALF
PyTorch implementation of Straggler-Aware Layer-Wise Low-Latency Federated Learning (submitted to ICASSP 2024)
![image](https://github.com/langnatalie/SALF/assets/55830582/972e14e0-b09f-48dc-8f0d-e3bb3b782443)

## Introduction
In this work we propose _Straggler-Aware Layer-Wise Low-Latency Federated Learning (SALF)_, that leverages the optimization procedure of neural networks via backpropagation to update the global model in a _layer-wise_ fashion. This repository contains a basic PyTorch implementation of SALF. Please refer to our [paper-TODO](https://arxiv.org/abs/2208.10888) for more details.

## Usage
This code has been tested on Python 3.7.3, PyTorch 1.8.0 and CUDA 11.1.

### Prerequisite
1. PyTorch=1.8.0: https://pytorch.org
2. scipy
3. tqdm
4. matplotlib
5. torchinfo
6. TensorboardX: https://github.com/lanpa/tensorboardX

### Training
```
python main.py --exp_name=salf --stragglers salf --stragglers_percent 0.9 --up_to_layer 1 --data mnist --model mlp
```

### Testing
```
python main.py --exp_name=salf --eval 
```
