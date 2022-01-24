# Automatic Pruning Rate Derivation for Structured Pruning of Deep Neural Networks
Structured pruning method for automatic pruning rate derivation for ICPR 2022

Automatic Pruning is a Python module for pruning neural networks.
This module has the following features.
* The pruning rate of each layer can be determined automatically.
* It can also be applied to convolution layers to which BatchNorm layers are not connected and fully connected layers.  
* Pre-trained model & pruned model data for example codes are published in following site.  
  Pre-trained model : https://zenodo.org/record/5725006#.YZ5cSNDP0uU  
  Pruned model      : https://zenodo.org/record/5725038#.YZ5cY9DP0uU  
  
|Dataset|Model|Pre-trained model accuracy(%)|Pruned model accuracy(%)|Pre-trained model size(MB)|Pruned model size(MB)|Model size compression ratio(%)| 
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|MNIST|3-layer MLP|98.44|97.48|7.46|1.14|84.7|
|CIFAR10|AlexNet|90.59|89.62|145.50|1.87|98.7|
|CIFAR10|vgg11|85.70|84.71|515.23|0.80|99.8|
|CIFAR10|vgg11_bn|92.39|91.58|112.64|2.60|97.7|
|CIFAR10|vgg16_bn|93.78|93.82|60.04|6.26|89.6|
|CIFAR10|ResNet18|92.62|91.71|44.78|1.56|96.5|
|CIFAR10|ResNet32|92.63|92.70|1.92|0.69|64.0|
|CIFAR10|ResNet56|93.39|93.40|3.52|0.93|73.5|
|CIFAR10|ResNet110|93.68|93.70|7.12|1.31|81.7|

## Requirements

Automatic Pruner requires:
* Python (>= 3.6.7)
* Torch (>= 1.5.0a0+ba48f58)
* Torchvision (>= 0.6.0+cu101)
* Numpy (>= 1.18.2)
* tqdm (>= 4.62.0)

## Quick start
### Run automatic pruner
1. Move to sample code directory  
```
cd /examples/<sample>
```
2. Download pre-trained model from https://zenodo.org/record/5725006#.YZ5cSNDP0uU to sample code directory  
```
>>> ls /examples/<sample>/*.pt  
pretrained_xxx.pt  
```
3. Execute `run.sh`  
```
chmod +x run.sh && ./run.sh
```
### Run inference with pruned model
1. Move to sample code directory  
```
cd /examples/<sample>
```
2. Download pruned model from https://zenodo.org/record/5725038#.YZ5cY9DP0uU to sample code directory
```
>>> ls /examples/<sample>/*.pt
pruned_xxx.pt
```
3. Execute `run_pruned.sh`
```
chmod +x run_pruned.sh && ./run_pruned.sh

