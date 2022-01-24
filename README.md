# Automatic Pruning Rate Derivation for Structured Pruning of Deep Neural Networks
Structured pruning method for automatic pruning rate derivation for ICPR 2022

Automatic Pruning is a Python module for pruning neural networks.
This module has the following features.
* The pruning rate of each layer can be determined automatically.
* It can also be applied to convolution layers to which BatchNorm layers are not connected and fully connected layers.  
* Pre-trained model & pruned model data for example codes are published in following site.  
  Pre-trained model : https://zenodo.org/record/5725006#.YZ5cSNDP0uU  
  Pruned model      : https://zenodo.org/record/5725038#.YZ5cY9DP0uU  
  
<p align="center">
<img src="images/results.PNG" width="900">
</p>


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

