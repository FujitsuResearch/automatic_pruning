# Automatic Pruning Rate Derivation for Structured Pruning of Deep Neural Networks
  
## Requirements

Automatic Pruner requires:
* Python 3.6
* Torch  >=1.6
* Torchvision >= 0.6.0+cu101
* Numpy >= 1.18.2
* tqdm  >= 4.62.0

## Quick start
1. Move to sample code directory.  
```
cd /examples/<sample>
```
2. Prepare pre-trained model, and dataset for re-training such as CIFAR-10 and ImageNet.  
Pre-trained models for example codes can be downloaded from the following links.
* https://zenodo.org/record/5725006#.YZ5cSNDP0uU (for CIFAR-10)   
* https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py  (for ImageNet)  
3. Set the file path of the dataset and pre-trained model in `run.sh`.  
Example of `/examples/resnet34_imagenet/run.sh`  
```
CUDA_VISIBLE_DEVICES='0' python3 main.py --data ../dataset/imagenet/ --pretrained_model_path ../pretrained_model/resnet34-b627a593.pth > log.log
```
* `--data` The file path for retraining dataset, e.g. CIFAR-10 and ImageNet.
* `--pretrained_model_path` The file path of pre-trained model.

4. Execute `run.sh`.  
```
chmod +x run.sh && ./run.sh
```

### Note: When running inference with pruned model by this code
The number of channels of pruned model by this code is changed from the model before pruning.
So, when run inference with pruned model by this code, change the number of channels defined in model file (e.g. `resnet32.py`).

## Results
<p align="center">
<img src="images/results.PNG" width="900">
</p>
