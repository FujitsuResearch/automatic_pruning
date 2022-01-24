CUDA_VISIBLE_DEVICES='0' python3 main.py --model_path ../pretrained_model/pretrained_cifar10_resnet56.pt --use_gpu --use_DataParallel --pruned_model_path ./pruned_cifar10_resnet56_00.pt > log.log
