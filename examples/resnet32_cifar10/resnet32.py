# resnet32.py COPYRIGHT Fujitsu Limited 2022

import torch
import torch.nn as nn
import torch.nn.functional as F

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def zero_padding(x1, x2):
    num_ch1 = x1.size()[1]
    num_ch2 = x2.size()[1]
    ch_diff = num_ch1 - num_ch2
    # path1 < path2 : zero padding to path1 tensor
    if num_ch1 < num_ch2:
        ch_diff = -1 * ch_diff
        if ch_diff%2 ==0:
            x1 = F.pad(x1[:, :, :, :], (0, 0, 0, 0, ch_diff//2, ch_diff//2), "constant", 0)
        else:
            x1 = F.pad(x1[:, :, :, :], (0, 0, 0, 0, ch_diff//2, (ch_diff//2)+1), "constant", 0)
    # path1 > path2 : zero padding to path2 tensor
    elif num_ch1 > num_ch2:
        if ch_diff%2 ==0:
            x2 = F.pad(x2[:, :, :, :], (0, 0, 0, 0, ch_diff//2, ch_diff//2), "constant", 0)
        else:
            x2 = F.pad(x2[:, :, :, :], (0, 0, 0, 0, ch_diff//2, (ch_diff//2)+1), "constant", 0)
    return x1, x2

def zero_padding_downsample_shortcut(x1, x2):
    # x1: tensor from residual block
    # x2: tensor for downsample
    num_ch1 = x1.size()[1]
    num_ch2 = x2.size()[1]
    ch_diff = num_ch1 - num_ch2
    #print("ch_diff: ", ch_diff)
    if num_ch1 < num_ch2:
        ch_diff = -1 * ch_diff
        if ch_diff%2 ==0:
            x1 = F.pad(x1[:, :, :, :], (0, 0, 0, 0, ch_diff//2, ch_diff//2), "constant", 0)
            x2 = F.pad(x2[:, :, ::2, ::2], (0, 0, 0, 0, 0, 0), "constant", 0)
        else:
            x1 = F.pad(x1[:, :, :, :], (0, 0, 0, 0, ch_diff//2, (ch_diff//2)+1), "constant", 0)
            x2 = F.pad(x2[:, :, ::2, ::2], (0, 0, 0, 0, 0, 0), "constant", 0)
    # path1 > path2 : zero padding to path2 tensor
    elif num_ch1 > num_ch2:
        if ch_diff%2 ==0:
            x2 = F.pad(x2[:, :, ::2, ::2], (0, 0, 0, 0, ch_diff//2, ch_diff//2), "constant", 0)
        else:
            x2 = F.pad(x2[:, :, ::2, ::2], (0, 0, 0, 0, ch_diff//2, (ch_diff//2)+1), "constant", 0)
    else: # num_ch1 = num_ch2
        x2 = F.pad(x2[:, :, ::2, ::2], (0, 0, 0, 0, 0, 0), "constant", 0) # downsample x2
    return x1, x2   

# for CIFAR-10
class ResNet32(nn.Module):
    def __init__(
        self,
        num_classes=10,

        ch_conv1=16,

        ch_l10conv1=16,
        ch_l10conv2=16,
        ch_l11conv1=16,
        ch_l11conv2=16,
        ch_l12conv1=16,
        ch_l12conv2=16,
        ch_l13conv1=16,
        ch_l13conv2=16,
        ch_l14conv1=16,
        ch_l14conv2=16,

        ch_l20conv1=32,
        ch_l20conv2=32,
        ch_l21conv1=32,
        ch_l21conv2=32,
        ch_l22conv1=32,
        ch_l22conv2=32,
        ch_l23conv1=32,
        ch_l23conv2=32,
        ch_l24conv1=32,
        ch_l24conv2=32,

        ch_l30conv1=64,
        ch_l30conv2=64,
        ch_l31conv1=64,
        ch_l31conv2=64,
        ch_l32conv1=64,
        ch_l32conv2=64,
        ch_l33conv1=64,
        ch_l33conv2=64,
        ch_l34conv1=64,
        ch_l34conv2=64,
    ):
        super(ResNet32, self).__init__()
        self.conv1 = nn.Conv2d(3, ch_conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_conv1)
        self.relu = nn.ReLU(inplace=True)

        # layer1-0
        self.l10_conv1 = nn.Conv2d(ch_conv1, ch_l10conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l10_bn1   = nn.BatchNorm2d(ch_l10conv1)
        self.l10_conv2 = nn.Conv2d(ch_l10conv1, ch_l10conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l10_bn2   = nn.BatchNorm2d(ch_l10conv2)
        # layer1-1
        ch_l11conv1_in=max(ch_conv1,ch_l10conv2)
        self.l11_conv1 = nn.Conv2d(ch_l11conv1_in, ch_l11conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l11_bn1   = nn.BatchNorm2d(ch_l11conv1)
        self.l11_conv2 = nn.Conv2d(ch_l11conv1, ch_l11conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l11_bn2   = nn.BatchNorm2d(ch_l11conv2)
        # layer1-2
        ch_l12conv1_in=max(ch_l11conv1_in, ch_l11conv2)
        self.l12_conv1 = nn.Conv2d(ch_l12conv1_in, ch_l12conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l12_bn1   = nn.BatchNorm2d(ch_l12conv1)
        self.l12_conv2 = nn.Conv2d(ch_l12conv1, ch_l12conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l12_bn2   = nn.BatchNorm2d(ch_l12conv2)
        # layer1-3
        ch_l13conv1_in=max(ch_l12conv1_in, ch_l12conv2)
        self.l13_conv1 = nn.Conv2d(ch_l13conv1_in, ch_l13conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l13_bn1   = nn.BatchNorm2d(ch_l13conv1)
        self.l13_conv2 = nn.Conv2d(ch_l13conv1, ch_l13conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l13_bn2   = nn.BatchNorm2d(ch_l13conv2)
        # layer1-4
        ch_l14conv1_in=max(ch_l13conv1_in, ch_l13conv2)
        self.l14_conv1 = nn.Conv2d(ch_l14conv1_in, ch_l14conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l14_bn1   = nn.BatchNorm2d(ch_l14conv1)
        self.l14_conv2 = nn.Conv2d(ch_l14conv1, ch_l14conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l14_bn2   = nn.BatchNorm2d(ch_l14conv2)


        # layer2-0
        ch_l20conv1_in=max(ch_l14conv1_in, ch_l14conv2)
        self.l20_conv1 = nn.Conv2d(ch_l20conv1_in, ch_l20conv1, kernel_size=3, stride=2, padding=1, bias=False)
        self.l20_bn1   = nn.BatchNorm2d(ch_l20conv1)
        self.l20_conv2 = nn.Conv2d(ch_l20conv1, ch_l20conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l20_bn2   = nn.BatchNorm2d(ch_l20conv2)
        # layer2-1
        ch_l21conv1_in=max(ch_l20conv1_in, ch_l20conv2)
        self.l21_conv1 = nn.Conv2d(ch_l21conv1_in, ch_l21conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l21_bn1   = nn.BatchNorm2d(ch_l21conv1)
        self.l21_conv2 = nn.Conv2d(ch_l21conv1, ch_l21conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l21_bn2   = nn.BatchNorm2d(ch_l21conv2)
        # layer2-2
        ch_l22conv1_in=max(ch_l21conv1_in, ch_l21conv2)
        self.l22_conv1 = nn.Conv2d(ch_l22conv1_in, ch_l22conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l22_bn1   = nn.BatchNorm2d(ch_l22conv1)
        self.l22_conv2 = nn.Conv2d(ch_l22conv1, ch_l22conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l22_bn2   = nn.BatchNorm2d(ch_l22conv2)
        # layer2-3
        ch_l23conv1_in=max(ch_l22conv1_in, ch_l22conv2)
        self.l23_conv1 = nn.Conv2d(ch_l23conv1_in, ch_l23conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l23_bn1   = nn.BatchNorm2d(ch_l23conv1)
        self.l23_conv2 = nn.Conv2d(ch_l23conv1, ch_l23conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l23_bn2   = nn.BatchNorm2d(ch_l23conv2)
        # layer2-4
        ch_l24conv1_in=max(ch_l23conv1_in, ch_l23conv2)
        self.l24_conv1 = nn.Conv2d(ch_l24conv1_in, ch_l24conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l24_bn1   = nn.BatchNorm2d(ch_l24conv1)
        self.l24_conv2 = nn.Conv2d(ch_l24conv1, ch_l24conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l24_bn2   = nn.BatchNorm2d(ch_l24conv2)


        # layer3-0
        ch_l30conv1_in=max(ch_l24conv1_in, ch_l24conv2)
        self.l30_conv1 = nn.Conv2d(ch_l30conv1_in, ch_l30conv1, kernel_size=3, stride=2, padding=1, bias=False)
        self.l30_bn1   = nn.BatchNorm2d(ch_l30conv1)
        self.l30_conv2 = nn.Conv2d(ch_l30conv1, ch_l30conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l30_bn2   = nn.BatchNorm2d(ch_l30conv2)
        # layer3-1
        ch_l31conv1_in=max(ch_l30conv1_in, ch_l30conv2)
        self.l31_conv1 = nn.Conv2d(ch_l31conv1_in, ch_l31conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l31_bn1   = nn.BatchNorm2d(ch_l31conv1)
        self.l31_conv2 = nn.Conv2d(ch_l31conv1, ch_l31conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l31_bn2   = nn.BatchNorm2d(ch_l31conv2)
        # layer3-2
        ch_l32conv1_in=max(ch_l31conv1_in, ch_l31conv2)
        self.l32_conv1 = nn.Conv2d(ch_l32conv1_in, ch_l32conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l32_bn1   = nn.BatchNorm2d(ch_l32conv1)
        self.l32_conv2 = nn.Conv2d(ch_l32conv1, ch_l32conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l32_bn2   = nn.BatchNorm2d(ch_l32conv2)
        # layer3-3
        ch_l33conv1_in=max(ch_l32conv1_in, ch_l32conv2)
        self.l33_conv1 = nn.Conv2d(ch_l33conv1_in, ch_l33conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l33_bn1   = nn.BatchNorm2d(ch_l33conv1)
        self.l33_conv2 = nn.Conv2d(ch_l33conv1, ch_l33conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l33_bn2   = nn.BatchNorm2d(ch_l33conv2)
        # layer3-4
        ch_l34conv1_in=max(ch_l33conv1_in, ch_l33conv2)
        self.l34_conv1 = nn.Conv2d(ch_l34conv1_in, ch_l34conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l34_bn1   = nn.BatchNorm2d(ch_l34conv1)
        self.l34_conv2 = nn.Conv2d(ch_l34conv1, ch_l34conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l34_bn2   = nn.BatchNorm2d(ch_l34conv2)

        # linear
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        ch_linear_in=max(ch_l34conv1_in, ch_l34conv2)
        self.linear = nn.Linear(ch_linear_in, num_classes)

    def forward(self, x):
        x = self.conv1(x) 
        x = self.bn1(x)     
        x = self.relu(x)

        # layer1-0
        identity = x
        x = F.relu(self.l10_bn1(self.l10_conv1(x)))
        x = self.l10_bn2(self.l10_conv2(x))
        x, identity = zero_padding(x, identity)   # zero padding
        x += identity
        x = F.relu(x)
        # layer1-1
        identity = x
        x = F.relu(self.l11_bn1(self.l11_conv1(x)))
        x = self.l11_bn2(self.l11_conv2(x))
        x, identity = zero_padding(x, identity)   # zero padding
        x += identity
        x = F.relu(x)
        # layer1-2
        identity = x
        x = F.relu(self.l12_bn1(self.l12_conv1(x)))
        x = self.l12_bn2(self.l12_conv2(x))
        x, identity = zero_padding(x, identity)   # zero padding
        x += identity
        x = F.relu(x)
        # layer1-3
        identity = x
        x = F.relu(self.l13_bn1(self.l13_conv1(x)))
        x = self.l13_bn2(self.l13_conv2(x))
        x, identity = zero_padding(x, identity)   # zero padding
        x += identity
        x = F.relu(x)
        # layer1-4
        identity = x
        x = F.relu(self.l14_bn1(self.l14_conv1(x)))
        x = self.l14_bn2(self.l14_conv2(x))
        x, identity = zero_padding(x, identity)   # zero padding
        x += identity
        x = F.relu(x)


        # layer2-0
        identity = x
        x = F.relu(self.l20_bn1(self.l20_conv1(x)))
        x = self.l20_bn2(self.l20_conv2(x))
        x, identity = zero_padding_downsample_shortcut(x, identity)   # zero padding with downsample
        x += identity
        x = F.relu(x)
        # layer2-1
        identity = x
        x = F.relu(self.l21_bn1(self.l21_conv1(x)))
        x = self.l21_bn2(self.l21_conv2(x))
        x, identity = zero_padding(x, identity)   # zero padding
        x += identity
        x = F.relu(x)
        # layer2-2
        identity = x
        x = F.relu(self.l22_bn1(self.l22_conv1(x)))
        x = self.l22_bn2(self.l22_conv2(x))
        x, identity = zero_padding(x, identity)   # zero padding
        x += identity
        x = F.relu(x)
        # layer2-3
        identity = x
        x = F.relu(self.l23_bn1(self.l23_conv1(x)))
        x = self.l23_bn2(self.l23_conv2(x))
        x, identity = zero_padding(x, identity)   # zero padding
        x += identity
        x = F.relu(x)
        # layer2-4
        identity = x
        x = F.relu(self.l24_bn1(self.l24_conv1(x)))
        x = self.l24_bn2(self.l24_conv2(x))
        x, identity = zero_padding(x, identity)   # zero padding
        x += identity
        x = F.relu(x)

        # layer3-0   
        identity = x
        x = F.relu(self.l30_bn1(self.l30_conv1(x)))
        x = self.l30_bn2(self.l30_conv2(x))
        x, identity = zero_padding_downsample_shortcut(x, identity)   # zero padding with downsample
        x += identity
        x = F.relu(x)
        # layer3-1
        identity = x
        x = F.relu(self.l31_bn1(self.l31_conv1(x)))
        x = self.l31_bn2(self.l31_conv2(x))
        x, identity = zero_padding(x, identity)   # zero padding
        x += identity
        x = F.relu(x)
        # layer3-2
        identity = x
        x = F.relu(self.l32_bn1(self.l32_conv1(x)))
        x = self.l32_bn2(self.l32_conv2(x))
        x, identity = zero_padding(x, identity)   # zero padding
        x += identity
        x = F.relu(x)
        # layer3-3
        identity = x
        x = F.relu(self.l33_bn1(self.l33_conv1(x)))
        x = self.l33_bn2(self.l33_conv2(x))
        x, identity = zero_padding(x, identity)   # zero padding
        x += identity
        x = F.relu(x)
        # layer3-4
        identity = x
        x = F.relu(self.l34_bn1(self.l34_conv1(x)))
        x = self.l34_bn2(self.l34_conv2(x))
        x, identity = zero_padding(x, identity)   # zero padding
        x += identity
        x = F.relu(x)
        
        x = F.avg_pool2d(x,x.size()[3])
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        
        return x
