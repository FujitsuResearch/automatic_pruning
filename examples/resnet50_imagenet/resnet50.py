# resnet50.py COPYRIGHT Fujitsu Limited 2022

import torch.nn as nn
import torch.nn.functional as F

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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):

    expansion = 4
 
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        n_in_channels=None,
        n_channels1=None,
        n_channels2=None,
        n_channels3=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(n_in_channels, n_channels1)
        self.bn1 = norm_layer(n_channels1)
        self.conv2 = conv3x3(n_channels1, n_channels2, stride, groups, dilation)
        self.bn2 = norm_layer(n_channels2)
        self.conv3 = conv1x1(n_channels2, n_channels3)
        self.bn3 = norm_layer(n_channels3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out, identity = zero_padding(out, identity)   # zero padding
        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
                 ch_conv1=64,

                 ch_l10_1=64,
                 ch_l10_2=64,
                 ch_l10_3=256,
                 ch_l10_ds=256,
                 ch_l11_1=64,
                 ch_l11_2=64,
                 ch_l11_3=256,
                 ch_l12_1=64,
                 ch_l12_2=64,
                 ch_l12_3=256,
                 
                 ch_l20_1=128,
                 ch_l20_2=128,
                 ch_l20_3=512,
                 ch_l20_ds=512,
                 ch_l21_1=128,
                 ch_l21_2=128,
                 ch_l21_3=512,
                 ch_l22_1=128,
                 ch_l22_2=128,
                 ch_l22_3=512,
                 ch_l23_1=128,
                 ch_l23_2=128,
                 ch_l23_3=512,

                 ch_l30_1=256,
                 ch_l30_2=256,
                 ch_l30_3=1024,
                 ch_l30_ds=1024,
                 ch_l31_1=256,
                 ch_l31_2=256,
                 ch_l31_3=1024,
                 ch_l32_1=256,
                 ch_l32_2=256,
                 ch_l32_3=1024,
                 ch_l33_1=256,
                 ch_l33_2=256,
                 ch_l33_3=1024,
                 ch_l34_1=256,
                 ch_l34_2=256,
                 ch_l34_3=1024,
                 ch_l35_1=256,
                 ch_l35_2=256,
                 ch_l35_3=1024,

                 ch_l40_1=512,
                 ch_l40_2=512,
                 ch_l40_3=2048,
                 ch_l40_ds=2048,
                 ch_l41_1=512,
                 ch_l41_2=512,
                 ch_l41_3=2048,
                 ch_l42_1=512,
                 ch_l42_2=512,
                 ch_l42_3=2048,
             ):
        super(ResNet50, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, ch_conv1, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = norm_layer(ch_conv1)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        in_ch_l11 = max(ch_l10_ds, ch_l10_3)
        in_ch_l12 = max(in_ch_l11, ch_l11_3)
        self.layer1 = self._make_layer_3(block=block, planes=64, blocks=layers[0],
                                        n_in_channels0=ch_conv1,
                                        n_channels00=ch_l10_1,
                                        n_channels01=ch_l10_2,
                                        n_channels02=ch_l10_3,
                                        n_channels_ds=ch_l10_ds,
                                        n_in_channels1=in_ch_l11,
                                        n_channels10=ch_l11_1,
                                        n_channels11=ch_l11_2,
                                        n_channels12=ch_l11_3,
                                        n_in_channels2=in_ch_l12,
                                        n_channels20=ch_l12_1,
                                        n_channels21=ch_l12_2,
                                        n_channels22=ch_l12_3,
                                       )

        in_ch_l20 = max(in_ch_l12, ch_l12_3)
        in_ch_l21 = max(ch_l20_ds, ch_l20_3)
        in_ch_l22 = max(in_ch_l21, ch_l21_3)
        in_ch_l23 = max(in_ch_l22, ch_l22_3)        
        self.layer2 = self._make_layer_4(block, 128, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0],
                                        n_in_channels0=in_ch_l20,
                                        n_channels00=ch_l20_1,
                                        n_channels01=ch_l20_2,
                                        n_channels02=ch_l20_3,
                                        n_channels_ds=ch_l20_ds,
                                        n_in_channels1=in_ch_l21,
                                        n_channels10=ch_l21_1,
                                        n_channels11=ch_l21_2,
                                        n_channels12=ch_l21_3,
                                        n_in_channels2=in_ch_l22,
                                        n_channels20=ch_l22_1,
                                        n_channels21=ch_l22_2,
                                        n_channels22=ch_l22_3,
                                        n_in_channels3=in_ch_l23,
                                        n_channels30=ch_l23_1,
                                        n_channels31=ch_l23_2,
                                        n_channels32=ch_l23_3,
                                       )

        in_ch_l30 = max(in_ch_l23, ch_l23_3)
        in_ch_l31 = max(ch_l30_ds, ch_l30_3)
        in_ch_l32 = max(in_ch_l31, ch_l31_3)
        in_ch_l33 = max(in_ch_l32, ch_l32_3)
        in_ch_l34 = max(in_ch_l33, ch_l33_3)
        in_ch_l35 = max(in_ch_l34, ch_l34_3)
        self.layer3 = self._make_layer_6(block, 256, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1],
                                        n_in_channels0=in_ch_l30,
                                        n_channels00=ch_l30_1,
                                        n_channels01=ch_l30_2,
                                        n_channels02=ch_l30_3,
                                        n_channels_ds=ch_l30_ds,
                                        n_in_channels1=in_ch_l31,
                                        n_channels10=ch_l31_1,
                                        n_channels11=ch_l31_2,
                                        n_channels12=ch_l31_3,
                                        n_in_channels2=in_ch_l32,
                                        n_channels20=ch_l32_1,
                                        n_channels21=ch_l32_2,
                                        n_channels22=ch_l32_3,
                                        n_in_channels3=in_ch_l33,
                                        n_channels30=ch_l33_1,
                                        n_channels31=ch_l33_2,
                                        n_channels32=ch_l33_3,
                                        n_in_channels4=in_ch_l34,
                                        n_channels40=ch_l34_1,
                                        n_channels41=ch_l34_2,
                                        n_channels42=ch_l34_3,
                                        n_in_channels5=in_ch_l35,
                                        n_channels50=ch_l35_1,
                                        n_channels51=ch_l35_2,
                                        n_channels52=ch_l35_3,
                                        )

        in_ch_l40 = max(in_ch_l35, ch_l35_3)
        in_ch_l41 = max(ch_l40_ds, ch_l40_3)
        in_ch_l42 = max(in_ch_l41, ch_l41_3)
        self.layer4 = self._make_layer_3(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2],
                                        n_in_channels0=in_ch_l40,
                                        n_channels00=ch_l40_1,
                                        n_channels01=ch_l40_2,
                                        n_channels02=ch_l40_3,
                                        n_channels_ds=ch_l40_ds,
                                        n_in_channels1=in_ch_l41,
                                        n_channels10=ch_l41_1,
                                        n_channels11=ch_l41_2,
                                        n_channels12=ch_l41_3,
                                        n_in_channels2=in_ch_l42,
                                        n_channels20=ch_l42_1,
                                        n_channels21=ch_l42_2,
                                        n_channels22=ch_l42_3,
                                        )

        in_ch_fc = max(in_ch_l42, ch_l42_3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch_fc, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer_3(self, block, planes, blocks, stride=1, dilate=False,
                     n_in_channels0=None,
                     n_channels00=None, n_channels01=None, n_channels02=None,
                     n_channels_ds=None,
                     n_in_channels1=None,
                     n_channels10=None, n_channels11=None, n_channels12=None,
                     n_in_channels2=None,
                     n_channels20=None, n_channels21=None, n_channels22=None,
                 ):

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential( conv1x1(n_in_channels0, n_channels_ds, stride), norm_layer(n_channels_ds) )

        self.inplanes = planes * block.expansion
        layers = []

        # layer_0
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                n_in_channels=n_in_channels0,
                n_channels1=n_channels00,
                n_channels2=n_channels01,
                n_channels3=n_channels02,
            )
        )
        # layer_1
        layers.append(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
                n_in_channels=n_in_channels1,
                n_channels1=n_channels10,
                n_channels2=n_channels11,
                n_channels3=n_channels12,
            )
        )
        # layer_2
        layers.append(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
                n_in_channels=n_in_channels2,
                n_channels1=n_channels20,
                n_channels2=n_channels21,
                n_channels3=n_channels22,
            )
        )
        return nn.Sequential(*layers)


    def _make_layer_4(self, block, planes, blocks, stride=1, dilate=False,
                     n_in_channels0=None,
                     n_channels00=None, n_channels01=None, n_channels02=None,
                     n_channels_ds=None,
                     n_in_channels1=None,
                     n_channels10=None, n_channels11=None, n_channels12=None,
                     n_in_channels2=None,
                     n_channels20=None, n_channels21=None, n_channels22=None,
                     n_in_channels3=None,
                     n_channels30=None, n_channels31=None, n_channels32=None,
                 ):

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential( conv1x1(n_in_channels0, n_channels_ds, stride), norm_layer(n_channels_ds) )

        self.inplanes = planes * block.expansion
        layers = []

        # layer_0
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                n_in_channels=n_in_channels0,
                n_channels1=n_channels00,
                n_channels2=n_channels01,
                n_channels3=n_channels02,
            )
        )
        # layer_1
        layers.append(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
                n_in_channels=n_in_channels1,
                n_channels1=n_channels10,
                n_channels2=n_channels11,
                n_channels3=n_channels12,
            )
        )
        # layer_2
        layers.append(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
                n_in_channels=n_in_channels2,
                n_channels1=n_channels20,
                n_channels2=n_channels21,
                n_channels3=n_channels22,
            )
        )
        # layer_3
        layers.append(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
                n_in_channels=n_in_channels3,
                n_channels1=n_channels30,
                n_channels2=n_channels31,
                n_channels3=n_channels32,
            )
        )
        return nn.Sequential(*layers)


    def _make_layer_6(self, block, planes, blocks, stride=1, dilate=False,
                     n_in_channels0=None,
                     n_channels00=None, n_channels01=None, n_channels02=None,
                     n_channels_ds=None,
                     n_in_channels1=None,
                     n_channels10=None, n_channels11=None, n_channels12=None,
                     n_in_channels2=None,
                     n_channels20=None, n_channels21=None, n_channels22=None,
                     n_in_channels3=None,
                     n_channels30=None, n_channels31=None, n_channels32=None,
                     n_in_channels4=None,
                     n_channels40=None, n_channels41=None, n_channels42=None,
                     n_in_channels5=None,
                     n_channels50=None, n_channels51=None, n_channels52=None,
                 ):

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential( conv1x1(n_in_channels0, n_channels_ds, stride), norm_layer(n_channels_ds) )

        self.inplanes = planes * block.expansion
        layers = []

        # layer_0
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                n_in_channels=n_in_channels0,
                n_channels1=n_channels00,
                n_channels2=n_channels01,
                n_channels3=n_channels02,
            )
        )
        # layer_1
        layers.append(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
                n_in_channels=n_in_channels1,
                n_channels1=n_channels10,
                n_channels2=n_channels11,
                n_channels3=n_channels12,
            )
        )
        # layer_2
        layers.append(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
                n_in_channels=n_in_channels2,
                n_channels1=n_channels20,
                n_channels2=n_channels21,
                n_channels3=n_channels22,
            )
        )
        # layer_3
        layers.append(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
                n_in_channels=n_in_channels3,
                n_channels1=n_channels30,
                n_channels2=n_channels31,
                n_channels3=n_channels32,
            )
        )
        # layer_4
        layers.append(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
                n_in_channels=n_in_channels4,
                n_channels1=n_channels40,
                n_channels2=n_channels41,
                n_channels3=n_channels42,
            )
        )
        # layer_5
        layers.append(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
                n_in_channels=n_in_channels5,
                n_channels1=n_channels50,
                n_channels2=n_channels51,
                n_channels3=n_channels52,
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
