import torch
import torch.nn as nn
from torchvision import models
def vggs():
    '''
    调用torchvision.models里面的vgg，
    修改对应的网络层，同样可以得到目标的backbone。
    '''
    vgg16 = models.vgg16()
    # print(vgg16)
    vggs = vgg16.features
    vggs[16] = nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True)
    vggs[-1] = nn.MaxPool2d(3, 1, 1, 1, ceil_mode=False)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    '''
    方法一：
    '''
    #vggs= nn.Sequential(feature, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True))

    '''
    方法二：
    '''
    vggs.add_module('31',conv6)
    vggs.add_module('32',nn.ReLU(inplace=True))
    vggs.add_module('33',conv7)#torch.Size([1, 1024, 19, 19])
    vggs.add_module('34',nn.ReLU(inplace=True))
    #print(vggs)
    # x = torch.randn(1,3,300,300)
    # print(vggs(x).shape)#torch.Size([1, 1024, 19, 19])

    return vggs


def add_extras(cfg, i, batch_norm=False):
    '''
    为后续多尺度提取，增加网络层
    '''
    layers = []
    # 初始输入通道为 1024
    in_channels = i
    # flag 用来选择 kernel_size= 1 or 3
    flag = False
    for k,v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k+1],
                                    kernel_size=(1,3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]

            flag = not flag # 反转flag

        in_channels = v # 更新 in_channels

    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    '''
    Args:
        vgg: 修改fc后的vgg网络
        extra_layers: 加在vgg后面的4层网络
        cfg: 网络参数，eg:[4, 6, 6, 6, 4, 4] 每个位置生成多少个anchor

        num_classes: 类别，VOC为 20+背景=21
    Return:
        vgg, extra_layers
        loc_layers: 多尺度分支的回归网络
        conf_layers: 多尺度分支的分类网络
    '''
    loc_layers = []
    conf_layers = []
    vgg_layer = [21, -2]
    # 第一部分，vgg 网络的 Conv2d-4_3(21层)， Conv2d-7_1(-2层)
    for k, v in enumerate(vgg_layer):
        # 回归 box*4(坐标)
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k]*4, kernel_size=3, padding=1)]
        # 置信度 box*(num_classes)
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k]*num_classes, kernel_size=3, padding=1)]

    # 第二部分，cfg从第三个开始作为box的个数，而且用于多尺度提取的网络分别为1,3,5,7层
    for k, v in enumerate(extra_layers[1::2],2):
        # 回归 box*4(坐标)
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]*4, kernel_size=3, padding=1)]
        # 置信度 box*(num_classes)
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]*(num_classes), kernel_size=3, padding=1)]

    return vgg, extra_layers, (loc_layers, conf_layers)



if __name__  == "__main__":
    extras = {
        '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        '512': [],
    }
    vgg=vggs()
    extras_layers=add_extras(extras['300'], 1024)
    print('vgg')
    print(vgg)
    print('---------------------------')
    print('extras_layers')
    print(extras_layers)
    print('---------------------------')
    vgg, extra_layers, (l, c) = multibox(vgg,
                                         extras_layers,
                                         [4, 6, 6, 6, 4, 4], 21)
    print(nn.Sequential(*l))
    print('---------------------------')
    print(nn.Sequential(*c))
