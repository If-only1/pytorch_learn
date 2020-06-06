from torch import nn
import torch.nn.functional as F
import torch


class SSD(nn.Module):
    '''
    Args:
        phase: string, 可选"train" 和 "test"
        size: 输入网络的图片大小
        base: VGG16的网络层（修改fc后的）
        extras: 用于多尺度增加的网络
        head: 包含了各个分支的loc和conf
        num_classes: 类别数

    return:
        output: List, 返回loc, conf 和 候选框
    '''
    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.size = size
        self.num_classes = num_classes
        # 配置config
        self.cfg = (coco, voc)[num_classes == 21]
        # 初始化先验框
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()
        # basebone 网络
        self.vgg = nn.ModuleList(base)
        # conv4_3后面的网络，L2 正则化
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        # 回归和分类网络
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
        '''
            # 预测使用
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 200, 0.01, 0.045)
        '''
            pass

    def forward(self, x):
        sources, loc ,conf = [], [], []

        # vgg网络到conv4_3
        for i in range(23):
            x = self.vgg[i](x)
        # l2 正则化
        s = self.L2Norm(x)
        sources.append(s)

        # conv4_3 到 fc
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        sources.append(x)

        # extras 网络
        for k,v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            # 把需要进行多尺度的网络输出存入 sources
            if k%2 == 1:
                sources.append(x)

        # 多尺度回归和分类网络
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())  
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
        '''
        # 预测使用
            output = self.detect(
                    # loc 预测
                    loc.view(loc.size(0), -1, 4),
                    # conf 预测
                    self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                    # default box
                    self.priors.type(type(x.data)),
                    )
        '''
            pass
        else:
            output = (
                # loc的输出，size:(batch, 8732, 4)
                loc.view(loc.size(0), -1 ,4),
                 # conf的输出，size:(batch, 8732, 21)
                conf.view(conf.size(0), -1, self.num_classes),
                # 生成所有的候选框 size([8732, 4])
                self.priors,
                )
#            print(type(x.data))
#            print((self.priors.type(type(x.data))).shape)
        return output

    # 加载模型参数
    def load_weights(self, base_file):
        print('Loading weights into state dict...')
        self.load_state_dict(torch.load(base_file))
        print('Finished!')
使用build_ssd()封装函数，增加可读性:

def build_ssd(phase, size=300, num_classes=21):
    # 判断phase是否为满足的条件
    if phase != "test" and phase !="train":
        print("Error: Phase:" + phase +" not recognized!\n")
        return
    # 判断size是否为满足的条件
    if size != 300:
        print("Error: currently only size=300 is supported!")
        return

    # 调用multibox，生成vgg,extras,head
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox['300'], num_classes,
                                     )
    return SSD(phase, size, base_, extras_, head_, num_classes)

# 调试函数
if __name__ == '__main__':
    ssd = build_ssd('train')
    x = torch.randn(1, 3, 300, 300)
    y = ssd(x)
    print("Loc    shape: ", y[0].shape)   
    print("Conf   shape: ", y[1].shape) 
    print("Priors shape: ", y[2].shape)