# -*- coding: utf-8 -*-
"""
@author: Li Xianyang
"""
import torch
from torch import nn
import torch.nn.functional  as F
from torch import optim


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=(3, 3))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(out)
        out = self.pool(out)
        out = self.fc(out)


model = MyModel()
model.train()
# for param in model.parameters():
#     print(param.requires_grad)

optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
print('model state_dict:')
print(type(model.state_dict()))
for param in model.state_dict():
    # print(param,model.state_dict()[param])
    print(model.state_dict()[param].size())
    break

for param in optimizer.state_dict():
    print(param,optimizer.state_dict()[param])


torch.save(model.state_dict(),f='./model_state_dict.pth')
model.load_state_dict(torch.load('./model_state_dict.pth'))
torch.save(model,'model.pth')
model2=torch.load('model.pth')
print(model2)



torch.save({
    'epoch':1,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict':optimizer.state_dict(),

},'model_checkpoint.pth')

model=MyModel()
optimizer=optim.SGD(model.parameters(),lr=0.1)
checkpoint=torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch=checkpoint['epoch']
print(optimizer)
print(epoch)
