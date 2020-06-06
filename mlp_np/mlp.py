# -*- coding: utf-8 -*-
"""
@author: Li Xianyang
"""
import numpy as np


class Mlp:
    def softmax(self):
        pass

    def __init__(self, input_size, hide_size, out_size):
        self.fc1_w = np.random.normal(size=(input_size, hide_size))
        self.fc1_b = np.random.normal(size=(hide_size))
        self.fc2_w = np.random.normal(size=(hide_size, out_size))
        self.fc2_b = np.random.normal(size=(out_size))

    def forward(self, x):
        hide = np.dot(x, self.fc1_w) + self.fc1_b
        hide_act = np.clip(hide, 0, 100)
        hide2 = np.dot(hide_act, self.fc2_w) + self.fc2_b

        return hide, hide_act, hide2

    def backward(self, x, y, lr=1e-3):
        hide, hide_act, hide2 = self.forward(x)
        pre = hide2
        loss = np.mean((pre - y) ** 2, axis=0).sum()
        d_h2 = 2 * (pre - y) / pre.shape[0]

        d_b2 = d_h2.sum(0)
        d_w2 = hide_act.T.dot(d_h2)

        d_hide_act = d_h2.dot(self.fc2_w.T)
        d_hide = (hide > 0) * d_hide_act

        d_b1 = d_hide.sum(0)
        d_w1 = x.T.dot(d_hide)

        self.fc2_w -= lr * d_w2
        self.fc1_w -= lr * d_w1
        self.fc2_b -= lr * d_b2
        self.fc1_b -= lr * d_b1
        return loss

    def __call__(self, x):
        return self.forward(x)[-1]


IN, HIDE, OUT = 10, 100, 10
model = Mlp(IN, HIDE, OUT)
BATCH_SIZE = 8

for i in range(10000):
    x = np.random.normal(size=(BATCH_SIZE, IN))
    y = x + 1
    loss = model.backward(x, y)
    print(loss)
x = np.random.normal(size=(BATCH_SIZE, IN))
y = model(x)
print(y - x)
print((y-x-1).sum())
