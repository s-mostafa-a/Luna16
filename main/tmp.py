import numpy as np
from model.loss import Loss
from model.net import Net
import torch

arr = np.ones((1, 1, 128, 128, 128), dtype=np.float32)
arr0 = np.zeros((1, 32, 32, 32, 3, 5), dtype=np.float32)
coord = torch.rand((1, 3, 32, 32, 32), dtype=torch.float32)
n = Net()
out = n(torch.from_numpy(arr), coord)
# print(out)
loz = Loss()
print('------------------')
print(out.shape)
# torch.Size([1, 32, 32, 32, 3, 5])
print(loz(out, arr0))
