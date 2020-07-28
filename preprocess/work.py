import numpy as np

from model.net import Net
import torch

arr = np.ones((1, 1, 128, 128, 128), dtype=np.float32)
coord = torch.rand((1, 3, 32, 32, 32), dtype=torch.float32)
n = Net()
out = n(torch.from_numpy(arr), coord)
print(out)

print('------------------')
print(out[0].shape, out[1].shape)
