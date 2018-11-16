from __future__ import print_function
import torch


result = torch.empty(5, 3,2)
result = torch.zeros(5, 3,2)
print(result)

x=result

print(x[:, 1])

x = torch.randn(4, 4)
y = x.view(16)
print(y)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())


x = torch.randn(1)
print(x)
print(x.item())

# Converting torch to numpy
a = torch.ones(5)
print(a)    

b = a.numpy()
print(b)

# convert numpy to torch
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

#cuda tensors
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!



