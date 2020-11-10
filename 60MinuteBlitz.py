from __future__ import print_function
import torch

x = torch.zeros(5, 3)
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)


print(x.size())

x = x.new_ones(5, 3)
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
# all pretty basic matrix stuff so far

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x) # Never been a fan of this form of assignment
print(y)

x = torch.rand(5,3)

print(x)
print(x[0, :])

print("Waypoint Marker")

x = torch.randn(4, 4) # TODO: Whats the diff between rand & randn?
y = x.view(16)
z = x.view(-1, 8)  # a -1 means it'll be inferred from other dims
print(x.size(), y.size(), z.size())
print(x)
print(y)
print(z)

x = torch.randn(1)
print(x)
print(x.item())

# TODO: https://pytorch.org/docs/stable/torch.html
# yes im abusing the todo tag as a marker

a = torch.ones(5)
b = a.numpy()
print(b)

a.add_(1) # easiest way to get my brain to agree with this is to think of it as a C++ pointer, which it probably is tbh.
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


if torch.cuda.is_available():
    device = torch.device("cuda") # creates device to send to GPU compute
    y = torch.ones_like(x, device=device) # creating *on* it
    x = x.to(device) # can also do x.to("cuda"), but this seems more elegant imo
    z = x + y # im noting that both on GPU, i wonder what happens if both aren't...
    print(z) # interesting it specifies when not CPU
    print(z.to("cpu", torch.double))

g = torch.ones(5, 3)

if False:#if torch.cuda.is_available():
    device = torch.device("cuda") # creates device to send to GPU compute
    h = torch.ones_like(g, device=device) # creating *on* it
    j = g + h  # solves the question on line 71 - "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!""
    print(z)
    print(z.to("cpu", torch.double))


