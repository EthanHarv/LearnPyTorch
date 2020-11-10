# Some really quick and dirty code to check a Brillaint.org excercise

import torch

w = torch.zeros(1,2) # initalize the zero-d out vectors
b = torch.tensor([0])

x = torch.tensor([[-1, 1], [ 0,  -1], [ 10,  1]]) # 3 datapoints
y = torch.tensor([[1], [-1], [1]])

def sign(num):
    if num > 0:
        return 1
    elif num < 0:
        return -1
    elif num == 0:
        return 0

# w•x+b
# Im assuming this is a HORRIBLY inefficent way of checking im just burnt out and tired rn and want to finish this lesson so pseudocode works
def isDone(weight, bias, x, y):
    for i in range(0, len(x)-1):
        if sign((weight * x[i]).sum() + bias) != y[i]: # Checks all 3 and continues
            return False
    return True

# w(k+1) = w(k)+ x(i)y(i)
while isDone(w, b, x, y) == False: # Update values untill classified
    for i in range(0, len(x-1)):
        #print(w)
        #print(b)
        if sign((w * x[i]).sum() + b) != y[i]: # Only apply to incorrectly identified points
            w = w + x[i] * y[i]
            b = b + y[i]

print(w)
print(b)

print("Sum:", (w.sum() + b).item())