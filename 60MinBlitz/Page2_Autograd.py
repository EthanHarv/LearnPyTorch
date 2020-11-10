from __future__ import print_function
import torch

# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

# Basically describes functions to output the computed differentation of the applied actions on a tensor, if im reading this right
# This is where I got lost before, because even though I knew the rules of calc it wasn't quite intuitive to me yet, but now moreso its clicking
# or Im completely misunderstanding, but im writing this before doing the excercise, so we'll see lol

# If you set its attribute .requires_grad as True, it starts to track all operations on it. (.detach() to stop)
# .backward() = all the gradients computed automatically, stored in .grad

# with torch.no_grad():

# In simple terms:
# If ya want to compute the derivatives, you can call .backward() on a Tensor


# "If Tensor is a scalar (i.e. it holds a one element data), 
# you don’t need to specify any arguments to backward(), however if it has more elements, 
# you need to specify a gradient argument that is a tensor of matching shape."

x = torch.ones(2, 2, requires_grad=True) # Create & Start trackin
print(x)

y = x + 2
print(y)

z = y * y * 3
out = z.mean()

print(z)
print(out) # dont know why we're taking the mean of 4 identical numbers, but ig its just to display that the .mean() function exists

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1)) # Untracked
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn) # lists out the function, but only displays most recent it seems 
b = b + 3
print(b.grad_fn) # yep (im assuming it stores it all?)

# this paper likes jumping around, so just to re-fresh my mind, heres our operations on order again
print(x)
print(y)
print(z)
print(out)

out.backward() # ah ok now the .mean() makes sense so we dont have to deal with supplying gradients for this example

print(x.grad) # d(out)/dx wrt x

# out tensor = mean of 3(x_i + 2)^2 (because order of ops, going x + 2, then squaring and *3)
# so in essence the function we're deriving is simply 3(x+2)^2, which is easy to derive to 3*2(x+2)*(1) = 6(x+2), and then /4 because of the summation operation
# so we get do/dx = 3/2(x+2), so for x=1 do/dx=4.5

# see https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#gradients for jacobian reference
# some of this is going over my head, need to touch up on linear algebra, but I get most of it and in the interest of time I'm going to steam ahead

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000: # keep doubling, until euclid norm is > 1000 
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float) # I understand \ on a mathematical level whats happening, but on a conceptual level im a bit lost. I plan to come back to this.
y.backward(v)

print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)