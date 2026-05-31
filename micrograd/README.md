# Micrograd from Scratch

## What is it?
The Micrograd engine's job is to find the gradient with respect to all its dependent variables. The Entire idea of ML and Deep learning relies on gradient descent. The Micrograd engine makes the task of calculating derivatives for Loss function with respec to the weights in order to minizmize loss and improve the model

## How to run
```
python verification.py
```

## Verification
```
x=2.0, y=-3.0, z=1.0
f = 4.9993

Gradients:
df/dx = 4.0036
df/dy = -0.0029
df/dz = -0.0015

Numerical gradient of x: 4.0037
Autograd gradient of x:  4.0036
Match: True

(ml) D:\Coding\AI\ml-from-scratch\micrograd>python verification.py
x=2.0, y=-3.0, z=1.0
f = 4.9993

Gradients:
df/dx = 4.0036
df/dy = -0.0029
df/dz = -0.0015

Numerical gradient of x: 4.0037
Autograd gradient of x:  4.0036
Match: True
```

## What I learned
- The Most Trivial thing I learn through this project is that AI isnt magic, it is only some random number that humans cant understand but have the power of doing great things
- **Why `+=` and not `=` for gradients** — a variable can appear 
  in multiple operations. For example if `x` is used in both `a = x*y` 
  and `b = x+z`, gradients flow into `x` from both paths. Using `=` 
  would overwrite one contribution and give wrong gradients. `+=` 
  accumulates all contributions correctly.

- **Why topological sort matters** — backprop multiplies each node's 
  local gradient by `out.grad` (the gradient flowing in from the output). 
  If you process nodes in the wrong order, `out.grad` hasn't been 
  computed yet and you multiply by zero. Topological sort guarantees 
  every node is processed only after all nodes that depend on it.

- **Chain rule in plain English** — the gradient of any variable is 
  its local gradient multiplied by the gradient of whatever it feeds 
  into. This is how gradients flow backwards through the entire network 
  from loss all the way to weights.

## References
Andrej Karpathy - Neural Networks: Zero to Hero