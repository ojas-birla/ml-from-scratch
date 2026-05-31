from engine import Value
import math

x = Value(2.0)
y = Value(-3.0)
z = Value(1.0)

a = x * y          
b = a + z          
c = b.tanh()       
d = c * x         
e = d ** 2         
f = e + Value(1.0) 

print(f"x={x.data}, y={y.data}, z={z.data}")
print(f"f = {f.data:.4f}")

f.backward()

print(f"\nGradients:")
print(f"df/dx = {x.grad:.4f}")
print(f"df/dy = {y.grad:.4f}")
print(f"df/dz = {z.grad:.4f}")

h = 0.0001
x2 = Value(2.0 + h)
a2 = x2 * Value(-3.0)
b2 = a2 + Value(1.0)
c2 = b2.tanh()
d2 = c2 * x2
e2 = d2 ** 2
f2 = e2 + Value(1.0)

numerical_grad = (f2.data - f.data) / h
print(f"\nNumerical gradient of x: {numerical_grad:.4f}")
print(f"Autograd gradient of x:  {x.grad:.4f}")
print(f"Match: {math.isclose(numerical_grad, x.grad, rel_tol=1e-3)}")