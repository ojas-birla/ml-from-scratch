import numpy as np

class Value:
    def __init__ (self, data, children=()):
        self.data = data
        self.prev = set(children)
        self.grad = 0.0;
        self._backward = lambda : None

    def __add__ (self, other) :
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward

        return out
    
    def __radd__ (self, other):
        return self + other
    
    def __mul__ (self, other) :
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out
    
    def __rmul__ (self, other):
        return self*other
    
    def __pow__ (self, other):
        assert isinstance(other, (int, float));
        out = Value(self.data ** other, (self, ))
        
        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        return (self * (other ** -1))
    
    def __rtruediv__(self, other):
        return ((self ** -1) * other)

    def __neg__ (self):
        return self * -1
    
    def tanh(self):
        x = np.exp(2*self.data)
        out = Value((x-1)/(x+1), (self, ))

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Value(max(0, self.data), (self, ))
        
        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
        
        out._backward = _backward

        return out

    def backward(self):
        self.grad = 1

        topo = []
        visited = set()

        def build_topo(z):
            visited.add(z)
            for x in z.prev:
                if x not in visited:
                    build_topo(x)
            topo.append(z)
        
        build_topo(self)

        for x in reversed(topo):
            x._backward()