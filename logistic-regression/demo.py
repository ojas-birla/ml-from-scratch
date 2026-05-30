import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

np.random.seed(42)
X0 = np.random.randn(50, 2) + np.array([-2, -2])
X1 = np.random.randn(50, 2) + np.array([2, 2])

X = np.vstack([X0, X1])
y = np.array([0]*50 + [1]*50)

model = LogisticRegression(lr=0.1, epochs=1000)
model.fit(X, y)

predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(accuracy*100)

# plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid).reshape(xx.shape)

plt.contourf(xx, yy, probs, levels=50, cmap='RdBu', alpha=0.6)
plt.colorbar(label='Predicted probability of class 1')

plt.scatter(X0[:, 0], X0[:, 1], color='blue', label='Class 0', edgecolors='k')
plt.scatter(X1[:, 0], X1[:, 1], color='red', label='Class 1', edgecolors='k')

plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig('assets/decision_boundary.png', dpi=150, bbox_inches='tight')
plt.show()