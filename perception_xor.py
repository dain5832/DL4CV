# import necessary packages
from DL4CV_PractitionerBundle.pyimagesearch import Perceptron
import numpy as np

# construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# trainig(fit) the data
print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

# evaluate the data
print("[INFO] testing perceptron...")

for (x, target) in zip(X, y):
    pred = p.predict(x)
    print("[INFO] data={}, ground-truth={}, pred={}".format(\
        x, target, pred))

