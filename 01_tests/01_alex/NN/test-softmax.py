import numpy as np

def softmax(z):
    z -= np.max(a, axis=1, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

a = np.array([[5000, 2, 3], [0, 0, 0], [1, 1, 1], [-1, -1, -1]])
print(a)
print('')
print(a.sum(axis=0, keepdims=True))
print('')
print(a.sum(axis=1, keepdims=True))
print('')

print(np.exp(a))
print('')
print(np.sum(np.exp(a), axis=1, keepdims=True))
print('')
print(softmax(a))
print(np.log(softmax(a)))
print(a)
