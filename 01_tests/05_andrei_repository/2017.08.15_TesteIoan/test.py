v = { i:[j for j in range(i)] for i in range(4) }
print(v)

def zipWith(f, l1, l2):
	return [ f(e1, e2) for (e1, e2) in zip(l1, l2)]

def f(*args, **kwargs):
	print(args, kwargs)

f(1, 2, **{'a': 1, 'b': 2, 'c': 3})

import operator
print(zipWith(operator.add, [1,2,3], [4,5,6]))





