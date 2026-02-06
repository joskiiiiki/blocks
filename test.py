import numpy as np
from opensimplex import OpenSimplex
from time import time as t

SIZE = 1000

x = np.arange(SIZE, dtype=np.int64)
y = np.arange(SIZE, dtype=np.int64)

gen = OpenSimplex(0)

t0 = t()

noise = gen.noise2array(x, y)

t1 = t()

print("vectorized", t1- t0)

t0 = t()

noise = np.zeros((SIZE, SIZE), dtype=np.int64)
for y in range(SIZE):
    for x in range(SIZE):
        noise[x, y] = gen.noise2(x, y)

t1 = t()

print("loop", t1 - t0)
