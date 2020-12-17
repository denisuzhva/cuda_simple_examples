import numpy as np
import time



# Matrix dimensionality (along one axis)
dim = 100

# Numpy matrices
a_np = np.random.randn(dim, dim)
b_np = np.random.randn(dim, dim)

# Time start
t0 = time.time()

# The loop
for _ in range(int(1e5)):
    a_np.dot(b_np)

# Time stop, print
print(time.time() - t0)
