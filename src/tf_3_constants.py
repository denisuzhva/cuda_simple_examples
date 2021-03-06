import numpy as np
import tensorflow as tf
import time



# Matrix dimensionality (along one axis)
dim = 100

# Numpy matrices
a_np = np.random.randn(dim, dim)
b_np = np.random.randn(dim, dim)

# Tensorflow constants
a_tf = tf.constant(a_np)
b_tf = tf.constant(b_np)

# THE MODEL
c_tf = tf.matmul(a_tf, b_tf)

# All calculations are performed during session 
with tf.Session() as sess:

    # Time start
    t0 = time.time()

    # The loop
    for _ in range(int(1e5)):
        sess.run(c_tf)    # what is calculated

    # Time stop, print
    print(time.time() - t0)
    

