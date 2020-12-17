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
i_tf = tf.constant(0)
iters_tf = tf.constant(int(1e5))

# Condition of the loop
def cond(i, iters, a, b):
    return tf.less(i, iters)

# Body of the loop
def body(i, iters, a, b):
    c = tf.matmul(a, b)
    i = tf.add(i, 1)
    return [i, iters, a, b]

# The model (the loop)
res = tf.while_loop(cond, body, [i_tf, iters_tf, a_tf, b_tf])

# All calculations are performed during session 
with tf.Session() as sess:

    # Time start
    t0 = time.time()

    sess.run(res)

    # Time stop, print
    print(time.time() - t0)
    
