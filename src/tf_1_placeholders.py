import numpy as np
import tensorflow as tf
import time



# Matrix dimensionality (along one axis)
dim = 100

# Numpy matrices
a_np = np.random.randn(dim, dim)
b_np = np.random.randn(dim, dim)

# Tensorflow placeholders (for feed data)
a_tf = tf.placeholder(tf.float32, shape=[dim, dim])
b_tf = tf.placeholder(tf.float32, shape=[dim, dim])

# THE MODEL
c_tf = tf.matmul(a_tf, b_tf)

# All calculations are performed during session 
with tf.Session() as sess:
    
    # Time start
    t0 = time.time()

    # The loop
    for _ in range(int(1e5)):
        sess.run(c_tf,              # what is calculated
            feed_dict={a_tf : a_np,     # what is 
                       b_tf : b_np})    # fed

    # Time stop, print
    print(time.time() - t0)
    



