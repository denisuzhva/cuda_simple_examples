import numpy as np
import tensorflow as tf
import time



# Metadata
dim = 100   # tf.constant(100)
iters = int(1e5)    # tf.constant(int(1e6))
batch_size = int(1e3)   # BE CAREFUL

# Tensorflow constants
a_tf = tf.constant(np.random.randn(dim, dim))   # tf.random.normal([dim, dim]) -- in Tensorflow 2.x
b_tf = tf.constant(np.random.randn(dim, dim))
i_tf = tf.constant(0)
iters_tf = tf.constant(iters // batch_size)

# Tensorflow datasets
dataset = tf.data.Dataset.from_tensors((a_tf, b_tf)).repeat(iters).batch(batch_size)

# Iterator
tf_iterator = dataset.make_one_shot_iterator()
a_next, b_next = tf_iterator.get_next()

# Condition of the loop
# @tf.function -- in Tensorflow 2.x
def cond(i, iters, a, b):
    return tf.less(i, iters)

# Body of the loop
# @tf.function -- in Tensorflow 2.x
def body(i, iters, a, b):
    c = tf.matmul(a, b)
    i = tf.add(i, 1)
    return [i, iters, a, b]

# The model (the loop)
res = tf.while_loop(cond, body, [i_tf, iters_tf, a_next, b_next])

# All calculations are performed during session 
with tf.Session() as sess:

    # Time start
    t0 = time.time()

    sess.run(res)

    # Time stop, print
    print(time.time() - t0)
