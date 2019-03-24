import tensorflow as tf
import numpy as np

def combine(tensor_data, dim=0):
    sp = tensor_data.get_shape()
    cm_1 = []
    cm_2 = []
    for i in range(sp[0]):
        for j in range(i+1, sp[0]):
            cm_1.append(tensor_data[i, :])
            cm_2.append(tensor_data[j, :])
    combine_tensor = tf.add(cm_1, cm_2)
    return combine_tensor


def combine_2(tensor_data, dim=0):
    sp = tensor_data.get_shape()
    combine_tensor = [tf.add(tensor_data[0, :],tensor_data[1, :])]
    print type(combine_tensor)
    for i in range(sp[0]):
        for j in range(i + 1, sp[0]):
            if i == 0 and j == 1:
                break
            else:
                combine_tensor = tf.concat([combine_tensor, [tf.add(tensor_data[i, :],tensor_data[j, :])]], 0)
    return combine_tensor

tf_v = tf.Variable([[1,2],[3,4]])
tf_v1 = tf.constant([1,2])
print(type(tf_v1), tf_v1)
tf_v2 = tf.Variable([[1], [2]])
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    a = sess.run(tf_v*tf_v1 + tf_v2)
    print(type(a))
    print(a.shape)
    print(a)
    print(sess.run(tf_v))
    print "*"*20
    print(sess.run(tf_v*tf_v1))