import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
#matplotlib inline
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
from tensorflow import keras

#index
t = tf.constant([[1.,2.,3.,],[4.,5.,6.,]])
print(t)
print(t[:,1:])
print(t[...,1])

# ops
print(t +10)
print (tf.square(t))
print(t @ tf.transpose(t))

# numpy conversion
print(t.numpy())
print(np.square(t))
np_t = np.array([[1.,2.,3.,],[4.,5.,6.,]])
print(tf.constant(np_t))

#scalars
t = tf.constant(2.212)
print(t.numpy())
print(t.shape)

#string
t = tf.constant('cafe')
print(t)
print(tf.strings.length(t))
print(tf.strings.length(t,unit = 'UTF8_CHAR'))
print(tf.strings.unicode_decode(t,'UTF8'))



#string array
t =tf.constant(["cafe","coffee"])
print(tf.strings.length(t,unit = "UTF8_CHAR"))
r = tf.strings.unicode_decode(t,"UTF8")
print(r)


#ragged tensor

r = tf.ragged.constant([[11,23],[21,33,22],[],[21]])
print(r)
print(r[1])
print(r[1:2])

r2 = tf.ragged.constant([[11,23],[21,33,22],[]])
print(tf.concat([r,r2],axis = 0))

print(r.to_tensor())

#sparse tensor

s = tf.SparseTensor(indices = [[0,1],[1,0],[2,3]],values = [1.,2.,3.],dense_shape = [3,4])
print(s)
print(tf.sparse.to_dense(s))
s2 =s *2.0
print(s2)
try:
    s3 =s +1
except TypeError as ex:
    print(ex)

s4 = tf.constant([
    [12,139],
    [12,131],
    [12,134],
    [12,136]

])



#variables

v= tf.Variable([[1.,2.,3.],[4.4,5.,6.]])
print(v)
print(v.value())
print(v.numpy())


#assign value
v.assign(2*v)
print(v.numpy())
v[0,1].assign(42)
print(v.value())
v[1].assign([4.,66.,77.])
print(v.numpy())


try:
    v[1]= [3.,4.,5]
except TypeError as ex:
    print(ex)