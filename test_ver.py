import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K

x = tf.constant([float(i) for i in range(1,11)])
y = x#tf.constant([[10.0],[5.0],[8.0],[9.0],[1.0],[4.0],[6.0],[7.0],[0.0],[1.0]])
x_m = K.mean(x)
x_v = K.var(x)
y_m = K.mean(y)
y_v = K.var(y)
xy_c = K.mean((x - x_m) * (y - y_m))

print(x_m)
print(y_m)
print(x_v)
print(y_v)
print(xy_c)
print(2*xy_c/(x_v + y_v + (y_m-x_m)**2))
