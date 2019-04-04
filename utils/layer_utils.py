import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten
from tflearn.layers.conv import global_avg_pool


def get_num_channels(x):
    return x.get_shape().as_list()[-1]


def weight_variable(name, shape):
    initer = tf.contrib.layers.xavier_initializer(uniform=False)
    return tf.get_variable('W_' + name, dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(name, shape):
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name, dtype=tf.float32,
                           initializer=initial)


def conv_layer(x, num_filters, kernel_size, add_reg=False, stride=1, layer_name="conv"):
    with tf.variable_scope(layer_name):
        num_in_channel = get_num_channels(x)
        shape = [kernel_size, kernel_size, num_in_channel, num_filters]
        weights = weight_variable(layer_name, shape=shape)
        layer = tf.nn.conv2d(input=x,
                             filter=weights,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        print('{}: {}'.format(layer_name, layer.get_shape()))
        biases = bias_variable(layer_name, [num_filters])
        layer += biases
        if add_reg:
            tf.add_to_collection('weights', weights)
        return layer


def flatten_layer(layer):
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat


def fc_layer(x, num_units, add_reg, layer_name):
    in_dim = x.get_shape()[1]
    with tf.variable_scope(layer_name):
        weights = weight_variable(layer_name, shape=[in_dim, num_units])
        tf.summary.histogram('W', weights)
        biases = bias_variable(layer_name, [num_units])
        layer = tf.matmul(x, weights)
        layer += biases
        if add_reg:
            tf.add_to_collection('weights', weights)
        print('{}: {}'.format(layer_name, layer.get_shape()))
        return layer


def max_pool(x, pool_size, stride, name, padding='VALID'):
    """Create a max pooling layer."""
    net = tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride,
                                  padding=padding, name=name)
    print('{}: {}'.format(name, net.get_shape()))
    return net


def drop_out(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def drop_connect(w, keep_prob):
    return tf.nn.dropout(w, keep_prob=keep_prob) * keep_prob


def average_pool(x, pool_size, stride, name, padding='VALID'):
    """Create an average pooling layer."""
    net = tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride,
                                      padding=padding, name=name)
    print('{}: {}'.format(name, net.get_shape()))
    return net


def global_average_pool(x, name='global_avg_pooling'):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)
    """
    net = global_avg_pool(x, name=name)
    print('{}: {}'.format(name, net.get_shape()))
    return net


def batch_normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        out = tf.cond(training,
                      lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                      lambda: batch_norm(inputs=x, is_training=training, reuse=True))
        return out


def concatenation(layers):
    return tf.concat(layers, axis=3)


def relu(x):
    return tf.nn.relu(x)

