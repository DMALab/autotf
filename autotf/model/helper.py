import tensorflow as tf

def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv_layer(bottom, in_channels, out_channels, name):
    with tf.variable_scope(name):
        filt, conv_biases = get_conv_var(3, in_channels, out_channels, name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        relu = tf.nn.relu(bias)

        return relu

def fc_layer(bottom, in_size, out_size, name):
    with tf.variable_scope(name):
        weights, biases = get_fc_var(in_size, out_size, name)

        x = tf.reshape(bottom, [-1, in_size])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc

def get_conv_var(filter_size, in_channels, out_channels, name):
    initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
    filters = get_var(initial_value, name, 0, name + "_filters")

    initial_value = tf.truncated_normal([out_channels], .0, .001)
    biases = get_var(initial_value, name, 1, name + "_biases")

    return filters, biases

def get_fc_var(in_size, out_size, name):
    initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
    weights = get_var(initial_value, name, 0, name + "_weights")

    initial_value = tf.truncated_normal([out_size], .0, .001)
    biases = get_var(initial_value, name, 1, name + "_biases")

    return weights, biases

def get_var(initial_value, name, idx, var_name, trainable=True):
    value = initial_value

    if trainable:
        var = tf.Variable(value, name=var_name)
    else:
        var = tf.constant(value, dtype=tf.float32, name=var_name)

    return var
