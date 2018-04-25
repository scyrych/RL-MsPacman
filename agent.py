import tensorflow as tf


class DQN:
    def __init__(self):
        self.input_height = 88
        self.input_width = 80
        self.input_channels = 1
        self.conv_n_maps = [32, 64, 64]
        self.conv_kernel_sizes = [(8, 8), (4, 4), (3, 3)]
        self.conv_strides = [4, 2, 1]
        self.conv_paddings = ["SAME"] * 3
        self.conv_activation = [tf.nn.relu] * 3
        self.n_hidden_in = 64 * 11 * 10
        self.n_hidden = 512
        self.hidden_activation = tf.nn.relu
        self.n_outputs = 9
        self.initializer = tf.contrib.layers.variance_scaling_initializer()

    def _zipped_params(self):
        return zip(self.conv_n_maps, self.conv_kernel_sizes,
                   self.conv_strides, self.conv_paddings, self.conv_activation)

    def create_model(self, state, name):
        prev_layer = state / 128.0
        with tf.variable_scope(name) as scope:
            for n_maps, kernel_size, strides, padding, activation in self._zipped_params():
                prev_layer = tf.layers.conv2d(prev_layer, filters=n_maps, kernel_size=kernel_size,
                                              strides=strides, padding=padding, activation=activation,
                                              kernel_initializer=self.initializer)

            last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, self.n_hidden_in])
            hidden = tf.layers.dense(last_conv_layer_flat, self.n_hidden, activation=self.hidden_activation,
                                     kernel_initializer=self.initializer)
            outputs = tf.layers.dense(hidden, self.n_outputs, kernel_initializer=self.initializer)

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}

        return outputs, trainable_vars_by_name
