import tensorflow as tf
from models.base_model import BaseModel
from utils.layer_utils import conv_2d, max_pool, flatten_layer, fc_layer, drop_out


class LeNet(BaseModel):
    def __init__(self, sess, conf):
        super(LeNet, self).__init__(sess, conf)
        self.build_network(self.inputs_aug)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('LeNet'):
            x = conv_2d(x, filter_size=5, num_filters=16, name='conv_1')
            x = conv_2d(x, filter_size=5, num_filters=16, name='conv_2')
            x = max_pool(x, 2, 2, 'pool_1')
            x = drop_out(x, self.keep_prob_pl)

            x = conv_2d(x, filter_size=5, num_filters=32, name='conv_3')
            x = conv_2d(x, filter_size=5, num_filters=32, name='conv_4')
            x = max_pool(x, 2, 2, 'pool_2')
            x = drop_out(x, self.keep_prob_pl)

            x = conv_2d(x, filter_size=5, num_filters=64, name='conv_5')
            x = conv_2d(x, filter_size=5, num_filters=64, name='conv_6')
            x = max_pool(x, 2, 2, 'pool_3')
            x = drop_out(x, self.keep_prob_pl)

            x = conv_2d(x, filter_size=5, num_filters=128, name='conv_7')
            x = conv_2d(x, filter_size=5, num_filters=128, name='conv_8')
            x = max_pool(x, 2, 2, 'pool_4')
            x = drop_out(x, self.keep_prob_pl)

            x = conv_2d(x, filter_size=5, num_filters=256, name='conv_9')
            x = conv_2d(x, filter_size=5, num_filters=256, name='conv_10')
            x = max_pool(x, 2, 2, 'pool_5')
            x = drop_out(x, self.keep_prob_pl)

            x = flatten_layer(x)
            x = fc_layer(x, 500, name='fc_1')
            x = drop_out(x, self.keep_prob_pl)

            self.logits = fc_layer(x, self.conf.num_cls, name='fc_2', use_relu=False)
