import tensorflow as tf
import numpy as np


def summation(team_vectors):
    return tf.reduce_sum(team_vectors, axis=1)

def aveg(team_vectors):
    return tf.reduce_mean(team_vectors, axis=1)

def concat(team_vectors):
    return tf.concat(team_vectors, axis=2)
    #return team_vectors
    #return tf.reshape(team_vectors, [32, -1]) # [batch_size, d * 5]


class GameEmbeddings(object):
    def __init__(self, num_players, embedding_size, agg_function, model_t):
        self.num_players = num_players
        self.embedding_size = embedding_size
        self.bound = np.sqrt(6) / embedding_size
        self.W_off = tf.Variable(tf.truncated_normal([num_players, embedding_size], stddev=self.bound))
        self.W_deff = tf.Variable(tf.truncated_normal([num_players, embedding_size], stddev=self.bound))

        self.model_t = model_t

        if self.model_t == 'conv':
            self.W_out = tf.Variable(tf.truncated_normal([200, 1]))
        else:
            self.hidden_size = 50
            self.W_out = tf.Variable(tf.truncated_normal([self.hidden_size, 1]))

        self.b_out = tf.Variable(tf.constant(0.0))
        self.agg_function = agg_function

    def build_model(self):
        self.team_a = tf.placeholder(tf.int32, [None, 5])
        self.team_b = tf.placeholder(tf.int32, [None, 5])

        self.score_team_a = tf.placeholder(tf.float32, [None])
        self.score_team_b = tf.placeholder(tf.float32, [None])

        self.team_a_off = tf.nn.embedding_lookup(self.W_off, self.team_a)   # [batch_size, 5, d]
        self.team_b_off = tf.nn.embedding_lookup(self.W_off, self.team_b)   # same

        self.team_a_deff = tf.nn.embedding_lookup(self.W_deff, self.team_a)  # [batch_size, 5, d]
        self.team_b_deff = tf.nn.embedding_lookup(self.W_deff, self.team_b)  # same

        # TODO: add more filters of different sizes
        if self.model_t == 'conv':

            self.agg_layer_team_a = tf.reshape(tf.concat([self.team_a_off, self.team_b_deff], axis=1),
                                               [-1, 2, 5, self.embedding_size])
            self.agg_layer_team_b = tf.reshape(tf.concat([self.team_b_off, self.team_a_deff], axis=1),
                                               [-1, 2, 5, self.embedding_size])

            self.W_conv_a = tf.Variable(tf.random_normal([2, 2, self.embedding_size, 20]))
            self.b_conv_a = tf.Variable(tf.random_normal([20]))

            net_a = tf.nn.conv2d(input=self.agg_layer_team_a, name='layer_conv2_a',
                                 filter=self.W_conv_a, strides=[1, 1, 1, 1],
                                 padding='SAME')
            act_a = tf.nn.relu(tf.nn.bias_add(net_a, self.b_conv_a))

            self.W_conv_b = tf.Variable(tf.random_normal([2, 2, self.embedding_size, 20]))
            self.b_conv_b = tf.Variable(tf.random_normal([20]))

            net_b = tf.nn.conv2d(input=self.agg_layer_team_b, name='layer_conv2_b',
                                 filter=self.W_conv_a, strides=[1, 1, 1, 1],
                                 padding='SAME')
            act_b = tf.nn.relu(tf.nn.bias_add(net_b, self.b_conv_b))

            #act_a = tf.nn.dropout(act_a, 0.8)
            #act_b = tf.nn.dropout(act_b, 0.8)

            act_a = tf.contrib.layers.flatten(act_a)
            act_b = tf.contrib.layers.flatten(act_b)
        else:
            self.agg_team_a_off = self.agg_function(self.team_a_off)  # [batch_size, d2]
            self.agg_team_b_off = self.agg_function(self.team_b_off)  # [batch_size, d2]

            self.agg_team_a_deff = self.agg_function(self.team_a_deff)  # [batch_size, d2]
            self.agg_team_b_deff = self.agg_function(self.team_b_deff)

            self.agg_layer_team_a = tf.concat([self.agg_team_a_off, self.agg_team_b_deff], axis=1)
            self.agg_layer_team_b = tf.concat([self.agg_team_b_off, self.agg_team_a_deff], axis=1)

            dims = self.agg_layer_team_a.get_shape().as_list()[1:]
            dim = np.prod(dims)

            self.agg_layer_team_a = tf.reshape([self.agg_layer_team_a], [-1, dim])
            self.agg_layer_team_b = tf.reshape([self.agg_layer_team_b], [-1, dim])

            # out_input = 50
            self.hidden_a = tf.Variable(tf.truncated_normal([2 * 5 * self.embedding_size, self.hidden_size]))
            self.b_a = tf.Variable(tf.random_normal([self.hidden_size]))
            self.hidden_b = tf.Variable(tf.truncated_normal([2 * 5 * self.embedding_size, self.hidden_size]))
            self.b_b = tf.Variable(tf.random_normal([self.hidden_size]))

            net_a = tf.nn.xw_plus_b(self.agg_layer_team_a, self.hidden_a, self.b_a)
            act_a = tf.nn.relu(net_a)
            net_b = tf.nn.xw_plus_b(self.agg_layer_team_b, self.hidden_b, self.b_b)
            act_b = tf.nn.relu(net_b)

        self.pred_team_a = tf.matmul(tf.transpose(self.W_out), tf.transpose(act_a)) \
                            + self.b_out

        self.pred_team_b = tf.matmul(tf.transpose(self.W_out), tf.transpose(act_b)) \
                           + self.b_out

        self.loss_a = tf.reduce_mean(tf.square(self.pred_team_a - self.score_team_a))
        self.loss_b = tf.reduce_mean(tf.square(self.pred_team_b - self.score_team_b))

        self.loss = self.loss_a + self.loss_b + 0.8 * ((tf.nn.l2_loss(self.W_deff) + tf.nn.l2_loss(self.W_off)))
        self.op = tf.train.AdagradOptimizer(0.1).minimize(self.loss)

    def updates(self):
        return [self.loss, self.op, self.pred_team_a, self.pred_team_b]

    def variables(self):
        if self.model_t == 'conv':
            return [self.W_off, self.W_deff, self.W_out, self.b_out, self.W_conv_a,
                    self.b_conv_a, self.W_conv_b, self.b_conv_b]
        else:
            return [self.W_off, self.W_deff, self.W_out, self.b_out, self.hidden_a,
                    self.b_a, self.hidden_b, self.b_b]
