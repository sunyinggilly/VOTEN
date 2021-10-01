import sys
sys.path.append('../')
sys.path.append('../../')

import tensorflow as tf
from Explain.Model.NNMain import NNMain
import numpy as np


def FPRD(tensor, n_dv, n_dim, dim_out, activation, l2_reg, name):  # input: [-1, n_dim]
    weights = {}
    glorot = np.sqrt(2.0 / (n_dim + dim_out))
    w = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(n_dv, dim_out)), dtype=np.float32, name="FPRD_weights")
    weights["%s_weights" % name] = w
    y_emb = tf.nn.embedding_lookup(w, tensor)  # [-1, n_dim, dim_out]
    if activation is not None:
        y_emb = activation(y_emb)
    penalty_loss = tf.contrib.layers.l2_regularizer(l2_reg)(w)
    return y_emb, weights, penalty_loss


def FPRC_transform(tensor, layers, res_layers, n_dim, out_dim, activation, l2_reg, name, last_act=None):
    weights = {}
    deep_x = tf.expand_dims(tensor, -1)
    layers_use = [1] + layers + [out_dim]
    lst = []  # -1, n_dim, 1
    layer_id = 0
    penalty_loss = 0
    glorot = 0.1
    for dim_in, dim_out in zip(layers_use[:-1], layers_use[1:]):
        layer_id += 1
        lst.append(deep_x)
        w = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(n_dim, dim_in, dim_out)), dtype=np.float32,
                        name="%s_%d_weights" % (name, layer_id))
        weights["%s_%d_weights" % (name, layer_id)] = w
        penalty_loss += tf.contrib.layers.l2_regularizer(l2_reg)(w)
        b = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(n_dim, dim_out)), dtype=np.float32,
                        name=name + "%s_%d_bias" % (name, layer_id))
        weights["%s_%d_bias" % (name, layer_id)] = b

        deep_x = tf.einsum('ibd,bdo->ibo', deep_x, w) + b

        if layer_id != len(layers_use) - 1:
            deep_x = activation(deep_x)
        if layer_id in res_layers:
            deep_x += lst[res_layers[layer_id]]

    deep_x = last_act(deep_x) if last_act is not None else deep_x

    return deep_x, weights, penalty_loss


class TRNN(NNMain):
    def __init__(self, n_dv, n_disc, n_cont, n_label, verbose=1, lr=0.1, 
                 deep_layers=[16, 16], batch_size=32, proj_layers=[5, 5, 5], proj_res={3: 1}, last_act=tf.nn.tanh,
                 scale_fac=[10], activation=tf.nn.relu, random_seed=2019, l2_reg=0.0, voting_mean=[], is_reg=False):

        super(TRNN, self).__init__(n_dv=n_dv, n_disc=n_disc, n_cont=n_cont, batch_size=batch_size, activation=activation, learning_rate=lr,
                                   l2_reg=l2_reg, verbose=verbose, random_seed=random_seed, n_label=n_label, is_reg=is_reg)

        self.deep_layers = deep_layers + [self.n_label]
        self.proj_res = proj_res
        self.proj_layers = proj_layers
        self.nodes, self.links = [], []
        self.last_act = last_act
        self.scale_fac = scale_fac
        self.voting_mean = voting_mean
        self._init_graph()
        tmp_layers = [self.n_cont + self.n_disc] + self.deep_layers
        self.voting_mask_one = [np.ones((tmp_layers[i], tmp_layers[i + 1])).tolist() for i, _ in enumerate(tmp_layers[:-1])]

    def _build_graph(self):
        loss = 0

        # ------------------------ DNN ----------------------------------
        deep_d, weights, loss_now = FPRD(self.Xd, self.n_dv, self.n_disc, self.deep_layers[0], activation=None,
                                         l2_reg=self.l2_reg, name="FPRD")   # [-1, n_d, deep[0]]
        loss += loss_now
        self.weights.update(weights)

        deep_c, weights, loss_now = FPRC_transform(self.Xc, self.proj_layers, self.proj_res, self.n_cont, self.deep_layers[0],
                                                   activation=self.activation, l2_reg=self.l2_reg,
                                                   last_act=self.last_act, name="FPRC_0")   # [-1, n_c, deep[0]]
        loss += loss_now
        self.weights.update(weights)

        self.feats_in = tf.concat((tf.cast(self.Xd, tf.float32), self.Xc), axis=1)
        self.nodes.append(self.feats_in)

        deep_y = tf.concat((deep_d, deep_c), axis=1)
        self.links.append(deep_y)

        sub_mean = tf.constant(self.voting_mean[0])
        print(deep_y)
        deviation = deep_y - sub_mean
        deviation_mask = tf.einsum('ibd,bd->ibd', deviation, self.mask[0])  # 偏移量mask掉
        deviation_val = tf.reduce_sum(deviation_mask, axis=1) / (self.n_cont + self.n_disc)  # N, deep[0]
        deep_y = tf.reduce_sum(sub_mean, axis=0) / (self.n_cont + self.n_disc) + deviation_val

        self.nodes.append(deep_y)
        deep_y = deep_y * self.scale_fac[0]

        loss += loss_now
        self.weights.update(weights)

        # 中间层处理
        for i in range(len(self.deep_layers) - 1):
            dim_in, dim_out = self.deep_layers[i], self.deep_layers[i + 1]
            deep_y, weights, loss_now = FPRC_transform(deep_y, self.proj_layers, self.proj_res, dim_in, dim_out, last_act=self.last_act,
                                                       name="FPRC_%d" % (i + 1), activation=self.activation, l2_reg=self.l2_reg)  # -1, dim_in, dim_out
            loss += loss_now
            self.weights.update(weights)
            self.links.append(deep_y)

            sub_mean = tf.constant(self.voting_mean[i + 1])
            deviation = deep_y - sub_mean
            deviation_mask = tf.einsum('ibd,bd->ibd', deviation, self.mask[i + 1])  # 偏移量mask掉
            deviation_val = tf.reduce_sum(deviation_mask, axis=1) / dim_in  # N, deep[0]
            deep_y = tf.reduce_sum(sub_mean, axis=0) / dim_in + deviation_val

            self.nodes.append(deep_y)
            deep_y = deep_y * self.scale_fac[i + 1]
        logits = deep_y
        return logits, loss
