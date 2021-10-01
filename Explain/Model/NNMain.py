import sys
sys.path.append('../')
sys.path.append('../../')

import warnings
import numpy as np
warnings.filterwarnings("ignore")
from tensorflow.python import debug as tf_debug
import tensorflow as tf
from time import time
import abc
from keras import backend as K
from keras.backend import binary_crossentropy
from tqdm import tqdm

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score, average_precision_score


def shuffle_in_union_scary(data):
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    for group, X in data.items():
        np.random.set_state(rng_state)
        if X is not None:
            np.random.shuffle(X)


def evaluate(y_true, y_pred):
    prc = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return prc, auc


def get_batches(data, batch_size):
    n_samples = len(data["label"])
    batches = []
    for i in range(0, n_samples, batch_size):
        now_in = {}
        for group, X in data.items():
            X_b = X[i: min(i + batch_size, n_samples)] if X is not None else None
            now_in[group] = X_b
        batches.append(now_in)
    return batches


class NNMain(object):
    def __init__(self, n_dv, n_disc, n_cont, n_label, batch_size, activation,
                 learning_rate, verbose, random_seed, l2_reg, is_debug=False, is_reg=False):
        self.n_dv = n_dv
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.verbose = verbose
        self.activation = activation
        self.l2_reg = l2_reg
        self.weights = {}
        self.feature_map = {}
        self.inputs = {}
        self.is_debug = is_debug
        self.n_label = n_label

        self.n_disc = n_disc
        self.n_cont = n_cont
        self.is_reg = is_reg

    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def run_batch(self, data_batch, train, voting_mask):
        if train:
            preds, y_true, loss, _ = self.sess.run((self.pred, self.Y, self.loss, self.optimizer), feed_dict=self.get_dict(data_batch, voting_mask=voting_mask, train=True))
        else:
            preds, y_true = self.sess.run((self.pred, self.Y), feed_dict=self.get_dict(data_batch, voting_mask=voting_mask, train=False))
        return preds, y_true

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            # build graph
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # ------------------------ feature map -----------------------------------------
            self.feature_map['label'] = tf.placeholder(tf.float32, [None, self.n_label], name='label') 
            self.feature_map['discrete_feats'] = tf.placeholder(tf.int64, [None, self.n_disc], name='Xd')
            self.feature_map['continuous_feats'] = tf.placeholder(tf.float32, [None, self.n_cont], name='Xc')

            # ------------------- placeholders: label, weight, dropout, feature_mask---------------
            self.mask = [tf.placeholder(tf.float32, shape=[self.n_disc + self.n_cont, self.deep_layers[0]], name='mask_0')]
            for i, layer in enumerate(self.deep_layers[:-1]):
                self.mask.append(tf.placeholder(tf.float32, shape=[layer, self.deep_layers[i + 1]], name='mask_%d' % i))

            self.is_training = tf.placeholder(tf.bool, [])  # [-1]

            self.inputs = self.feature_map
            self.Y = self.inputs['label']
            self.Xc = self.inputs['continuous_feats']
            self.Xd = self.inputs['discrete_feats']

            # -------------------- build graph --------------------------------------------
            self.logits, self.loss = self._build_graph()

            # -------------------- loss function ------------------------------------------
            if not self.is_reg:
                self.pred = tf.nn.sigmoid(self.logits) if self.n_label == 1 else tf.nn.softmax(self.logits, axis=1)
                if self.n_label == 1:
                    self.compo_loss = binary_crossentropy(self.Y, self.logits, from_logits=True)  # N * n_label
                else:
                    self.compo_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits)  # N * n_label
                self.loss += tf.reduce_mean(self.compo_loss)  # N, 1
            else:
                self.pred = self.logits
                self.loss += tf.losses.mean_squared_error(labels=self.Y, predictions=self.logits)

            # -------------------- optimizer -----------------------------------------------
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                        beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)

            # init
            self.saver = tf.train.Saver(list(self.weights.values()), max_to_keep=1)
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            K.set_session(self.sess)
            self.sess.run(init)
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

            # debug
            if self.is_debug:
                self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
                self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            # number of params
            print("#params: %d" % self._count_parameters())

    def fit(self, data_train, data_test, voting_mask, n_step=10000):
        valid_result = []
        predictions_train, label_train = [], []
        t1 = time()
        shuffle_in_union_scary(data_train)
        batches = get_batches(data_train, self.batch_size)
        p = 0
        for now_step in range(n_step):
            data_batch = batches[p]
            p = p + 1
            preds, y_true = self.run_batch(data_batch, train=True)
            predictions_train.extend(preds)
            label_train.extend(y_true)
            if now_step > 0 and now_step % self.verbose == 0:
                train_eval = self._evaluate_metrics(predictions_train, label_train)
                predictions_train, label_train = [], []
                print('[%d]' % (now_step // self.verbose), end='')
                predictions_test = self.predict(data_test, voting_mask)
                label_test = data_test['label']
                valid_eval = self._evaluate_metrics(predictions_test, label_test)
                valid_result.append(valid_eval[0][1])
                self.print_result(train_eval, endch=' | ')
                self.print_result(valid_eval, endch='')
                print('[%.1f s]' % (time() - t1))
                t1 = time()
            if p == len(batches):
                p = 0
                shuffle_in_union_scary(data_train)
                batches = get_batches(data_train, self.batch_size)

    def _dict_input(self, data, train, voting_mask):
        feed_dict = {self.is_training: train}
        for i, mask in enumerate(self.mask):
            feed_dict[mask] = voting_mask[i]
        for placeholder_name, placeholder in self.inputs.items():
            feed_dict[placeholder] = data[placeholder_name]
        return feed_dict

    def print_result(self, train_eval, endch="\n"):
        printstr = ""
        for i, name_val in enumerate(train_eval):
            if i != 0:
                printstr += ','
            printstr += '%s: %f' % name_val
        print(printstr, end=endch)

    def predict(self, data, voting_mask):
        batches = get_batches(data, self.batch_size)
        predictions = []
        for data_batch in batches:
            preds = self.sess.run(self.pred, feed_dict=self.get_dict(data_batch, train=False, voting_mask=voting_mask))
            predictions.extend(preds)
        return predictions

    def predict_with_mask(self, data, voting_mask, return_feat=None, score_type='pred'):
        batches = get_batches(data, self.batch_size)
        predictions = []
        feature_inputs = []
        score_op = self.pred
        if score_type != 'pred':
            score_op = self.logits
        for data_batch in batches:
            # for data_batch in tqdm(batches):
            if return_feat is None:
                preds = self.sess.run(score_op, feed_dict=self.get_dict(data_batch, train=False, voting_mask=voting_mask))
            else:
                preds, feats = self.sess.run((score_op, self.feats_in[:, return_feat]), feed_dict=self.get_dict(data_batch, train=False, voting_mask=voting_mask))
                feature_inputs.extend(feats)
            predictions.extend(preds)
        if return_feat is None:
            return predictions
        else:
            return predictions, feature_inputs

    def predict_and_evaluate(self, data, voting_mask):
        predictions_test, label_test = [], []
        data_batches = get_batches(data, self.batch_size)
        for data_batch in data_batches:
            preds, y_true = self.run_batch(data_batch=data_batch, train=False, voting_mask=voting_mask)
            predictions_test.extend(preds)
            label_test.extend(y_true)
        valid_eval = self._evaluate_metrics(predictions_test, label_test)
        self.print_result(valid_eval, endch='')
        return predictions_test

    def _count_parameters(self):
        total_parameters = 0
        for name, variable in self.weights.items():
            shape = variable.get_shape()
            print(name, shape)
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    def _evaluate_metrics(self, predictions, labels):
        metrics = []
        if not self.is_reg:
            if self.n_label == 1:
                y_true = [u[0] for u in labels]
                y_pred = [u[0] + 1e-10 for u in predictions]                        
                prc = average_precision_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_pred)
                ll = log_loss(y_true, y_pred)
                metrics.extend([('prc', prc), ('auc', auc), ('ll', ll)])
            else:
                y_true = labels
                y_pred = predictions
                prc_macro = average_precision_score(y_true, y_pred, average='macro')
                auc_macro = roc_auc_score(y_true, y_pred, average='macro')
                prc_micro = average_precision_score(y_true, y_pred, average='micro')
                auc_micro = roc_auc_score(y_true, y_pred, average='micro')
                ll = log_loss(y_true, y_pred)
                metrics.extend([('prc_macro', prc_macro), ('auc_macro', auc_macro), ('prc_micro', prc_micro), ('auc_micro', auc_micro), ('ll', ll)])
        else:
            # 算mae和mse和r1
            rmse = math.sqrt(mean_squared_error(labels, predictions))
            mae = mean_absolute_error(labels, predictions)
            metrics.extend([('rmse', rmse), ('mae', mae)])
        return metrics

    def save_model(self, save_path):
        self.saver.save(self.sess, save_path)

    def load_model(self, save_path):
        self.saver.restore(self.sess, save_path)

    def get_concept(self, data, mid_l, mid_h, voting_mask=None):
        if voting_mask is None:
            voting_mask = self.voting_mask_one
        batches = get_batches(data, self.batch_size)
        node_out = self.nodes[mid_l][:, mid_h]
        concept_val = []
        for data_batch in batches:
            node_val = self.sess.run(node_out, feed_dict=self.get_dict(data_batch, train=False, voting_mask=voting_mask))
            concept_val.extend(node_val)
        return concept_val

    def get_voting_to(self, data, mid_l, mid_h, voting_mask):
        batches = get_batches(data, self.batch_size)
        links_out = self.links[mid_l - 1][:, :, mid_h]
        voting_val = []
        for data_batch in batches:
            links_val = self.sess.run(links_out, feed_dict=self.get_dict(data_batch, train=False, voting_mask=voting_mask))
            voting_val.extend(links_val)
        return voting_val

    def get_voting_input_and_link(self, data, mid_l, mid_f, mid_t, voting_mask):
        batches = get_batches(data, self.batch_size)
        link = self.links[mid_l][:, mid_f, mid_t]
        concept_input = self.nodes[mid_l][:, mid_f]
        voting_val, voter_val = [], []
        for data_batch in batches:
            input_val, link_val = self.sess.run((concept_input, link), feed_dict=self.get_dict(data_batch, train=False, voting_mask=voting_mask))
            voting_val.extend(link_val)
            voter_val.extend(input_val)
        return voter_val, voting_val

    def mid_output_avg(self, data_batch, voting_mask):  # 一个batch的output取平均
        rets = self.sess.run(self.nodes + self.links, feed_dict=self.get_dict(data_batch, train=False, voting_mask=voting_mask))
        nodes = rets[: len(self.nodes)]
        links = rets[len(self.nodes):]

        nodes = [node_layer.mean(axis=0) for node_layer in nodes]
        links = [link_layer.mean(axis=0) for link_layer in links]
        return nodes, links

    def mid_output_std(self, data_batch, nodes_avg, links_avg, voting_mask):
        rets = self.sess.run(self.nodes + self.links, feed_dict=self.get_dict(data_batch, train=False, voting_mask=voting_mask))
        nodes = rets[: len(self.nodes)]
        links = rets[len(self.nodes):]

        nodes = [np.square(node_layer - node_avg).mean(axis=0) for node_layer, node_avg in zip(nodes, nodes_avg)]
        links = [np.square(link_layer - link_avg).mean(axis=0) for link_layer, link_avg in zip(links, links_avg)]
        return nodes, links

    def mid_node_raw(self, data):
        batches = get_batches(data, self.batch_size)
        ret = [[] for u in self.nodes]
        for data_batch in batches:
            ret_now = self.sess.run(self.nodes, feed_dict=self.get_dict(data_batch, train=False, voting_mask=self.voting_mask_one))
            for u, v in zip(ret, ret_now):
                u.extend(v)
        return [np.mat(u) for u in ret]

    def mid_out_statistics(self, data, voting_mask):
        batches = get_batches(data, self.batch_size)
        n_now = 0
        nodes_avg, links_avg = None, None
        for data_batch in batches:
            n_batch = len(data_batch['label'])
            nodes, links = self.mid_output_avg(data_batch, voting_mask)  # 一个batch的均值
            if n_now == 0:
                nodes_avg = [np.zeros_like(node_layer) for node_layer in nodes]
                links_avg = [np.zeros_like(link_layer) for link_layer in links]

            for u in range(len(nodes)):  # zip(nodes, links):
                node_add = nodes[u]
                nodes_avg[u] = nodes_avg[u] * (n_now / (n_now + n_batch)) + node_add * (n_batch / (n_now + n_batch))

            for u in range(len(links)):  # zip(nodes, links):
                link_add = links[u]
                links_avg[u] = links_avg[u] * (n_now / (n_now + n_batch)) + link_add * (n_batch / (n_now + n_batch))
            n_now += n_batch

        nodes_std, links_std = None, None
        n_now = 0
        for data_batch in batches:
            n_batch = len(data_batch['label'])
            nodes, links = self.mid_output_std(data_batch, nodes_avg, links_avg, voting_mask=voting_mask)
            if n_now == 0:
                nodes_std = [np.zeros_like(node_layer) for node_layer in nodes]
                links_std = [np.zeros_like(link_layer) for link_layer in links]

            for u in range(len(nodes)):
                node_add = nodes[u]
                nodes_std[u] = nodes_std[u] * (n_now / (n_now + n_batch)) + node_add * (n_batch / (n_now + n_batch))

            for u in range(len(links)):  # zip(nodes, links):
                link_add = links[u]
                links_std[u] = links_std[u] * (n_now / (n_now + n_batch)) + link_add * (n_batch / (n_now + n_batch))

            n_now += n_batch

        nodes_std = [np.sqrt(u) for u in nodes_std]
        links_std = [np.sqrt(u) for u in links_std]
        return nodes_avg, nodes_std, links_avg, links_std

    def part_feature_prediction(self, data, voting_mask):
        batches = get_batches(data, self.batch_size)
        predictions = []
        for data_batch in batches:
            preds = self.sess.run(self.pred, feed_dict=self.get_dict(data_batch, train=False, voting_mask=voting_mask))
            predictions.extend(preds)
        return predictions

    def get_dict(self, data, train=False, voting_mask=None):
        feed_dict = self._dict_input(data=data, train=train,
                                     voting_mask=self.voting_mask_one if voting_mask is None else voting_mask)
        return feed_dict

    @abc.abstractmethod
    def _build_graph(self):
        pass
