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
import math
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error


def shuffle_in_unison_scary(data):
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    for group, X in data.items():
        np.random.set_state(rng_state)
        if X is not None:
            np.random.shuffle(X)


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
    def __init__(self, n_dv, n_disc, n_cont, n_label, batch_size, activation, dataset_train, dataset_test,
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

        self.example_train = dataset_train
        self.example_test = dataset_test

    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8 
        return tf.Session(config=config)

    def run_batch(self, train):
        if train:
            preds, y_true, loss, _ = self.sess.run((self.pred, self.Y, self.loss, self.optimizer), feed_dict=self.get_dict(train=True))
        else:
            preds, y_true = self.sess.run((self.pred, self.Y), feed_dict=self.get_dict(train=False))
        return preds, y_true

    def _parse_function(self, data):
        examples = tf.parse_single_example(data, self.feature_map)
        return examples

    def _init_graph(self):
#self.graph = tf.Graph()
#        with self.graph.as_default():
            if self.random_seed != -1:
                tf.set_random_seed(self.random_seed)

            # build graph
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # ------------------------ feature map -----------------------------------------
            self.feature_map['label'] = tf.FixedLenFeature(dtype=tf.float32, shape=[self.n_label])
            self.feature_map['discrete_feats'] = tf.FixedLenFeature(shape=[self.n_disc], dtype=tf.int64)
            self.feature_map['continuous_feats'] = tf.FixedLenFeature(shape=[self.n_cont], dtype=tf.float32)

            # ------------------- placeholders: label, weight, dropout, feature_mask---------------
            self.is_training = tf.placeholder(tf.bool, [])  # [-1]

            self.dataset_train = self.example_train.repeat().shuffle(buffer_size=self.batch_size * 100)
            self.dataset_train = self.dataset_train.apply(tf.contrib.data.map_and_batch(
                lambda record: self._parse_function(record), batch_size=self.batch_size))

            self.dataset_test = self.example_test.repeat()
            self.dataset_test = self.dataset_test.apply(tf.contrib.data.map_and_batch(
                lambda record: self._parse_function(record), batch_size=self.batch_size))

            self.iterator_train = self.dataset_train.make_one_shot_iterator()
            self.iterator_test = self.dataset_test.make_one_shot_iterator()

            self.inputs = tf.cond(self.is_training, lambda: self.iterator_train.get_next(), lambda: self.iterator_test.get_next())
            self.Y = self.inputs['label']
            self.Xc = self.inputs['continuous_feats']
            self.Xd = self.inputs['discrete_feats']

            # -------------------- build graph --------------------------------------------
            self.logits, self.loss = self._build_graph()  # N, n_label

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

    def fit(self, data_train, n_test, n_step=10000):
        valid_result = []
        predictions_train, label_train = [], []
        t1 = time()
        for now_step in range(n_step):
            preds, y_true = self.run_batch(train=True)
            predictions_train.extend(preds)
            label_train.extend(y_true)
            if now_step > 0 and now_step % self.verbose == 0:
                train_eval = self._evaluate_metrics(predictions_train, label_train)
                predictions_train, label_train = [], []
                predictions_test, label_test = [], []
                print('[%d]' % (now_step // self.verbose), end='')
                for i in range(0, n_test, self.batch_size):
                    preds, y_true = self.run_batch(train=False)
                    predictions_test.extend(preds)
                    label_test.extend(y_true)
                valid_eval = self._evaluate_metrics(predictions_test, label_test)
                valid_result.append(valid_eval[0][1])
                self.print_result(train_eval, endch=' | ')
                self.print_result(valid_eval, endch='')
                print('[%.1f s]' % (time() - t1))
                t1 = time()

    def predict_and_evaluate(self, n_test):
        predictions_test, label_test = [], []
        for i in range(0, n_test, self.batch_size):
            preds, y_true = self.run_batch(train=False)
            predictions_test.extend(preds)
            label_test.extend(y_true)
        print(len(label_test))
        valid_eval = self._evaluate_metrics(predictions_test, label_test)
        self.print_result(valid_eval, endch='')

    def get_test_data(self, n_test):
        name_lst = []
        val_lst = []
        ret_dct = {}
        for col_name, val in self.inputs.items():
            name_lst.append(col_name)
            val_lst.append(val)
            ret_dct[col_name] = []
        predictions = []
        for i in range(0, n_test, self.batch_size):
            ret_lst = self.sess.run(val_lst + [self.pred], feed_dict=self.get_dict(train=False))
            for name, val in zip(name_lst, ret_lst[:-1]):
                ret_dct[name].extend(val)
            predictions.extend(ret_lst[-1])
        return ret_dct, predictions

    def print_result(self, train_eval, endch="\n"):
        printstr = ""
        for i, name_val in enumerate(train_eval):
            if i != 0:
                printstr += ','
            printstr += '%s: %f' % name_val
        print(printstr, end=endch)

    def get_dict(self, train=True):  # 这里少东西，data没feed进去
        feed_dict = {self.is_training: train}
        return feed_dict

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

    @abc.abstractmethod
    def _build_graph(self):
        pass
