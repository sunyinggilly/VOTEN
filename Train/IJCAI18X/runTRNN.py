import os
import sys
sys.path.append('../')
sys.path.append('/')
sys.path.append('../../')
from DataProcess.MetaData import MetaData
from CONFIG import IJCAI18X_DATASET_PATH, MODEL_PATH
import tensorflow as tf
from Train.Model.TRNN import TRNN
import random
import numpy as np
import pickle
random.seed(1)
np.random.seed(1)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    meta = MetaData()
    meta.restore("%s/metadata.meta" % IJCAI18X_DATASET_PATH)
    data_train = meta.read_TFRecord("%s/train.tf_record" % IJCAI18X_DATASET_PATH)  # 683741
    data_test = meta.read_TFRecord("%s/test.tf_record" % IJCAI18X_DATASET_PATH)  # 14798

    model = TRNN(meta.n_dv, n_disc=meta.n_disc, n_cont=meta.n_cont, n_label=meta.n_label, verbose=meta.n_train // 128, lr=0.0001, 
                 dataset_train=data_train, dataset_test=data_test, last_act=tf.nn.leaky_relu, 
                 scale_fac=[120, 8, 8, 8], proj_layers=[8, 8, 8], proj_res={},
                 deep_layers=[8, 8], batch_size=128, activation=tf.nn.leaky_relu,
                 random_seed=1, l2_reg=1e-9)

    model.fit(data_train, meta.n_test, n_step=80 * model.verbose)
    model.predict_and_evaluate(meta.n_test)

    if not os.path.exists("%s/IJCAI18X/VOTERS" % MODEL_PATH):
        os.makedirs("%s/IJCAI18X/VOTERS" % MODEL_PATH)   
    model.save_model("%s/IJCAI18X/VOTERS/model" % MODEL_PATH)
