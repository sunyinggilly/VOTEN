import os
import sys
sys.path.append('../')
sys.path.append('/')
sys.path.append('../../')
from DataProcess.MetaData import MetaData
from CONFIG import COVTYPE_DATASET_PATH, MODEL_PATH
import tensorflow as tf
from Train.Model.TRNN import TRNN
import random
import numpy as np
random.seed(1)
np.random.seed(0)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    meta = MetaData()
    meta.restore("%s/metadata.meta" % COVTYPE_DATASET_PATH)
    data_train = meta.read_TFRecord("%s/train.tf_record" % COVTYPE_DATASET_PATH)  # 683741
    data_test = meta.read_TFRecord("%s/test.tf_record" % COVTYPE_DATASET_PATH)  # 14798

    model = TRNN(meta.n_dv, n_disc=meta.n_disc, n_cont=meta.n_cont, n_label=meta.n_label, verbose=meta.n_train // 128, lr=0.0003, 
                 dataset_train=data_train, dataset_test=data_test, last_act=tf.nn.leaky_relu, 
                 scale_fac=[44] + [64] * 2, proj_layers=[32, 32, 32], proj_res={},
                 deep_layers=[64] * 2, batch_size=128, activation=tf.nn.leaky_relu,
                 random_seed=1, l2_reg=1e-9)

    model.fit(data_train, meta.n_test, n_step=400 * model.verbose)
    model.predict_and_evaluate(meta.n_test)

    if not os.path.exists("%s/COVTYPE/VOTERS" % MODEL_PATH):
        os.makedirs("%s/COVTYPE/VOTERS" % MODEL_PATH)    
    model.save_model("%s/COVTYPE/VOTERS/model" % MODEL_PATH)
