import tensorflow as tf
from SaveInter.Model.TRNN import TRNN as TRNN_save
from Explain.Model.TRNN import TRNN as TRNN_explain

from CONFIG import MODEL_PATH


def param_factory(data_name, model_name, meta):
    if data_name == "IJCAI18X":
        params = {
            "n_dv": meta.n_dv,
            "n_disc": meta.n_disc,
            "n_cont": meta.n_cont,
            "n_label": meta.n_label,
            "verbose": meta.n_train // 128, 
            "lr": 0.0001, 
            "last_act": tf.nn.leaky_relu, 
            "scale_fac": [120, 8, 8, 8], 
            "proj_layers": [8, 8, 8], 
            "proj_res": {},
            "deep_layers": [8, 8], 
            "batch_size": 2048,
            "activation": tf.nn.leaky_relu,
            "random_seed": 1, 
            "l2_reg": 1e-9}
    elif data_name == "KDC":
        params = {
            "n_dv": meta.n_dv, 
            "n_disc": meta.n_disc, 
            "n_cont": meta.n_cont, 
            "n_label": meta.n_label, 
            "verbose": meta.n_train // 128, 
            "lr": 0.0003, 
            "last_act": tf.nn.leaky_relu, 
            "scale_fac": [50, 16, 16], 
            "proj_layers": [64, 96, 64], 
            "proj_res": {},
            "deep_layers": [16, 16], 
            "batch_size": 4096, 
            "activation": tf.nn.leaky_relu,
            "random_seed": 1, 
            "l2_reg": 1e-9}
    elif data_name == "COVTYPE":
        params = {
            "n_dv": meta.n_dv, 
            "n_disc": meta.n_disc, 
            "n_cont": meta.n_cont, 
            "n_label": meta.n_label, 
            "verbose": meta.n_train // 128, 
            "lr": 0.0003, 
            "last_act": tf.nn.leaky_relu, 
            "scale_fac": [44, 64, 64], 
            "proj_layers": [32, 32, 32], 
            "proj_res": {},
            "deep_layers": [64, 64], 
            "batch_size": 2048, 
            "activation": tf.nn.leaky_relu,
            "random_seed": 1, 
            "l2_reg": 1e-9}
    elif data_name == "CENSINCOME":
        params = {
            "n_dv": meta.n_dv, 
            "n_disc": meta.n_disc, 
            "n_cont": meta.n_cont, 
            "n_label": meta.n_label,
            "verbose": meta.n_train // 64, 
            "lr": 0.0001, 
            "last_act": tf.nn.leaky_relu, 
            "scale_fac": [112, 4, 4], 
            "proj_layers": [8, 64, 16],
            "proj_res": {},
            "deep_layers": [4, 4], 
            "batch_size": 64, 
            "activation": tf.nn.leaky_relu,
            "random_seed": 1, 
            "l2_reg": 1e-9
        }
    return params


def get_model(data_name, model_name, meta, use, voting_mean=None):
    assert model_name == 'VOTERS', 'only support VOTEN explanation currently.'
    params = param_factory(data_name, model_name, meta)
    if voting_mean is not None:
        params['voting_mean'] = voting_mean
    if use == 'Explain':
        model = TRNN_explain(**params)
    else:
        model = TRNN_save(**params)
    model.load_model("%s/%s/%s/model" % (MODEL_PATH, data_name, model_name))
    return model
