import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import os
import numpy as np
from DataProcess.MetaData import MetaData
from CONFIG import *
from model_factory import get_model
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def solve(data_name, model_name):
    data_path = eval("%s_DATASET_PATH" % data_name)

    meta = MetaData()
    meta.restore("%s/metadata.meta" % data_path)
    data_train = meta.read_pkl("%s/train.pkl" % data_path)
    data_test = meta.read_pkl("%s/test.pkl" % data_path)

    data = data_train.copy()
    for key, val in data.items():
        if val is None:
            continue
        if not isinstance(data[key], list): 
            data[key] = val.tolist()
            data_test[key] = data_test[key].tolist()
        data[key].extend(data_test[key])
    model = get_model(data_name, model_name, meta, use='SaveInter')

    tmp_layers = [model.n_cont + model.n_disc] + model.deep_layers
    voting_mask_one = [np.ones((tmp_layers[i], tmp_layers[i + 1])).tolist() for i, _ in enumerate(tmp_layers[:-1])]
    voter_mask_one = [[1] * n_node for n_node in tmp_layers[:-1]]
    with open("%s/%s/%s/masks.pkl" % (MODEL_PATH, data_name, model_name), 'wb') as f:
        pickle.dump({'voter_mask': voter_mask_one, 'voting_mask': voting_mask_one}, f)

    nodes_avg, nodes_std, links_avg, links_std = model.mid_out_statistics(data)
    with open("%s/%s/%s/statistics.pkl" % (MODEL_PATH, data_name, model_name), 'wb') as f:
        pickle.dump({"nodes_avg": nodes_avg, "nodes_std": nodes_std, "links_avg": links_avg, "links_std": links_std}, f)


if __name__ == '__main__':
    solve("KDC", "VOTERS")
    solve("COVTYPE", "VOTERS")
    solve("IJCAI18X", "VOTERS")
    solve("CENSINCOME", "VOTERS")
    