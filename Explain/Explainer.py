import sys
sys.path.append('../')
sys.path.append('../../')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from model_factory import get_model
from DataProcess.MetaData import MetaData
from CONFIG import *
import pickle


class Explainer(object):
    def __init__(self, meta, model, nodes_avg, nodes_std, links_avg, links_std):
        self.nodes_avg = nodes_avg
        self.nodes_std = nodes_std
        self.links_avg = links_avg
        self.links_std = links_std
        self.model = model
        self.feat_name = meta.disc_cols + meta.cont_cols
        self.meta = meta
        self.lst_val = [0] * self.meta.n_dv
        for val, ind in self.meta.df_val[['value', 'index']].values.tolist():
            self.lst_val[ind] = val
        self.feat_dct = {}
        for feat_id, feat_name in enumerate(self.feat_name):  # 验：这里需要验证，输入到模型的是不是这样的
            self.feat_dct[feat_name] = feat_id

    def discrete_feat_val(self, feat_val_id):
        return self.lst_val[feat_val_id]

    def get_feature_value(self, data, feat_id):
        if feat_id < self.meta.n_disc:
            return np.array(data['discrete_feats'])[:, feat_id]
        else:
            feat_id -= self.meta.n_disc
            return np.array(data['continuous_feats'])[:, feat_id]

    def get_feature_value_all(self, data):
        return np.concatenate((data['discrete_feats'], data['continuous_feats']), axis=1)

    def get_feat_name(self, feat_id):
        return self.feat_name[feat_id]

    def get_node_name(self, l_id, h_id):
        if l_id == 0:
            return self.get_feat_name(h_id).split('@')[1]
        elif l_id == len(self.model.deep_layers):
            return "O%d" % h_id
        else:
            return "L%dH%d" % (l_id, h_id)

    def mask_score(self, data, voting_masks_now, return_feat, score_type='pred'):
        if return_feat is None:
            predictions = self.model.predict_with_mask(data, voting_masks_now, return_feat=return_feat, score_type=score_type)
            return predictions
        else:
            predictions, features = self.model.predict_with_mask(data, voting_masks_now, return_feat=return_feat, score_type=score_type)
            return predictions, features

    def feature_subset_score(self, data, features, return_feat):
        voting_masks_now = [np.ones_like(np.mat(mat)) for mat in self.model.voting_mask_one]
        voting_masks_now[0] = np.zeros_like(voting_masks_now[0])
        for feat_id in features:
            voting_masks_now[0][feat_id, :] = 1
        return self.mask_score(data, voting_masks_now, return_feat)

    def feature_subsetdrop_score(self, data, features, return_feat):
        voting_masks_now = [np.ones_like(np.mat(mat)) for mat in self.model.voting_mask_one]
        for feat_id in features:
            voting_masks_now[0][feat_id, :] = 0
        return self.mask_score(data, voting_masks_now, return_feat)

    def feature_single_score(self, data, return_feat):
        return self.feature_subset_score(data, [return_feat], return_feat=return_feat)

    def feature_function_plot(self, data, feat_id):
        predictions, features = self.feature_single_score(data, feat_id, return_feat=feat_id)
        df = pd.DataFrame()
        df['x'] = predictions
        df['y'] = features
        df = df.sort_values(subset='x', ascending=True)
        sns.lineplot(x='x', y='y', data=df)
        plt.show()
        plt.close()

    def disable_voters_predict(self, data, voter_lists):
        voting_masks_now = [np.ones_like(np.mat(mat)) for mat in self.model.voting_mask_one]
        for l_id, voter_lst in enumerate(voter_lists):
            for voter in voter_lst:
                voting_masks_now[l_id][voter, :] = 0
        return self.mask_score(data, voting_masks_now)


def read_data(data_name, model_name, data_split):
    data_path = eval("%s_DATASET_PATH" % data_name)
    meta = MetaData()
    meta.restore("%s/metadata.meta" % data_path)
    data = meta.read_pkl("%s/train.pkl" % data_path)
    data_test = meta.read_pkl("%s/test.pkl" % data_path)

    if data_split == 'all':
        for key, val in data.items():
            if val is None:
                continue
            if not isinstance(data[key], list): 
                data[key] = val.tolist()
                data_test[key] = data_test[key].tolist()
            data[key].extend(data_test[key])

    with open("%s/%s/%s/masks.pkl" % (MODEL_PATH, data_name, model_name), 'rb') as f:
        dct = pickle.load(f)
    voter_mask = dct['voter_mask']
    voting_mask = dct['voting_mask']

    with open("%s/%s/%s/statistics.pkl" % (MODEL_PATH, data_name, model_name), 'rb') as f:
        dct = pickle.load(f)
    nodes_avg = dct['nodes_avg']
    nodes_std = dct['nodes_std']
    links_avg = dct['links_avg']
    links_std = dct['links_std']
    model = get_model(data_name, model_name, meta, use='Explain', voting_mean=links_avg)
    data = data_test if data_split == 'test' else data
    return meta, model, data, voter_mask, voting_mask, nodes_avg, nodes_std, links_avg, links_std


def get_data_and_explainer(data_name, model_name, data_split="test"):
    meta, model, data, voter_mask, voting_mask, nodes_avg, nodes_std, links_avg, links_std = read_data(data_name, model_name, data_split=data_split)
    explainer = Explainer(meta, model, nodes_avg, nodes_std, links_avg, links_std)
    return data, explainer
