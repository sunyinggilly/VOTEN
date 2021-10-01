import sys
sys.path.append('../')
sys.path.append('../../')

import os
from Explain.Explainer import get_data_and_explainer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from CONFIG import HOME_PATH, KDC_DATASET_PATH, IJCAI18X_DATASET_PATH
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from DataProcess.MetaData import MetaData
import tensorflow as tf
import pickle


def validate(y_true, y_pred):
    if len(y_pred[0]) > 1:
        auc = roc_auc_score(np.array(y_true), np.array(y_pred), average='macro')
    else:
        auc = roc_auc_score(np.array(y_true), np.array(y_pred))
    return auc


def change_voting(voting_mask, l_id, hf_id, ht_id='all', to=1):
    if ht_id == 'all':
        voting_mask[l_id][hf_id, :] = to
    else:
        voting_mask[l_id][hf_id, ht_id] = to


def get_abscent_score_voting(data, explainer, voting_mask, l_id, hf_id, ht_id='all'):
    change_voting(voting_mask, l_id, hf_id, ht_id, to=0)
    predictions = explainer.mask_score(data, voting_mask, return_feat=None)
    change_voting(voting_mask, l_id, hf_id, ht_id, to=1)
    return validate(data['label'], predictions)


def get_abscent_score_all(data, explainer, voting_mask, save_path):
    node_score, link_score = [], []
    data_save = []
    for l_id, n_in in enumerate([explainer.model.n_cont + explainer.model.n_disc] + explainer.model.deep_layers[:-1]):
        print(l_id)
        n_out = explainer.model.deep_layers[l_id]
        node_score.append([])
        link_score.append([])
        for hf_id in tqdm(range(n_in)):
            link_score[-1].append([])
            score = get_abscent_score_voting(data, explainer, voting_mask, l_id, hf_id, ht_id='all')
            node_score[-1].append(score)
            data_save.append((l_id, hf_id, 'all', score))
            for ht_id in range(n_out):
                score = get_abscent_score_voting(data, explainer, voting_mask, l_id, hf_id, ht_id)
                link_score[-1][-1].append(score)
                data_save.append((l_id, hf_id, ht_id, score))
    df = pd.DataFrame(data_save, columns=['layer', 'hidden_from', 'hidden_to', 'score'])
    df_node = df[df['hidden_to'] == 'all']
    df_link = df[df['hidden_to'] != 'all']
    df_node = df_node.sort_values(by='score', ascending=False)
    df_link = df_link.sort_values(by='score', ascending=False)
    df_node.to_csv("%s/node_score.csv" % save_path, index=False)
    df_link.to_csv("%s/link_score.csv" % save_path, index=False)
    return df_node, df_link


def draw_network(node_score, link_score, save_path):
    pass


def link_prune(data, explainer, voting_mask, step=0.0003):
    # 初始预测
    predictions = explainer.mask_score(data, voting_mask, return_feat=None)
    lst = [validate(data['label'], predictions)]
    del_vis = set(), set()
    for i in range(10):
        flag = True
        for l_id, hf_id, ht_id in tqdm(del_lst):
            

vis_set = set()
        

    vis_set = set()
    
def del_experiment(data, explainer, voting_mask, del_lst, save_path, prefix='link', thres=0):
    predictions = explainer.mask_score(data, voting_mask, return_feat=None)
    lst = [validate(data['label'], predictions)]
    vis_set = set()

    del_vis = set()
    flag = True
    while flag:
        vis_set = set()
        flag = False
        for l_id, hf_id, ht_id in tqdm(del_lst):
            if (l_id, hf_id, ht_id) in del_vis:
                continue
            if ht_id in vis_set and ht_id != 'all':
                continue
            vis_set.add(ht_id)
            del_vis.add((l_id, hf_id, ht_id))
            flag = True
            change_voting(voting_mask, l_id, hf_id, ht_id=ht_id, to=0)
            predictions = explainer.mask_score(data, voting_mask, return_feat=None)
            score = validate(data['label'], predictions)
            if score >= thres:
                lst.append(score)
            else:
                change_voting(voting_mask, l_id, hf_id, ht_id=ht_id, to=1)

    for l_id, hf_id, ht_id in del_lst:
        change_voting(voting_mask, l_id, hf_id, ht_id=ht_id, to=1)
    df = pd.DataFrame()
    df['score'] = lst
    df['id'] = range(df.shape[0])
    print(df)
    df.to_csv("%s/%s_prune_voting.csv" % (save_path, prefix), index=False)
    sns.lineplot(x="id", y="score", data=df)
    # plt.show()
    plt.savefig("%s/%s_prune_voting.pdf" % (save_path, prefix))
    plt.close()
    return


def solve(data_name, model_name, read_previous=False, del_thres=0):
    save_path = "%s/Explain/out/%s/%s/abscent_score" % (HOME_PATH, data_name, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data, explainer = get_data_and_explainer(data_name, model_name, split='test')
    voting_mask = [np.mat(u) for u in explainer.model.voting_mask_one]
    if not read_previous:
        df_node, df_link = get_abscent_score_all(data, explainer, voting_mask, save_path)
    else:
        df_node = pd.read_csv("%s/node_score.csv" % save_path)
        df_link = pd.read_csv("%s/link_score.csv" % save_path)

    df_link.to_csv("%s/link_score.csv" % save_path, index=False)
    del_lst = df_link[['layer', 'hidden_from', 'hidden_to']].values.tolist()
    del_experiment(data, explainer, voting_mask, del_lst, save_path, prefix='link', thres=del_thres)

    del_lst = df_node[['layer', 'hidden_from', 'hidden_to']].values.tolist()[:100]
    del_experiment(data, explainer, voting_mask, del_lst, save_path, prefix='node', thres=del_thres)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # data_name = "KDC"
    # # data_name = "IJCAI18X"
    # model_name = "VOTERS" 
    solve("KDC", "VOTERS", read_previous=True, del_thres=0.924)
