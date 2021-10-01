import sys
sys.path.append('../')
sys.path.append('../../')

import os
from Explain.Explainer import get_data_and_explainer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from CONFIG import HOME_PATH, KDC_DATASET_PATH, IJCAI18X_DATASET_PATH, COVTYPE_DATASET_PATH
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from DataProcess.MetaData import MetaData
import tensorflow as tf
import pickle
import matplotlib
matplotlib.use('Agg')

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


def update_cand(data, explainer, voting_mask, cand_lst):
    ret_score = []
    for l_id, hf_id, ht_id, _ in tqdm(cand_lst):
        score = get_abscent_score_voting(data, explainer, voting_mask, l_id, hf_id, ht_id)
        ret_score.append((l_id, hf_id, ht_id, score))
    return ret_score


def link_prune(data, explainer, cand_lst, l_step=0.0003, n_step=10, thres_init=0.9242, save_path=None, avg_line=0, timeout=None, kk=0):
    cand_lst = [(u[0], u[1], u[2], u[3], 0) for u in cand_lst]
    n_links = len(cand_lst)
    voting_mask = [np.mat(u) for u in explainer.model.voting_mask_one]
    # 初始预测
    predictions = explainer.mask_score(data, voting_mask, return_feat=None)
    lst = [(-1, -1, -1, validate(data['label'], predictions))]
    thres_now = thres_init
    thres_bottom = thres_init
    l_now = l_step 
    for i in range(n_step):
        thres_bottom = thres_bottom - l_now
        l_now += l_now * kk

    vis_set = set()
    update_counter = 0
    for i in range(n_step):
        flag = True
        while flag:
            vis_set = set()
            flag = False
            count_no_update = 0
            count_add = 0
            cand_lst_new = []
            for ele_id in tqdm(range(len(cand_lst))):
                u = cand_lst[ele_id]
                l_id, hf_id, ht_id, score, last_update = int(u[0]), int(u[1]), int(u[2]), u[3], u[4]
                if score < thres_bottom:
                    continue
                if (l_id, ht_id) in vis_set or score < thres_now:
                    cand_lst_new.append((l_id, hf_id, ht_id, score, last_update))
                    continue
                
                if update_counter > last_update:
                    change_voting(voting_mask, l_id, hf_id, ht_id=ht_id, to=0)
                    predictions = explainer.mask_score(data, voting_mask, return_feat=None)
                    score = validate(data['label'], predictions)
                    cand_lst[ele_id] = (l_id, hf_id, ht_id, score, update_counter)
                    count_no_update += 1
                if score >= thres_now:
                    vis_set.add((l_id, ht_id))
                    lst.append([l_id, hf_id, ht_id, score])
                    count_add += 1
                    update_counter += 1
                    flag = True
                    count_no_update = 0
                else:
                    change_voting(voting_mask, l_id, hf_id, ht_id=ht_id, to=1)
                    cand_lst_new.append((l_id, hf_id, ht_id, score, update_counter))
                
                if timeout is not None and count_no_update > timeout:
                    break
            print(thres_now, update_counter)
            cand_lst = cand_lst_new
            cand_lst.sort(key=lambda x: x[-1], reverse=True)
        thres_now -= l_step
        l_step += l_step * kk

    df = pd.DataFrame(lst, columns=['layer', 'from', 'to', 'score'])
    # df['score'] = lst
    df['id'] = range(df.shape[0])
    df['ratio'] = df['id'].apply(lambda x: x / n_links)

    df.to_csv("%s/link_prune_voting2.csv" % save_path, index=False)
    sns.lineplot(x="ratio", y="score", data=df)
    plt.axhline(y=avg_line, ls="-", c="green", lw=8)

    # plt.show()
    plt.savefig("%s/link_prune_voting2.pdf" % save_path)
    plt.close()
    return 


def solve(data_name, model_name, read_previous=False, l_step=0.0003, n_step=10, thres_init=0.9242, avg_line=0, timeout=None, kk=0):
    save_path = "%s/Explain/out/%s/%s/link_prune" % (HOME_PATH, data_name, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data, explainer = get_data_and_explainer(data_name, model_name)
    voting_mask = [np.mat(u) for u in explainer.model.voting_mask_one]
    if not read_previous:
        df_node, df_link = get_abscent_score_all(data, explainer, voting_mask, save_path)
    else:
        df_node = pd.read_csv("%s/node_score.csv" % save_path)
        df_link = pd.read_csv("%s/link_score.csv" % save_path)

    cand_lst = df_link[['layer', 'hidden_from', 'hidden_to', 'score']].values.tolist()
    link_prune(data, explainer, cand_lst, save_path=save_path, l_step=l_step, n_step=n_step, thres_init=thres_init, avg_line=avg_line, timeout=timeout, kk=kk)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    solve("KDC", "VOTERS", read_previous=True, l_step=0.0003, n_step=100, thres_init=0.9242, avg_line=0.9199)
    # solve("IJCAI18X", "VOTERS", read_previous=True, l_step=0.00005, n_step=40, thres_init=0.734)
    solve("COVTYPE", "VOTERS", read_previous=True, l_step=0.0001, n_step=20, thres_init=0.9983, avg_line=0.9965, timeout=1000, kk=0.05)
