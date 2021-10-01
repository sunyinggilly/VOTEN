import sys
sys.path.append('../')
sys.path.append('../../')

import os
from Explain.Explainer import get_data_and_explainer
import pandas as pd
from math import fabs
import numpy as np


def get_color(links):
    vals = [u[3][0] for u in links]
    print(vals)
    max_abs = max([fabs(u) for u in vals])
    vals = [u / max_abs for u in vals]
    return vals


def find_influential(target_id, link_now_layer, link_avg_layer, link_std_layer, threshold, threshold_ratio=0.01):
    start_nodes = []
    links = []
    link_bias = link_now_layer - link_avg_layer
    # link_sum = sum(link_bias)
    # link_deviation_ratio = [u / link_sum for u in link_bias]  # 偏移比例
    link_contribution = np.fabs(link_bias)
    cont_sum = sum(link_contribution)
    link_contribution = link_contribution / cont_sum
    link_contribution = [[i, u] for i, u in enumerate(link_contribution)]
    link_contribution.sort(key=lambda x: x[1], reverse=True)

    for i in range(len(link_contribution)):
        link_contribution[i].append(link_contribution[i][1])
        if i > 0:
            link_contribution[i][2] += link_contribution[i - 1][2]

    for node, cont, u in link_contribution:
        start_nodes.append(node)
        links.append((node, target_id, (cont * link_bias[node] / fabs(link_bias[node]), 
                      link_now_layer[node], link_avg_layer[node], link_std_layer[node])))
        if u > threshold or cont < link_contribution[0][1] * threshold_ratio:
            break
    return links, start_nodes


def read_sample(data, sample_id):
    data_ret = {}
    for key, val in data.items():
        data_ret[key] = val[sample_id: sample_id + 1] if val is not None else None
    print(data_ret)
    return data_ret


def explain_sample(data, sample_id, explainer, threshold, label_id=None, threshold_ratio=0.01):
    links_avg = explainer.links_avg
    links_std = explainer.links_std
    nodes_avg = explainer.nodes_avg
    nodes_std = explainer.nodes_std

    data_sample = read_sample(data, sample_id)
    nodes_sample, links_sample = explainer.model.mid_output_avg(data_sample, explainer.model.voting_mask_one)

    transform_depth = len(links_avg)
    if label_id is None:
        Q = list(range(explainer.meta.n_label))
    else:
        Q = [label_id]
    links_show = []

    nodes_show = [(transform_depth, u, (nodes_sample[transform_depth][u], nodes_avg[transform_depth][u], nodes_std[transform_depth][u])) for u in Q]
    for d in range(transform_depth):
        depth_now = transform_depth - d - 1
        Q_new, links_new = [], []
        for u in Q:
            link_now_layer = links_sample[- d - 1][:, u]
            link_avg_layer = links_avg[- d - 1][:, u]
            link_std_layer = links_std[- d - 1][:, u]
            links_use, start_nodes = find_influential(u, link_now_layer, link_avg_layer, link_std_layer, threshold, threshold_ratio=threshold_ratio)
            links_new.extend(links_use)
            Q_new.extend(start_nodes)
        Q = list(set(Q_new))
        print(len(links_new), len(Q))
        nodes_show.extend([(depth_now, u, (nodes_sample[depth_now][u], nodes_avg[depth_now][u], nodes_std[depth_now][u])) for u in Q])
        links_show.extend([(depth_now, a, b, c) for a, b, c in links_new])
    links_color = get_color(links_show)
    return links_color, links_show, nodes_show


def sample_select(data, voting_mask, label_id=0):
    df = pd.DataFrame()
    labels = [u[label_id] for u in data['label']]
    predictions = explainer.mask_score(data, voting_mask, return_feat=None)
    df['label'] = labels
    df['prediction'] = [u[label_id] for u in predictions]
    df['id'] = range(df.shape[0])
    df = df.sort_values(by='prediction', ascending=False)
    # df = df.sort_values(by='prediction', ascending=True)
    print(df.head(100)['id'].tolist())


if __name__ == '__main__':
    data_name = "IJCAI18X"
    model_name = "VOTERS" 
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    data, explainer = get_data_and_explainer(data_name, model_name)
    sample_select(data, explainer.model.voting_mask_one, label_id=0)

    # links_color, links_show, nodes_show = explain_sample(data, 6297, explainer, 0.6, label_id=0)  # 6297KDC
    # print(len(links_show), len(nodes_show))
