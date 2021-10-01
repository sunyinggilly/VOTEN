import sys
sys.path.append('../')
sys.path.append('../../')

import os
from Explain.Explainer import get_data_and_explainer
import numpy as np


def min_max_scale(lst):
    min_val = min(lst)
    max_val = max(lst)
    return [(u - min_val) / (max_val - min_val) for u in lst]


def find_influential(target_id, link_var_layer, threshold, threshold_ratio):
    start_nodes, links = [], []

    link_var_scale = min_max_scale(link_var_layer)

    link_val = [[i, u] for i, u in enumerate(link_var_scale)]
    link_val.sort(key=lambda x: x[1], reverse=True)

    for i in range(len(link_val)):
        link_val[i].append(link_val[i][1])
        if i > 0:
            link_val[i][2] += link_val[i - 1][2]   # id, link_val, 前缀和
    link_val = [(u[0], u[1], u[2] / link_val[-1][2]) for u in link_val]  # id, link_val, 前缀和比例

    for node, cont, u in link_val:
        if cont < link_val[0][1] * threshold_ratio:
            break
        start_nodes.append(node)
        links.append((node, target_id, cont))
        if u > threshold:
            break
    return links, start_nodes


def explain_global(data, explainer, threshold, threshold_ratio, label_id=None):
    links_std = explainer.links_std

    transform_depth = len(links_std)
    if label_id is None:
        Q = list(range(explainer.meta.n_label))
    else:
        Q = [label_id]

    links_show = []
    nodes_show = [(transform_depth, u) for u in Q]
    for d in range(transform_depth):
        depth_now = transform_depth - d - 1
        Q_new, links_new = [], []
        for u in Q:
            link_var_layer = np.square(links_std[- d - 1][:, u])
            links_use, start_nodes = find_influential(u, link_var_layer, threshold, threshold_ratio)
            links_new.extend(links_use)
            Q_new.extend(start_nodes)
        Q = list(set(Q_new))
        print(len(links_new), len(Q))
        nodes_show.extend([(depth_now, u) for u in Q])
        links_show.extend([(depth_now, a, b, c) for a, b, c in links_new])
    links_color = [u[3] for u in links_show]
    return links_color, links_show, nodes_show


if __name__ == '__main__':
    data_name = "KDC"
    model_name = "VOTERS" 
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    data, explainer = get_data_and_explainer(data_name, model_name)
    explain_global(data, explainer, threshold=0.6, threshold_ratio=0.1)
