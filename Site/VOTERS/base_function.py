import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')

import json
from math import fabs
import numpy as np
from Explain.analysis.path import explain_sample
from Explain.analysis.path_global import explain_global


def ctx_prepare_local(ctx, data, explainer, model, layers, n_label, hiddens, color_gen):
    nodes, links, input_links = [], [], {}

    label_id = int(ctx['label_id']) if ctx['label_id'] != 'all' else None
    instance_id = ctx['eid']
    link_thres = ctx['link_thres']

    links_color, links_show, nodes_show = explain_sample(data, sample_id=instance_id, 
                                                         explainer=explainer, threshold=link_thres, 
                                                         label_id=label_id)

    # ------------------------------ nodes ---------------------------------------------------------
    y_range, x_range = 1000, 2000
    y_block = y_range / (len(hiddens) - 1)
    x_block = []

    for hidden_now in hiddens:
        if len(hidden_now) > 1:
            x_block.append(x_range / (len(hidden_now) - 1))
        else:
            x_block.append(x_range / 2)

    for d, nid, info in nodes_show:
        # print(d, nid)
        hidden_name = hiddens[d][nid]
        x = np.float(x_block[d] * nid) if len(hiddens[d]) > 1 else np.float(x_range / 2)
        y = y_range - np.float(y_block * d)
        val = np.float(info[0] - info[1])  # 偏移量
        color = np.float(val / fabs(info[1]))    # 偏移比

        dct = {'name': hidden_name, 'x': x, 'y': y, 'value': color, 'deviation': val, 
               'symbolSize': 20 if d > 0 else 20}
        nodes.append(dct)

    # ------------------------------- links ---------------------------------------------------------
    for feat_name in hiddens[0] + hiddens[1]:
        input_links[feat_name] = []

    for d, nfrom, nto, info in links_show:
        color_now = min(max(info[0] * 2, -1), 1)
        dct_linestyle = {'width': np.float(fabs(info[0])) * 30, 'color': color_gen.convert_to_hex(color_now)}  # 分数贡献
        dct = {'source': hiddens[d][nfrom], 'target': hiddens[d + 1][nto], 
               'lineStyle': dct_linestyle, 'contribution_ratio': np.float(info[0]),
               'average': np.float(info[1]), 'std': np.float(info[2])}
        if d == 0:
            input_links[hiddens[d][nfrom]].append(dct)
            input_links[hiddens[d + 1][nto]].append(dct)
            links.append(dct)
        else:
            links.append(dct)

    ctx['nodes'] = json.dumps(nodes)
    ctx['links'] = json.dumps(links)
    ctx['input_links'] = json.dumps(input_links)

    return ctx


def ctx_prepare_global(ctx, data, explainer, model, layers, n_label, hiddens, color_gen):
    nodes, links, input_links = [], [], {}

    label_id = int(ctx['label_id']) if ctx['label_id'] != 'all' else None
    link_std_thres = ctx['link_std_thres']
    link_val_ratio_thres = ctx['link_val_ratio_thres']

    links_color, links_show, nodes_show = explain_global(data, explainer=explainer, threshold=link_std_thres, 
                                                         threshold_ratio=link_val_ratio_thres, label_id=label_id)
    print(nodes_show)
    # ------------------------------ nodes ---------------------------------------------------------
    y_range, x_range = 1000, 2000
    y_block = y_range / (len(hiddens) - 1)
    x_block = []

    for hidden_now in hiddens:
        if len(hidden_now) > 1:
            x_block.append(x_range / (len(hidden_now) - 1))
        else:
            x_block.append(x_range / 2)

    for d, nid in nodes_show:
        hidden_name = hiddens[d][nid]
        x = np.float(x_block[d] * nid) if len(hiddens[d]) > 1 else np.float(x_range / 2)
        y = y_range - np.float(y_block * d)

        dct = {'name': hidden_name, 'x': x, 'y': y, 'value': 0, 
               'symbolSize': 20 if d > 0 else 20}
        nodes.append(dct)

    # ------------------------------- links ---------------------------------------------------------
    for feat_name in hiddens[0] + hiddens[1]:
        input_links[feat_name] = []

    for d, nfrom, nto, info in links_show:
        color_now = min(max(info * 2, -1), 1)
        dct_linestyle = {'width': np.float(fabs(info)) * 10, 'color': color_gen.convert_to_hex(color_now)}  # 分数贡献
        dct = {'source': hiddens[d][nfrom], 'target': hiddens[d + 1][nto], 
               'lineStyle': dct_linestyle, 'std': np.float(info)}
        if d == 0:
            input_links[hiddens[d][nfrom]].append(dct)
            input_links[hiddens[d + 1][nto]].append(dct)
            links.append(dct)
        else:
            links.append(dct)

    ctx['nodes'] = json.dumps(nodes)
    ctx['links'] = json.dumps(links)
    ctx['input_links'] = json.dumps(input_links)
    return ctx


def ctx_prepare_global_flexible(ctx, data, explainer, model, layers, n_label, hiddens, color_gen):
    nodes, links, input_links = [], [], {}

    label_id = int(ctx['label_id']) if ctx['label_id'] != 'all' else None
    link_std_thres = ctx['link_std_thres']
    link_val_ratio_thres = ctx['link_val_ratio_thres']

    links_color, links_show, nodes_show = explain_global(data, explainer=explainer, threshold=link_std_thres, 
                                                         threshold_ratio=link_val_ratio_thres, label_id=label_id)
    print(nodes_show)
    # ------------------------------ nodes ---------------------------------------------------------
    y_range, x_range = 1000, 500
    y_block = y_range / (len(hiddens) - 1)
    x_block = []

    # --------------------- 每层个数 ----------------
    node_lst = [[] for i in range(len(hiddens))]
    d_cnt = [0] * len(hiddens)

    for d, nid in nodes_show:
        node_lst[d].append(nid)
        d_cnt[d] += 1

    for u in node_lst:
        u.sort()

    for d, hidden_now in enumerate(hiddens):
        if d_cnt[d] > 1:
            x_block.append(x_range / (d_cnt[d] - 1))
        else:
            x_block.append(x_range / 2)

    for d, nid in nodes_show:
        hidden_name = hiddens[d][nid]
        locate = node_lst[d].index(nid)
        x = np.float(x_block[d] * locate) if d_cnt[d] > 1 else np.float(x_range / 2)
        y = y_range - np.float(y_block * d)

        dct = {'name': hidden_name, 'x': x, 'y': y, 'value': 0, 
               'symbolSize': 20 if d > 0 else 20}
        nodes.append(dct)

    # ------------------------------- links ---------------------------------------------------------
    for feat_name in hiddens[0] + hiddens[1]:
        input_links[feat_name] = []

    for d, nfrom, nto, info in links_show:
        color_now = min(max(info * 2, -1), 1)
        dct_linestyle = {'width': max(4, np.float(fabs(info)) * 10), 'color': color_gen.convert_to_hex(color_now)}  # 分数贡献
        dct = {'source': hiddens[d][nfrom], 'target': hiddens[d + 1][nto], 
               'lineStyle': dct_linestyle, 'std': np.float(info)}
        if d == 0:
            input_links[hiddens[d][nfrom]].append(dct)
            input_links[hiddens[d + 1][nto]].append(dct)
            links.append(dct)
        else:
            links.append(dct)

    ctx['nodes'] = json.dumps(nodes)
    ctx['links'] = json.dumps(links)
    ctx['input_links'] = json.dumps(input_links)
    return ctx