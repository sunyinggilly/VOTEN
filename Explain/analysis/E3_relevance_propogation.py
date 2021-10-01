import sys
sys.path.append('../')
sys.path.append('../../')

import os
from Explain.Explainer import get_data_and_explainer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from CONFIG import HOME_PATH
from math import fabs
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = '16' 
rcParams['figure.figsize'] = (10.0, 4.0)


def devide(score_up, split_map):
    # score_up: list: n_{d+1}
    # split_map: matrix: n_d, n_{d+1} 
    split_map = np.mat(split_map)
    score_up = np.mat(score_up)
    split_map = split_map / (np.sum(split_map, axis=0) + np.sign(split_map) * 1e-9)
    rel_in = split_map * score_up
    return rel_in.tolist()


def score_propogation(top_score, split_maps):
    # top_score: float
    # split_maps: list[matrix]
    score_now = [top_score]
    for split_map in split_maps[::-1]:
        score_now = devide(score_now, split_map)
    return score_now


def explain_global(data, explainer, label_id):
    links_std = explainer.links_std.copy()
    links_std[-1] = np.reshape(links_std[-1][:, label_id], [-1, 1])
    links_std = [np.square(u) for u in links_std]
    return score_propogation(top_score=1, split_maps=links_std)


def read_sample(data, sample_id):
    data_ret = {}
    for key, val in data.items():
        data_ret[key] = val[sample_id: sample_id + 1] if val is not None else None
    return data_ret


def explain_local(data, explainer, label_id, sample_id):
    data_sample = read_sample(data, sample_id)
    _, links_sample = explainer.model.mid_output_avg(data_sample, explainer.model.voting_mask_one)
    links_avg = explainer.links_avg
    split_maps = [u_sample - u_avg for u_sample, u_avg in zip(links_sample, links_avg)]
    split_maps[-1] = np.reshape(split_maps[-1][:, label_id], [-1, 1])
    return score_propogation(top_score=1, split_maps=split_maps)


def solve(data, explainer, sample_id, label_id, save_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if sample_id == 'global':
        rel_inputs = explain_global(data, explainer, label_id)
    else:
        rel_inputs = explain_local(data, explainer, label_id, sample_id)
    df = pd.DataFrame()
    df['relevance'] = [u[0] for u in rel_inputs]
    df['feat_id'] = range(df.shape[0])
    df['feat_name'] = df['feat_id'].apply(lambda x: explainer.get_feat_name(x))
    df = df.sort_values(by='relevance', ascending=False)
    df.to_csv("%s/relevance_%s_%d.csv" % (save_path, str(sample_id), label_id))
    return [u[0] for u in rel_inputs]


def scale(lst):
    v_max = max([fabs(u) for u in lst])
    return [u / v_max for u in lst]


def run_experiment(data_name, model_name, explain_samples, exp_name):
    save_path = "%s/Explain/out/%s/%s/relevance_propogation/%s" % (HOME_PATH, data_name, model_name, exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data, explainer = get_data_and_explainer(data_name, model_name)
    lst = []
    for sample_id, label_id in explain_samples:
        now_lst = solve(data, explainer, sample_id, label_id, save_path)
        now_lst = scale(now_lst)
        lst.append(now_lst)

    # print(lst)
    sns.heatmap(lst, cmap="vlag", center=0, yticklabels=['#%d_%d' % (sid, u[1]) for sid, u in enumerate(explain_samples)])
    plt.subplots_adjust(left=0.10, bottom=0.110, right=0.999, top=0.980)
    # plt.show()
    plt.savefig("%s/relevance_heatmap.pdf" % save_path)
    plt.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    # run_experiment(data_name="KDC", model_name="VOTERS", exp_name="local_1",
    #                explain_samples=[(6297, 1), (28373, 1), (3033, 2), (4194, 2), 
    #                                 (23768, 3), (43498, 3), (8575, 4), (49242, 4), 
    #                                 (28373, 5), (36315, 5)])

    # run_experiment(data_name="IJCAI18X", model_name="VOTERS", exp_name="local_1",
    #                explain_samples=[(41679, 0), (28690, 0), (16909, 0), (53765, 0), 
    #                                 (31652, 0), (6290, 0), (49422, 0), (31645, 0), 
    #                                 (1728, 0), (51923, 0)])

    # run_experiment(data_name="COVTYPE", model_name="VOTERS", exp_name="local_1",
    #                explain_samples=[(0, 0), (98979, 0), (116202, 1), (86111, 1), 
    #                                 (99493, 2), (40696, 2), (52706, 3), (45436, 3), 
    #                                 (80645, 4), (20523, 4)])

    run_experiment(data_name="CENSINCOME", model_name="VOTERS", exp_name="local_1",
                   explain_samples=[(59310, 0), (70700, 0), 
                                    (37779, 0), (55113, 0), (95891, 0), (85499, 0),
                                    (3983, 0), (23858, 0), (50493, 0), (78400, 0), (6347, 0)])

# [, 18682, 42479, 9571, 90322, 22077, 64123, , 76855, 6345, 59904, 8720, 3608, , 78471, 50156, 1016, , 23069, 23332, , 91549, 8782, , 1960, , 94627, 16056, 76729, , 74909, , 5582, , 2686, , 70285, 39224, 18537, 23912, , 98642, 77695, 28244, 6358, , 26511, 43101, 88550, , 57006, 87924, 89800, 1254, , 29073, 47953, 31972, 1326, 59541, 39194, 96148, 13305, 94580, 36671, 82639, 62359, 41827, 94986, 33035, 1133, 52300, 73170, 89957, 6352, 93502, 42249, 97093, 44444, 35862, 90873, 28964, 72378, 63168, 59562, 54112, 19398, 58086, 3823, 89204, 50318, 43073, 4579, 81805, 10956, 57764, 46337, 21330, 27070, 83083]