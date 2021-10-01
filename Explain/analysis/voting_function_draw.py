import sys
sys.path.append('../')
sys.path.append('../../')
import seaborn as sns
from CONFIG import HOME_PATH
import os
from Explainer import get_data_and_explainer
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from random import shuffle


def draw(x, y, x_name, y_name, save_path, min_x=-1000000, max_x=1000000, pin_point=None, avg_draw=False):
    df = pd.DataFrame()
    df[x_name] = x
    df[y_name] = y
    df = df[(df[x_name] >= min_x) & (df[x_name] <= max_x)]
    # df['size'] = 10
    sns.scatterplot(data=df, x=x_name, y=y_name, s=150)
    # sns.lineplot(data=df, x=x_name, y=y_name)

    if pin_point is not None:
        plt.scatter(x=[u[0] for u in pin_point], y=[u[1] for u in pin_point], c='red', s=200)
    if pin_point is not None or avg_draw:
        avg_y = df[y_name].mean()
        plt.axhline(y=avg_y, ls="-", c="green", lw=8)
    plt.subplots_adjust(left=0.130, bottom=0.120, right=0.980, top=0.970)
    # plt.show()
    plt.savefig("%s/%s_%s.png" % (save_path, x_name, y_name))
    plt.close()


def voting_function(explainer, data, mid_l, mid_f, mid_t, voting_mask, save_path, pin_point=None, avg_draw=False):
    concept_val, voting_val = explainer.model.get_voting_input_and_link(data, mid_l, mid_f, mid_t, voting_mask)    
    draw(concept_val, voting_val, save_path=save_path, pin_point=pin_point, avg_draw=avg_draw, 
         x_name=explainer.get_node_name(mid_l, mid_f), y_name=explainer.get_node_name(mid_l + 1, mid_t))


if __name__ == '__main__':
    data_name = "KDC"
    model_name = "VOTERS"
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    save_path = "%s/Explain/out/%s/%s/voting_functions" % (HOME_PATH, data_name, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data, explainer = get_data_and_explainer(data_name, model_name)

    hids = [explainer.model.n_disc + explainer.model.n_cont] + explainer.model.deep_layers
    for d in range(len(hids) - 1):
        hid_id = list(range(hids[d + 1]))
        for feat_id in tqdm(range(hids[d])):
            shuffle(hid_id)
            for k in hid_id[:3]:
                voting_function(explainer, data, d, feat_id, k, explainer.model.voting_mask_one, save_path)
