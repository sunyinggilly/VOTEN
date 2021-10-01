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
from matplotlib import rcParams
from voting_function_draw import voting_function
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = '16' 


def solve(data_name, model_name):
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


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    solve("KDC", "VOTERS")
    solve("COVTYPE", "VOTERS")
    solve("IJCAI18X", "VOTERS")
