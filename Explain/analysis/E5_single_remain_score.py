import sys
sys.path.append('../')
sys.path.append('../../')
import pandas as pd
import os
from Explainer import get_data_and_explainer
from sklearn.metrics import roc_auc_score  # log_loss
import numpy as np
from CONFIG import HOME_PATH
import scipy as sc
from voting_function_draw import draw
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = '16' 


def change_voting(voting_mask, l_id, hf_id, ht_id='all', to=1):
    if hf_id == 'all':
        if ht_id == 'all':
            voting_mask[l_id][:, :] = to
        else:
            voting_mask[l_id][:, ht_id] = to
    elif ht_id == 'all':
        voting_mask[l_id][hf_id, :] = to
    else:
        voting_mask[l_id][hf_id, ht_id] = to


def validate(y_true, y_pred):
    auc = roc_auc_score(np.array(y_true), np.array(y_pred))
    pearson = sc.stats.pearsonr(np.array(y_true), np.array(y_pred))
    return auc, pearson[0]


def single_predict(data, explainer, voting_mask_zero, l_id, h_id):
    change_voting(voting_mask_zero, l_id, h_id, 'all', to=1)
    predictions = explainer.mask_score(data, voting_mask_zero, return_feat=None, score_type='logit')
    change_voting(voting_mask_zero, l_id, h_id, 'all', to=0)
    return predictions


def get_concept(data, explainer):
    return explainer.model.mid_node_raw(data)


def calculate(data, explainer, label_id, save_path):
    rcParams['figure.figsize'] = (10.0, 8.0)
    concepts = get_concept(data, explainer)
    voting_mask = [np.mat(u) for u in explainer.model.voting_mask_one]
    labels = [u[label_id] for u in data['label']]
    data_save = []
    for l_id, n_hidden in enumerate([explainer.model.n_cont + explainer.model.n_disc] + explainer.model.deep_layers[:-1]):
        change_voting(voting_mask, l_id, 'all', 'all', to=0)
        for h_id in tqdm(range(n_hidden)):
            feat_name = explainer.get_node_name(l_id, h_id)
            predictions = single_predict(data, explainer, voting_mask, l_id, h_id)
            predictions = [u[label_id] for u in predictions]
            input_x = [u[0] for u in concepts[l_id][:, h_id].tolist()]
            draw(input_x, predictions, feat_name, "O%d" % label_id, save_path, min_x=-1000000, max_x=1000000, pin_point=None)
            auc_y, pearson_y = validate(labels, predictions)
            auc_y_neg, pearson_y_neg = validate(labels, [-u for u in predictions])
            auc_x, pearson_x = validate(labels, input_x)
            auc_x_neg, pearson_x_neg = validate(labels, [-u for u in input_x])

            data_save.append((l_id, h_id, feat_name, 'corr_input', auc_x, pearson_x, auc_x_neg, pearson_x_neg))
            data_save.append((l_id, h_id, feat_name, 'corr_predict', auc_y, pearson_y, auc_y_neg, pearson_y_neg))

        change_voting(voting_mask, l_id, 'all', 'all', to=1)

    df = pd.DataFrame(data_save, columns=['layer', 'hidden', 'feat_name', 'type', 'auc', 'pearson', 'auc_neg', 'pearson_neg'])
    df.to_csv("%s/relevance_%d.csv" % (save_path, label_id), index=False)
    df = df[df['type'] == 'corr_predict']
    return df


def solve(data_name, model_name, label_id, read_previous=False):
    save_path = "%s/Explain/out/%s/%s/single_remain_score" % (HOME_PATH, data_name, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not read_previous:
        data, explainer = get_data_and_explainer(data_name, model_name, data_split='train')
        df = calculate(data, explainer, label_id, save_path)

    df = pd.read_csv("%s/relevance_%d.csv" % (save_path, label_id))

    # 关联性rank及heatmap
    df_ana = df[df['layer'] == 0]
    df_ana['auc'] = df_ana[['auc', 'auc_neg']].apply(lambda x: max(x[0], x[1]), axis=1)
    df_ana['pearson'] = df_ana[['pearson', 'pearson_neg']].apply(lambda x: max(x[0], x[1]), axis=1)

    df_rank = df_ana[df_ana['type'] == 'corr_predict']
    df_rank = df_rank.sort_values(by='auc', ascending=False)
    df_rank.to_csv("%s/auc_rank_%d.csv" % (save_path, label_id))

    df_ana = df_ana[['hidden', 'feat_name', 'type', 'auc', 'pearson']]
    df_ana['type'] = df_ana['type'].apply(lambda x: 'Input' if x == 'corr_input' else 'Vote')
    df_ana_input = df_ana[df_ana['type'] == 'Input']
    df_ana_vote = df_ana[df_ana['type'] == 'Vote']

    df_ana_input = df_ana_input.sort_values(by='hidden', ascending=True)
    df_ana_vote = df_ana_vote.sort_values(by='hidden', ascending=True)

    rcParams['figure.figsize'] = (10.0, 2.0)
    data_auc = [df_ana_input['auc'].tolist(), df_ana_vote['auc'].tolist()]    
    sns.heatmap(data=data_auc, cmap="vlag", center=0.5, yticklabels=['Input', 'Vote'])
    plt.subplots_adjust(left=0.04, bottom=0.250, right=1.0, top=0.980)
    # plt.show()
    plt.savefig('%s/heatmap_auc_%d.pdf' % (save_path, label_id))
    plt.close()

    data_pearson = [df_ana_input['pearson'].tolist(), df_ana_vote['pearson'].tolist()]    
    sns.heatmap(data=data_pearson, cmap="vlag", center=0, yticklabels=['Input', 'Vote'])
    plt.subplots_adjust(left=0.04, bottom=0.250, right=1.0, top=0.980)
    # plt.show()
    plt.savefig('%s/heatmap_pearson_%d.pdf' % (save_path, label_id))
    plt.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    # solve("KDC", "VOTERS", label_id=0, read_previous=False)
    solve("KDC", "VOTERS", label_id=1, read_previous=False)
    # solve("IJCAI18X", "VOTERS", label_id=0, read_previous=False)
    # solve("COVTYPE", "VOTERS", label_id=0, read_previous=False)
    # solve("CENSINCOME", "VOTERS", label_id=0, read_previous=False)
