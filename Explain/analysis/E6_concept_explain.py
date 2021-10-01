import sys
sys.path.append('../')
sys.path.append('../../')
import pandas as pd
import os
from Explainer import get_data_and_explainer
from sklearn.svm import SVC
import numpy as np
from CONFIG import HOME_PATH
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import fabs


def represent_sample(data, explainer, feats, l_id, h_id, n_top=100):
    concept = explainer.model.get_concept(data, l_id, h_id, voting_mask=None)
    df = pd.DataFrame(feats, columns=explainer.feat_name)
    df['id'] = range(df.shape[0])
    df['concept_value'] = concept
    df = df.sort_values(by='concept_value', ascending=True)
    return df.head(n_top), df.tail(n_top), df


def sperate(dfp, dfn):
    X_pos = dfp.drop(['id', 'concept_value', 'group'], axis=1).values
    X_neg = dfn.drop(['id', 'concept_value', 'group'], axis=1).values
    X = np.concatenate((X_neg, X_pos), axis=0)
    Y = [0] * X_neg.shape[0] + [1] * X_pos.shape[0]
    clf = SVC(kernel='linear')  # LogisticRegression(random_state=0).fit(X_train, Y_train)
    clf.fit(X, Y)
    return clf.coef_


def explain(data, explainer, feats, l_id, h_id, n_top, filepath):
    if not os.path.exists("%s/%d_%d" % (filepath, l_id, h_id)):
        os.makedirs("%s/%d_%d" % (filepath, l_id, h_id))

    df_neg, df_pos, df_all = represent_sample(data, explainer, feats, l_id, h_id, n_top)

    # ----------------------------- feature distribution --------------------------
    df_neg['group'] = 'negative'
    df_pos['group'] = 'positive'
    df_all['group'] = 'all'
    df_show = df_all.append(df_neg)
    df_show = df_show.append(df_pos)
    print(df_show['cont@mode2'].drop_duplicates())
    for u, feat_name in tqdm(enumerate(explainer.feat_name)):
        sns.boxplot(x="group", y=feat_name, data=df_show, palette="Set3")
        plt.savefig("%s/%d_%d/%s.pdf" % (filepath, l_id, h_id, feat_name))
        plt.close()

    # ------------------------------ train linear model ---------------------------
    pos_neg_rel = sperate(df_pos, df_neg)
    sns.heatmap(data=pos_neg_rel, cmap="vlag", center=0)
    plt.savefig("%s/%d_%d/pos_neg_rel_heatmap.pdf" % (filepath, l_id, h_id))
    plt.close()
    neg_all_rel = sperate(df_neg, df_all)
    sns.heatmap(data=neg_all_rel, cmap="vlag", center=0)
    plt.savefig("%s/%d_%d/neg_all_rel_heatmap.pdf" % (filepath, l_id, h_id))
    plt.close()
    pos_all_rel = sperate(df_pos, df_all)
    sns.heatmap(data=pos_all_rel, cmap="vlag", center=0)
    plt.savefig("%s/%d_%d/pos_all_rel_heatmap.pdf" % (filepath, l_id, h_id))
    plt.close()

    pos_neg_rel_name = [(explainer.feat_name[u], val) for u, val in enumerate(pos_neg_rel[0].tolist())]
    neg_all_rel_name = [(explainer.feat_name[u], val) for u, val in enumerate(neg_all_rel[0].tolist())]
    pos_all_rel_name = [(explainer.feat_name[u], val) for u, val in enumerate(pos_all_rel[0].tolist())]
    pos_neg_rel_name.sort(key=lambda x: fabs(x[1]), reverse=True)
    neg_all_rel_name.sort(key=lambda x: fabs(x[1]), reverse=True)
    pos_all_rel_name.sort(key=lambda x: fabs(x[1]), reverse=True)
    print('pos_neg_rel_name', pos_neg_rel_name)
    print('neg_all_rel_name', neg_all_rel_name)
    print('pos_all_rel_name', pos_all_rel_name)

    return pos_neg_rel, neg_all_rel, pos_all_rel


def solve(data_name, model_name, concepts=[]):
    save_path = "%s/Explain/out/%s/%s/concept_explain" % (HOME_PATH, data_name, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data, explainer = get_data_and_explainer(data_name, model_name, data_split='train')
    feats = explainer.get_feature_value_all(data)
    for l_id, h_id in concepts:
        explain(data, explainer, feats, l_id, h_id, n_top=100, filepath=save_path)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    solve("KDC", "VOTERS", [(1, 4), (1, 8)])
