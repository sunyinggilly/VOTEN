import sys
sys.path.append('../')
sys.path.append('../../')

import os
from Explain.Explainer import get_data_and_explainer
import pandas as pd


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
    data_name = "CENSINCOME"
    model_name = "VOTERS" 
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    data, explainer = get_data_and_explainer(data_name, model_name)
    print(explainer.meta.n_disc, explainer.meta.n_cont)
    sample_select(data, explainer.model.voting_mask_one, label_id=0)

    # links_color, links_show, nodes_show = explain_sample(data, 6297, explainer, 0.6, label_id=0)  # 6297KDC
    # print(len(links_show), len(nodes_show))
