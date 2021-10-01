import sys
sys.path.append('../')
sys.path.append('../../')
import pandas as pd
from sklearn.metrics import f1_score
import json
import numpy as np
from datetime import datetime
from CONFIG import KDC_DATASET_PATH
from DataProcessTools import data_process


def read_data(path):
    df_queries = pd.read_csv(path + 'train_queries.csv').astype(str)
    df_plans = pd.read_csv(path + 'train_plans.csv').astype(str)
    df_clicks = pd.read_csv(path + 'train_clicks.csv').astype(str)
    df_profiles = pd.read_csv(path + 'profiles.csv').astype(str)
    for i in range(66):
        df_profiles['p' + str(i)] = df_profiles['p' + str(i)].apply(lambda x: int(float(x)))
    df_profiles['has_profile'] = 1
    return df_profiles, df_queries, df_plans, df_clicks

# 原版，有问题的，mode1被覆盖了，sid对应mode1, mode1对应distance1……也就是删掉
# cols = ['sid', 'mode1', 'distance1', 'eta1', 'price1', 'mode2', 'distance2', 'eta2', 'price2', 'mode3', 'distance3',
#         'eta3', 'price3',
#         'mode4', 'distance4', 'eta4', 'price4', 'mode5', 'distance5', 'eta5', 'price5', 'mode6', 'distance6',
#         'eta6', 'price6',
#         'mode7', 'distance7', 'eta7', 'price7', 'mode8', 'distance8', 'eta8', 'price8', 'mode9', 'distance9',
#         'eta9', 'price9',
#         'mode10', 'distance10', 'eta10', 'price10', 'mode11', 'distance11', 'eta11', 'price11']

# 实际应该删掉sid
cols = ['mode1', 'distance1', 'eta1', 'price1', 'mode2', 'distance2', 'eta2', 'price2', 'mode3', 'distance3',
        'eta3', 'price3',
        'mode4', 'distance4', 'eta4', 'price4', 'mode5', 'distance5', 'eta5', 'price5', 'mode6', 'distance6',
        'eta6', 'price6',
        'mode7', 'distance7', 'eta7', 'price7', 'mode8', 'distance8', 'eta8', 'price8', 'mode9', 'distance9',
        'eta9', 'price9',
        'mode10', 'distance10', 'eta10', 'price10', 'mode11', 'distance11', 'eta11', 'price11']


def extrac_plans_feature(df):
    plan_lst = df['plans'].tolist()
    ret_lst = []
    for plan in plan_lst:
        dct_lst = json.loads(plan)
        ret = []
        for dct in dct_lst:
            ret.extend([int(dct["transport_mode"]), int(dct["distance"]), int(dct["eta"]), int(dct["price"]) if dct["price"] != '' else 0])
        for u in range(len(ret), len(cols)):
            ret.append(np.nan)
        ret_lst.append(ret)
    df_ret = pd.DataFrame(ret_lst, columns=cols)
    df_ret['sid'] = df['sid']
    df = pd.merge(df, df_ret, on='sid')
    df = df.drop('mode1')
    return df


def extrac_query_feature(df, df_profiles):
    df['hour'] = df['req_time'].apply(lambda x: int(x.split(' ')[1].split(":")[0]))
    df['weekday'] = df['req_time'].apply(lambda x: datetime.strptime(x.split(' ')[0], "%Y-%m-%d").weekday())
    df['pid'] = df['pid'].apply(lambda x: str(int(float(x))) if x != 'nan' else 'nan')
    df = pd.merge(df, df_profiles, on='pid', how='left')
    df[['p%d' % i for i in range(66)]] = df[['p%d' % i for i in range(66)]].fillna(-1)
    return df


def filter_vali(x):
    date = x.split(' ')[0].split('-')
    return int(date[1]) == 11 and int(date[2]) >= 24


def validate(df):
    y_true = df['click_mode'].tolist()
    y_pred = df['recommend_mode'].tolist()
    return f1_score(y_true, y_pred, average='weighted')


def process_P1():
    df_profiles, df_queries, df_plans, df_clicks = read_data("/home/suny/VOTERS/Data/Downloads/KDDCUP19P1/")
    vali_list = df_queries[df_queries['req_time'].apply(filter_vali)]['sid'].tolist()
    df_queries = extrac_query_feature(df_queries, df_profiles)
    df_plans = extrac_plans_feature(df_plans)

    df_queries_vali = df_queries[df_queries['sid'].isin(vali_list)]
    df_plans_vali = df_plans[df_plans['sid'].isin(vali_list)]
    df_clicks_vali = df_clicks[df_clicks['sid'].isin(vali_list)]

    df_queries_train = df_queries[~df_queries['sid'].isin(vali_list)]
    df_plans_train = df_plans[~df_plans['sid'].isin(vali_list)]
    df_clicks_train = df_clicks[~df_clicks['sid'].isin(vali_list)]

    columns = ('sid', 'hour', 'weekday', 'has_profile') + tuple(['p' + str(i) for i in range(66)]) + tuple(cols[1:]) + tuple(['click_mode'])
    df_train = pd.merge(df_plans_train, df_queries_train, on='sid', how='left')
    df_train = pd.merge(df_train, df_clicks_train, on='sid', how='inner')[list(columns)]
    df_train['click_mode'] = df_train['click_mode'].fillna('0').astype(float)
    df_vali = pd.merge(df_plans_vali, df_queries_vali, on='sid', how='left')
    df_vali = pd.merge(df_vali, df_clicks_vali, on='sid', how='inner')[list(columns)]
    df_vali['click_mode'] = df_vali['click_mode'].fillna('0').astype(float)
    df_train = df_train.fillna(0)
    df_vali = df_vali.fillna(0)
    for i in range(11):
        df_train['label_%d' % i] = df_train['click_mode'].apply(lambda x: 1 if x == i + 1 else 0)
        df_vali['label_%d' % i] = df_vali['click_mode'].apply(lambda x: 1 if x == i + 1 else 0)
    df_train = df_train.drop('click_mode', axis=1)
    df_vali = df_vali.drop('click_mode', axis=1)

    return df_train, df_vali


if __name__ == '__main__':
    disc_feats = ['hour', 'weekday', 'has_profile'] + ['p%d' % i for i in range(66)]
    cont_feats = ['distance1', 'eta1', 'price1', 'mode2', 'distance2', 'eta2', 'price2', 'mode3', 'distance3', 'eta3', 'price3',
                  'mode4', 'distance4', 'eta4', 'price4', 'mode5', 'distance5', 'eta5', 'price5', 'mode6', 'distance6', 'eta6',
                  'price6', 'mode7', 'distance7', 'eta7', 'price7', 'mode8', 'distance8', 'eta8', 'price8', 'mode9', 'distance9',
                  'eta9', 'price9', 'mode10', 'distance10', 'eta10', 'price10', 'mode11', 'distance11', 'eta11', 'price11']

    df_train, df_test = process_P1()

    data_process(df_train, df_test, 'sid', ['label_%d' % i for i in range(11)], disc_cols=disc_feats, cont_cols=cont_feats, other_info={},
                 skip_norm_cols=[], max_disc_val=100000, drop_disc_thres=1000000, outpath=KDC_DATASET_PATH)
