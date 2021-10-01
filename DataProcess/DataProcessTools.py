import sys
sys.path.append("../")

import pickle

import tensorflow as tf
from tqdm import tqdm
import numpy as np
from DataProcess.MetaData import MetaData
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def label_encode_join(disc_cols, df_data, max_val=50, drop_thres=200):
    vals = []
    for feat in disc_cols:
        df_data[feat] = df_data[feat].fillna('NaN')
        cnt_series = df_data[feat].value_counts()
        if cnt_series.shape[0] > max_val:
            cnt_series = cnt_series[cnt_series >= drop_thres]
        val_lst = set(list(cnt_series.index))

        df_data[feat] = df_data[feat].apply(lambda x: feat + '#' + str(x) if x in val_lst else feat + '#others')
        vals.extend(list(set(df_data[feat].tolist())))

    le = LabelEncoder()
    le.fit(vals)
    val_ids = le.transform(vals)
    lst = []
    for val, val_id in zip(vals, val_ids):
        lst.append((val, val_id))
    df_val = pd.DataFrame(lst, columns=['value', 'index'])

    for feat in disc_cols:
        df_data[feat] = le.transform(df_data[feat].tolist())

    return df_data, df_val['index'].max() + 1, df_val


def normalization(df, cols, skip_norm_cols=[]):
    for col in tqdm(cols):
        mm = df[col].mean()
        ss = StandardScaler()
        df[col] = df[col].fillna(mm)
        if col not in skip_norm_cols:
            df[[col]] = ss.fit_transform(df[[col]])
    return df


def input_gen(df, instance_col, label_cols, disc_cols, cont_cols):
    data_out = {}
    data_out['instance_id'] = df[[instance_col]].values.tolist()
    data_out['label'] = df[label_cols].values.tolist()
    data_out['continuous_feats'] = df[cont_cols].values.tolist()
    data_out['discrete_feats'] = df[disc_cols].values.tolist()
    return data_out


def data_process(df_train, df_test, instance_col, label_cols, disc_cols, cont_cols, other_info, skip_norm_cols, max_disc_val, drop_disc_thres, outpath):
    print(df_train['mode1'].drop_duplicates())
    # ----------------- label 列 -------------------------------
    label_cols = label_cols

    # ----------------- 特征处理 ------------------------------
    df_train['train'] = 'train'
    df_test['train'] = 'test'
    df_join = df_train.append(df_test)
    feat_rename_dct = {}
    feat_cols_ordered = []
    for feat in disc_cols:
        feat_rename_dct[feat] = 'disc@%s' % feat
        feat_cols_ordered.append('disc@%s' % feat)
    for feat in cont_cols:
        feat_rename_dct[feat] = 'cont@%s' % feat
        feat_cols_ordered.append('cont@%s' % feat)

    disc_cols = [feat_rename_dct[u] for u in disc_cols]
    cont_cols = [feat_rename_dct[u] for u in cont_cols]
    skip_norm_cols = [feat_rename_dct[u] for u in skip_norm_cols]

    df_join.rename(columns=feat_rename_dct, inplace=True)
    df_join = df_join[[instance_col] + label_cols + feat_cols_ordered + ['train']]
    print(df_join['cont@mode1'].drop_duplicates())
    df_join = normalization(df_join, cont_cols, skip_norm_cols)
    print(df_join['cont@mode1'].drop_duplicates())
    df_join, n_dv, df_val = label_encode_join(disc_cols, df_join, max_val=max_disc_val, drop_thres=drop_disc_thres)

    df_train = df_join[df_join['train'] == 'train']
    df_train = df_train.drop('train', axis=1)
    df_test = df_join[df_join['train'] == 'test']
    df_test = df_test.drop('train', axis=1)

    # ------------------------ generate inputs ---------------------------
    df_train.to_csv("%s/train.csv" % outpath, index=False)
    df_test.to_csv("%s/test.csv" % outpath, index=False)
    data_train = input_gen(df_train, instance_col, label_cols, disc_cols, cont_cols)
    data_test = input_gen(df_test, instance_col, label_cols, disc_cols, cont_cols)

    with open("%s/train.pkl" % outpath, 'wb') as f:
        pickle.dump(data_train, f)
    with open("%s/test.pkl" % outpath, 'wb') as f:
        pickle.dump(data_test, f)

    data_to_TFRecord_file(data_train, outpath + '/train.tf_record')
    data_to_TFRecord_file(data_test, outpath + '/test.tf_record')
    n_train = len(data_train['label'])
    n_test = len(data_test['label'])

    metadata = MetaData(label_cols=label_cols, feat_cols_ordered=feat_cols_ordered, 
                        cont_cols=cont_cols, disc_cols=disc_cols, 
                        n_dv=n_dv, df_val=df_val, n_train=n_train, n_test=n_test, other_info=other_info)
    metadata.save(outpath + '/metadata.meta')


def data_to_TFRecord_file(data, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    n_data = len(data['label'])
    for ex_index in tqdm(range(n_data)):
        features = {}
        for group, val in data.items():
            if val is None:
                continue
            features[group] = _int64_feature(val[ex_index]) if group.find('disc') != -1 else _float32_feature(val[ex_index])
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(value).astype(np.int64)))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=np.array(value).astype(float)))
