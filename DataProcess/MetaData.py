import tensorflow as tf
import pickle
import pandas as pd


class MetaData(object):
    def __init__(self, label_cols=None, feat_cols_ordered=None, cont_cols=None, disc_cols=None,
                 group_len=None, group_ranges=None, n_dv=None, df_val=None, feat_groups=None,
                 n_train=None, n_test=None, other_info=None):

        if label_cols is None:
            return
        self.feat_cols_ordered = feat_cols_ordered
        self.label_cols = label_cols
        self.n_label = len(label_cols)
        self.cont_cols, self.disc_cols = cont_cols, disc_cols
        self.other_info = other_info
        self.group_ranges, self.group_len = group_ranges, group_len

        self.n_cont = len(self.cont_cols)
        self.n_disc = len(self.disc_cols)
        self.n_dv, self.df_val = n_dv, df_val
        self.n_train = n_train
        self.n_test = n_test

    def read_TFRecord(self, input_file):
        dataset = tf.data.TFRecordDataset(input_file)
        return dataset

    def read_csv(self, input_file):
        df = pd.read_csv(input_file)
        return df['instance_id'], df[self.feat_cols_ordered], df[self.label_cols]

    def read_pkl(self, input_file):
        with open(input_file, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    def save(self, out_file):
        dct = {"feat_cols_ordered": self.feat_cols_ordered, "label_cols": self.label_cols, 
               "n_label": self.n_label, "cont_cols": self.cont_cols,
               "disc_cols": self.disc_cols, "other_info": self.other_info,
               "n_cont": self.n_cont, "n_disc": self.n_disc, "n_dv": self.n_dv, "df_val_col": self.df_val.columns.tolist(),
               "df_val": self.df_val.values, "n_train": self.n_train, "n_test": self.n_test}
        with open(out_file, "wb") as f:
            pickle.dump(dct, f)

    def restore(self, store_file):
        with open(store_file, "rb") as f:
            dct = pickle.load(f)

        self.feat_cols_ordered = dct["feat_cols_ordered"]
        self.label_cols = dct["label_cols"]
        self.n_label = dct["n_label"]
        self.cont_cols = dct["cont_cols"]
        self.disc_cols = dct["disc_cols"]
        self.other_info = dct["other_info"]
        self.n_cont = dct["n_cont"]
        self.n_disc = dct["n_disc"]
        self.n_dv = dct["n_dv"]
        self.df_val = pd.DataFrame(dct['df_val'], columns=dct['df_val_col'])
        self.n_train = dct['n_train']
        self.n_test = dct['n_test']
