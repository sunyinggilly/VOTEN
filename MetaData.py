import tensorflow as tf
import pickle
import pandas as pd


class MetaData(object):
    def __init__(self, use_labels=None, all_cols=None, cont_cols=None, disc_cols=None, group_len_ctrl=None, group_len_feat=None,
                 group_ranges=None, group_len=None, n_dv=None, df_val=None, n_train=None, n_test=None, other_info=None):
        if use_labels is None:
            return
        self.all_cols = all_cols
        self.use_labels = use_labels
        self.n_label = len(use_labels)
        self.cont_cols, self.disc_cols = cont_cols, disc_cols
        self.other_info = other_info
        self.group_ranges, self.group_len = group_ranges, group_len

        self.group_len_ctrl, self.group_len_feat = group_len_ctrl, group_len_feat
        self.dim_cont = len(self.cont_cols)
        self.dim_disc = len(self.disc_cols)
        self.n_dv, self.df_val = n_dv, df_val
        self.n_train = n_train
        self.n_test = n_test

    def read_TFRecord(self, input_file):
        dataset = tf.data.TFRecordDataset(input_file)
        return dataset

    def save(self, out_file):
        dct = {"all_cols": self.all_cols, "use_labels": self.use_labels, "n_label": self.n_label, "cont_cols": self.cont_cols,
               "disc_cols": self.disc_cols, "other_info": self.other_info, "group_ranges": self.group_ranges,
               "group_len": self.group_len, "group_len_ctrl": self.group_len_ctrl, "group_len_feat": self.group_len_feat,
               "dim_cont": self.dim_cont, "dim_disc": self.dim_disc, "n_dv": self.n_dv, "df_val_col": self.df_val.columns.tolist(),
               "df_val": self.df_val.values, "n_train": self.n_train, "n_test": self.n_test}
        with open(out_file, "wb") as f:
            pickle.dump(dct, f)

    def restore(self, store_file):
        with open(store_file, "rb") as f:
            dct = pickle.load(f)
        self.all_cols = dct["all_cols"]
        self.use_labels = dct["use_labels"]
        self.n_label = dct["n_label"]
        self.cont_cols = dct["cont_cols"]
        self.disc_cols = dct["disc_cols"]
        self.other_info = dct["other_info"]
        self.group_ranges = dct["group_ranges"]
        self.group_len = dct["group_len"]
        self.group_len_ctrl = dct["group_len_ctrl"]
        self.group_len_feat = dct["group_len_feat"]
        self.dim_cont = dct["dim_cont"]
        self.dim_disc = dct["dim_disc"]
        self.n_dv = dct["n_dv"]
        self.df_val = pd.DataFrame(dct['df_val'], columns=dct['df_val_col'])
        self.n_train = dct['n_train']
        self.n_test = dct['n_test']
