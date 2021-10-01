import sys
sys.path.append('../')
sys.path.append('../../')
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import json
import numpy as np
from datetime import datetime
from CONFIG import COVTYPE_DATASET_PATH, HOME_PATH
from DataProcessTools import data_process


def read_data():
    cont_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 
                 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
                 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
    disc_cols = ['Wilderness_Area_%d' % i for i in range(4)] + ['Soil_Type_%d' % i for i in range(40)]
    all_cols = cont_cols + disc_cols + ['Cover_Type']

    df = pd.read_csv("%s/Data/Downloads/COVTYPE/covtype.data" % HOME_PATH, header=None)
    df.columns = all_cols
    df['instance_id'] = range(df.shape[0])
    for i in range(7):
        df['label_%d' % i] = df['Cover_Type'].apply(lambda x: 1 if x == i + 1 else 0)
    return df, cont_cols, disc_cols, ['label_%d' % i for i in range(7)]


if __name__ == '__main__':
    df, cont_feats, disc_feats, use_labels = read_data()
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=82)

    data_process(df_train, df_test, 'instance_id', use_labels, 
                 disc_cols=disc_feats, cont_cols=cont_feats, other_info={},
                 skip_norm_cols=[], max_disc_val=100000, drop_disc_thres=1000000,
                 outpath=COVTYPE_DATASET_PATH)
