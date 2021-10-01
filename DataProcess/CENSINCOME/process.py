import sys
sys.path.append('../')
sys.path.append('../../')
import pandas as pd
from CONFIG import CENSINCOME_DATASET_PATH, HOME_PATH
from DataProcessTools import data_process
import os


def read_data():
    df_train = pd.read_csv("%s/Data/Downloads/census-income/census-income.data" % HOME_PATH, header=None)
    df_test = pd.read_csv("%s/Data/Downloads/census-income/census-income.test" % HOME_PATH, header=None)
    cols = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY', 'AHSCOL', 'AMARITL', 
            'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN', 'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN',
            'CAPLOSS', 'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL', 'MARSUPWT', 
            'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN', 'NOEMP', 'PARENT', 'PEFNTVTY',
            'PEMNTVTY', 'PENATVTY', 'PRCITSHP', 'SEOTR', 'VETQVA', 'VETYN', 'WKSWORK', 'YEAR', 'LABEL']
    df_train.columns = cols
    df_train['instance_id'] = range(df_train.shape[0])
    df_test.columns = cols
    df_test['instance_id'] = range(df_test.shape[0])
    df_test['instance_id'] = df_test['instance_id'] + df_train.shape[0]

    lst = df_test['LABEL'].drop_duplicates().tolist()
    df_train['LABEL'] = df_train['LABEL'].apply(lambda x: 1 if x == ' 50000+.' else 0)
    df_test['LABEL'] = df_test['LABEL'].apply(lambda x: 1 if x == ' 50000+.' else 0)
    cont_feats = ['AAGE', 'AHRSPAY', 'CAPGAIN', 'CAPLOSS', 'DIVVAL', 'NOEMP']
    disc_feats = ['ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN', 'ASEX', 
                  'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL',
                  'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN', 'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY',  
                  'PRCITSHP', 'SEOTR', 'VETQVA', 'VETYN', 'WKSWORK']
    return df_train, df_test, cont_feats, disc_feats


if __name__ == '__main__':
    df_train, df_test, cont_feats, disc_feats = read_data()

    if not os.path.exists(CENSINCOME_DATASET_PATH):
        os.makedirs(CENSINCOME_DATASET_PATH)    
    data_process(df_train, df_test, 'instance_id', ['LABEL'], disc_cols=disc_feats, cont_cols=cont_feats, other_info={},
                 skip_norm_cols=[], max_disc_val=100000, drop_disc_thres=1000000, outpath=CENSINCOME_DATASET_PATH)
