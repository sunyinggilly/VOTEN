import os
import sys
sys.path.append('../')
sys.path.append('../../')
import pandas as pd
from CONFIG import IJCAI18X_RAWDATA_PATH, IJCAI18X_DATASET_PATH
from DataProcessTools import data_process


def read_data():
    cont_cols = ['brand_age_delta', 'brand_age_mean', 'brand_age_std', 'brand_collected_delta', 'brand_collected_mean', 'brand_collected_sum', 
                 'brand_gender_ratio', 'brand_gender_ratio_delta', 'brand_his_show_perday', 'brand_his_trade_ratio', 'brand_item_count', 'brand_price_delta', 
                 'brand_pv_delta', 'brand_sales_delta', 'brand_sales_mean', 'brand_sales_sum', 'cate_age_delta', 'cate_age_mean', 'cate_age_std', 'cate_sales_sum', 
                 'cate_collected_mean', 'cate_collected_sum', 'cate_price_sum', 'cate_pv_mean', 'cate_pv_sum', 'cate_item_count', 'cate_his_trade_ratio', 
                 'cate_his_show_perday', 'cate_price_mean', 'cate_sales_mean', 'item_age_mean', 'item_age_std', 'item_gender_ratio', 'item_gender_ratio_delta', 
                 'item_his_show_delta', 'item_his_show_perday', 'item_his_show_ratio', 'item_his_trade_perday', 'item_his_trade_ratio', 'item_price_delta', 
                 'item_prop_num', 'item_sales_delta', 'ui_next_show_timedelta', 'ui_last_show_timedelta', 'ui_lasthour_show', 'ui_lastdate_show', 'ui_lastdate_trade', 
                 'user_his_show_perday', 'user_his_trade_ratio', 'user_last_show_timedelta', 'user_lastdate_show', 'user_near_timedelta', 'user_next_show_timedelta', 
                 'ia_his_trade_ratio', 'ia_his_trade_delta', 'uc_last_show_timedelta', 'uc_lasthour_show', 'uc_near_timedelta', 'uc_next_show_timedelta', 'uc_his_trade', 
                 'uc_his_trade_ratio', 'ia_his_show_delta', 'ia_his_show_ratio', 'ca_his_trade_ratio', 'cg_his_trade_ratio', 'ig_his_show_delta', 'ig_his_show_ratio', 
                 'ig_his_trade_ratio', 'ig_his_trade_delta', 'hour_his_trade_ratio', 'shop_age_delta', 'shop_age_mean', 'shop_age_std', 'shop_collected_delta', 
                 'shop_collected_mean', 'shop_collected_sum', 'shop_gender_ratio', 'shop_gender_ratio_delta', 'shop_his_show_perday', 'shop_his_show_ratio', 
                 'shop_his_trade_delta', 'shop_his_trade_ratio', 'shop_item_count', 'shop_item_count_delta', 'shop_price_delta', 'shop_price_mean', 'shop_price_sum', 
                 'shop_pv_delta', 'shop_pv_mean', 'shop_pv_sum', 'shop_review_num_delta', 'shop_review_positive_delta', 'shop_review_positive_rate', 'shop_sales_delta', 
                 'shop_sales_mean', 'shop_sales_sum', 'shop_score_delivery', 'shop_score_delivery_delta', 'shop_score_description', 'shop_score_description_delta',
                 'shop_score_service', 'shop_score_service_delta', 'shop_star_level_delta', 'prop_jaccard']
    disc_cols = ['item_category1', 'item_city_id', 'item_collected_level', 'item_price_level', 'item_pv_level', 'item_sales_level', 'hour', 'context_page_id', 'hour2',
                 'shop_star_level', 'shop_review_num_level', 'predict_cate_num_level', 'predict_prop_num_level', 'prop_intersect_num_level', 'is_predict_category']

    df_train = pd.read_csv("%s/train.csv" % IJCAI18X_RAWDATA_PATH)
    df_test = pd.read_csv("%s/test.csv" % IJCAI18X_RAWDATA_PATH)

    return df_train, df_test, cont_cols, disc_cols, ['is_trade']


if __name__ == '__main__':
    df_train, df_test, cont_feats, disc_feats, label_cols = read_data()

    if not os.path.exists(IJCAI18X_DATASET_PATH):
        os.makedirs(IJCAI18X_DATASET_PATH)    
    data_process(df_train, df_test, 'instance_id', label_cols, disc_cols=disc_feats, cont_cols=cont_feats, other_info={},
                 skip_norm_cols=[], max_disc_val=100000, drop_disc_thres=1000000, outpath=IJCAI18X_DATASET_PATH)
