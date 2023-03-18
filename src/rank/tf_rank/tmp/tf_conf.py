user_schema = [
    {'field':'user_device', 'fieldColumn':'Single'},
    {'field':'user_os', 'fieldColumn':'Single'},
    {'field':'user_province', 'fieldColumn':'Single'},
    {'field':'user_city', 'fieldColumn':'Single'},
    {'field':'user_age', 'fieldColumn':'KeyValue'},
    {'field':'user_gender', 'fieldColumn':'KeyValue'},
]

doc_schema = [
    {'field':'doc_title', 'fieldColumn':'KeyValue'},
    {'field':'doc_picNum', 'fieldColumn':'Numeric'},
    {'field':'doc_cate1', 'fieldColumn':'Single'},
    {'field':'doc_cate2', 'fieldColumn':'Single'},
    {'field':'doc_keywords', 'fieldColumn':'KeyValue'}
]
ctx_schema = [
    {'field':'user_id', 'fieldColumn':'Single'},
    {'field':'doc_id', 'fieldColumn':'Single'},
    {'field':'ctx_network', 'fieldColumn':'Single'},
    {'field':'ctx_refreshTimes', 'fieldColumn':'Single'},
]

label_schema = [
    {'field':'is_click', 'fieldColumn':'Single'},
    {'field':'watch_time', 'fieldColumn':'Numeric'},
]


MatchOpSchema = [
    {'field':'doc_refreshTimes_clk_mean_ratio','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_clk_cnt','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_exp_cnt','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_clk_ratio','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_exp_ratio','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_clk_mean','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_clk_mode','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_clk_median','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_clk_var','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_clk_skew','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_clk_kurt','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_clk_mean_ratio','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_watch_sum','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_watch_mean','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_watch_sum_ratio','fieldColumn':'Numeric'},
    {'field':'doc_refreshTimes_watch_mean_ratio','fieldColumn':'Numeric'},

    {'field':'doc_clk_cnt','fieldColumn':'Numeric'},
    {'field':'doc_exp_cnt','fieldColumn':'Numeric'},
    {'field':'doc_clk_mean','fieldColumn':'Numeric'},
    {'field':'doc_clk_mode','fieldColumn':'Numeric'},
    {'field':'doc_clk_median','fieldColumn':'Numeric'},
    {'field':'doc_clk_var','fieldColumn':'Numeric'},
    {'field':'doc_clk_skew','fieldColumn':'Numeric'},
    {'field':'doc_clk_kurt','fieldColumn':'Numeric'},
    {'field':'doc_watch_sum','fieldColumn':'Numeric'},
    {'field':'doc_watch_mean','fieldColumn':'Numeric'},

    {'field':'user_clk_cnt','fieldColumn':'Numeric'},
    {'field':'user_exp_cnt','fieldColumn':'Numeric'},
    {'field':'user_clk_mean','fieldColumn':'Numeric'},
    {'field':'user_clk_mode','fieldColumn':'Numeric'},
    {'field':'user_clk_median','fieldColumn':'Numeric'},
    {'field':'user_clk_var','fieldColumn':'Numeric'},
    {'field':'user_clk_skew','fieldColumn':'Numeric'},
    {'field':'user_clk_kurt','fieldColumn':'Numeric'},
    {'field':'user_watch_sum','fieldColumn':'Numeric'},
    {'field':'user_watch_mean','fieldColumn':'Numeric'},

    {'field':'user_cate1_clk_cnt','fieldColumn':'Numeric'},
    {'field':'user_cate1_exp_cnt','fieldColumn':'Numeric'},
    {'field':'user_cate1_clk_ratio','fieldColumn':'Numeric'},
    {'field':'user_cate1_exp_ratio','fieldColumn':'Numeric'},
    {'field':'user_cate1_clk_mean','fieldColumn':'Numeric'},
    {'field':'user_cate1_clk_mode','fieldColumn':'Numeric'},
    {'field':'user_cate1_clk_median','fieldColumn':'Numeric'},
    {'field':'user_cate1_clk_var','fieldColumn':'Numeric'},
    {'field':'user_cate1_clk_skew','fieldColumn':'Numeric'},
    {'field':'user_cate1_clk_kurt','fieldColumn':'Numeric'},
    {'field':'user_cate1_clk_mean_ratio','fieldColumn':'Numeric'},
    {'field':'user_cate1_watch_sum','fieldColumn':'Numeric'},
    {'field':'user_cate1_watch_mean','fieldColumn':'Numeric'},
    {'field':'user_cate1_watch_sum_ratio','fieldColumn':'Numeric'},
    {'field':'user_cate1_watch_mean_ratio','fieldColumn':'Numeric'},

    {'field':'user_title_clk_cnt','fieldColumn':'Sequence', 'sparse':True, 'dtype':'float'},
    {'field':'user_title_exp_cnt','fieldColumn':'Sequence', 'sparse':True, 'dtype':'float'},
    {'field':'user_title_clk_ratio','fieldColumn':'Sequence', 'sparse':True, 'dtype':'float'},
    {'field':'user_title_exp_ratio','fieldColumn':'Sequence', 'sparse':True, 'dtype':'float'},
    {'field':'user_title_clk_mean','fieldColumn':'Sequence', 'sparse':True, 'dtype':'float'},
    {'field':'user_title_clk_mode','fieldColumn':'Sequence', 'sparse':True, 'dtype':'float'},
    {'field':'user_title_clk_median','fieldColumn':'Sequence', 'sparse':True, 'dtype':'float'},
    {'field':'user_title_clk_var','fieldColumn':'Sequence', 'sparse':True, 'dtype':'float'},
    {'field':'user_title_clk_skew','fieldColumn':'Sequence', 'sparse':True, 'dtype':'float'},
    {'field':'user_title_clk_kurt','fieldColumn':'Sequence', 'sparse':True, 'dtype':'float'},
    {'field':'user_title_clk_mean_ratio','fieldColumn':'Sequence', 'sparse':True, 'dtype':'float'},
    {'field':'user_title_watch_sum','fieldColumn':'Sequence','sparse':True, 'dtype':'float'},
    {'field':'user_title_watch_mean','fieldColumn':'Sequence','sparse':True, 'dtype':'float'},
    {'field':'user_title_watch_sum_ratio','fieldColumn':'Sequence','sparse':True, 'dtype':'float'},
    {'field':'user_title_watch_mean_ratio','fieldColumn':'Sequence', 'sparse':True, 'dtype':'float'},

    {'field':'user_cate2_clk_cnt','fieldColumn':'Numeric'},
    {'field':'user_cate2_clk_ratio','fieldColumn':'Numeric'},
    {'field':'user_cate2_exp_cnt','fieldColumn':'Numeric'},
    {'field':'user_cate2_exp_ratio','fieldColumn':'Numeric'},
    {'field':'user_cate2_clk_mean','fieldColumn':'Numeric'},
    {'field':'user_cate2_clk_mode','fieldColumn':'Numeric'},
    {'field':'user_cate2_clk_median','fieldColumn':'Numeric'},
    {'field':'user_cate2_clk_var','fieldColumn':'Numeric'},
    {'field':'user_cate2_clk_skew','fieldColumn':'Numeric'},
    {'field':'user_cate2_clk_kurt','fieldColumn':'Numeric'},
    {'field':'user_cate2_clk_mean_ratio','fieldColumn':'Numeric'},
    {'field':'user_cate2_watch_sum','fieldColumn':'Numeric'},
    {'field':'user_cate2_watch_mean','fieldColumn':'Numeric'},
    {'field':'user_cate2_watch_sum_ratio','fieldColumn':'Numeric'},
    {'field':'user_cate2_watch_mean_ratio','fieldColumn':'Numeric'},
]
TimeOpSchema = [
    {'field':'clk_user_time_since_first','fieldColumn':'Numeric'},
    {'field':'clk_user_time_since_last','fieldColumn':'Numeric'},
    {'field':'clk_user_time_since_mean','fieldColumn':'Numeric'},
    {'field':'clk_user_time_since_mean_std','fieldColumn':'Numeric'},
    {'field':'clk_doc_time_since_first','fieldColumn':'Numeric'},
    {'field':'clk_doc_time_since_last','fieldColumn':'Numeric'},
    {'field':'clk_doc_time_since_mean','fieldColumn':'Numeric'},
    {'field':'clk_doc_time_since_mean_std','fieldColumn':'Numeric'},
    {'field':'clk_user_cate1_time_since_first','fieldColumn':'Numeric'},
    {'field':'clk_user_cate1_time_since_last','fieldColumn':'Numeric'},
    {'field':'clk_user_cate1_time_since_mean','fieldColumn':'Numeric'},
    {'field':'clk_user_cate1_time_since_mean_std','fieldColumn':'Numeric'},
    {'field':'clk_user_cate2_time_since_first','fieldColumn':'Numeric'},
    {'field':'clk_user_cate2_time_since_last','fieldColumn':'Numeric'},
    {'field':'clk_user_cate2_time_since_mean','fieldColumn':'Numeric'},
    {'field':'clk_user_cate2_time_since_mean_std','fieldColumn':'Numeric'},

    {'field':'clk_user_time_diff_mean', 'fieldColumn':'Numeric'},
    {'field':'clk_user_time_diff_std', 'fieldColumn':'Numeric'},
    {'field':'clk_user_time_diff_mean_std', 'fieldColumn':'Numeric'},
    {'field':'clk_doc_time_diff_mean', 'fieldColumn':'Numeric'},
    {'field':'clk_doc_time_diff_std', 'fieldColumn':'Numeric'},
    {'field':'clk_doc_time_diff_mean_std', 'fieldColumn':'Numeric'},
    {'field':'clk_user_cate1_time_diff_mean', 'fieldColumn':'Numeric'},
    {'field':'clk_user_cate1_time_diff_std', 'fieldColumn':'Numeric'},
    {'field':'clk_user_cate1_time_diff_mean_std', 'fieldColumn':'Numeric'},
    {'field':'clk_user_cate2_time_diff_mean', 'fieldColumn':'Numeric'},
    {'field':'clk_user_cate2_time_diff_std', 'fieldColumn':'Numeric'},
    {'field':'clk_user_cate2_time_diff_mean_std', 'fieldColumn':'Numeric'},
]
ProductOpSchema = [
    {'field':'user_cate1_product', 'fieldColumn':'Single'},
    {'field':'user_cate2_product', 'fieldColumn':'Single'},
    {'field':'clk_doc_product', 'fieldColumn':'Sequence', 'dtype':'int', 'sparse':True}
]
SequenceSchema = [
    {'field':'clk_doc_id', 'fieldColumn':'Sequence', 'dtype':'int'},
    {'field':'clk_doc_cate1', 'fieldColumn':'Sequence', 'dtype':'int'},
    {'field':'clk_doc_cate2', 'fieldColumn':'Sequence', 'dtype':'int'},
    {'field':'clk_watchTime', 'fieldColumn':'Sequence', 'dtype':'float'},
    {'field':'clk_refreshTimes', 'fieldColumn':'Sequence', 'dtype':'int'},
    {'field':'clk_doc_id_cnt', 'fieldColumn':'Sequence', 'dtype':'float'},
    {'field':'clk_doc_cate1_cnt', 'fieldColumn':'Sequence', 'dtype':'float'},
    {'field':'clk_doc_cate2_cnt', 'fieldColumn':'Sequence', 'dtype':'float'},
]

x_col = user_schema + doc_schema + ctx_schema + MatchOpSchema + TimeOpSchema  + SequenceSchema  + ProductOpSchema
y_col = label_schema