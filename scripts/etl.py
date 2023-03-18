import pandas as pd
import sys
import glob

f_dir = "/home/tiejianjie/news_rec/data/feature_data/"
sup = int(sys.argv[1])

f_list = pd.concat([pd.read_csv(fs, sep='\t',nrows=sup) for d in range(24, 30) for fs in glob.glob(f"{f_dir}/processed_train_info.csv/day={d}/*.csv")], axis=0)
print(set(f_list[(f_list['6'] == 1) & (f_list['7'] <= 0) ]['7']))
