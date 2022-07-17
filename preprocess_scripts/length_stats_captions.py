import os 
import pandas as pd 
import numpy as np 
from tqdm import tqdm
from collections import Counter

csv_file="/bigdata/digbose92/Emotic/captions/caption_results_framesdb_emodb_small_mscoco_ade20k.csv"
csv_data=pd.read_csv(csv_file)
print(csv_data.shape)
csv_data=csv_data.dropna()
print(csv_data.shape)

caption_data=list(csv_data['caption'])
len_list=[]
for i in np.arange(len(caption_data)):
    caption_c=caption_data[i]
    #print(caption_c,type(caption_c))
    caption_c_list=caption_c.split(" ")
    len_list.append(len(caption_c_list))

print(Counter(len_list))
print(max(len_list)) #maximum length is 16


    
