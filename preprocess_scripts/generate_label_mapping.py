#obtain label mapping for label string to actual integer label
import os 
import pickle 
import pandas as pd 
import numpy as np
from collections import Counter
from tqdm import tqdm

pkl_file="/bigdata/digbose92/Emotic/pkl_files/val_aligned_bbox_caption_data.pkl"
with open(pkl_file,"rb") as f:
    bbox_caption_data=pickle.load(f)

disc_emotion_list=[]
for key in tqdm(list(bbox_caption_data.keys())):
    bbox_c_data=bbox_caption_data[key]
    subsamp_keys=list(bbox_c_data.keys())
    person_keys=[p for p in subsamp_keys if 'person' in p]
    for key in person_keys:
        c_data=bbox_c_data[key]
        #print(c_data['cont_emotion'])
        #print(c_data['disc_emotion'])
        if(type(c_data['disc_emotion']) is str):
            disc_emotion_list=disc_emotion_list+[c_data['disc_emotion']]
        else:
            disc_emotion_list=disc_emotion_list+list(c_data['disc_emotion'])

disc_emotion_set=list(np.unique(np.array(disc_emotion_list)))
#print(disc_emotion_set)
dict_mapping={l:k for k,l in enumerate(disc_emotion_set)}
#print(dict_mapping)


# with open("/bigdata/digbose92/Emotic/pkl_files/discrete_emotion_mapping_Emotic.pkl","rb") as f:
#     discrete_emotion_mapping_Emotic=pickle.load(f)
# print(discrete_emotion_mapping_Emotic)
# print(dict_mapping)

with open("/bigdata/digbose92/Emotic/pkl_files/discrete_emotion_mapping_Emotic.pkl","wb") as f:
    pickle.dump(dict_mapping,f)
#print(len(set(Counter(disc_emotion_list)))))
        #print(c_data)
    #print(bbox_c_data.keys())