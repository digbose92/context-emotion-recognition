#helper code to check if the bottom up bounding box features are same for diff annotated instances in the same image

import os 
import pandas as pd 
import numpy as np
import cv2 
import pickle

pkl_file="/bigdata/digbose92/Emotic/pkl_files/train_aligned_bbox_caption_data.pkl"

with open(pkl_file,"rb") as f:
    data=pickle.load(f)

print(list(data.keys())[0:10])
#sample_data=data['COCO_train2014_000000462955.jpg']
#print(sample_data.keys())
for key in list(data.keys()):
    person_data=data[key]
    key_list=[k for k in list(person_data.keys()) if 'person' in k]
    if len(key_list) > 1:
        #print(key)
        # for k in key_list:
        #     person_feat=person_data[k]

        feat_list=np.array([person_data[k]['feat'] for k in key_list])
        val=((feat_list==feat_list[0]).all())
        if(val==True):
            print(key,feat_list.shape)


    #print(person_data.keys())
#print(data.keys())
