import os 
import pandas as pd 
import numpy as np 
import cv2
import pickle
from tqdm import tqdm




train_data_pkl="/bigdata/digbose92/Emotic/pkl_files/train_data.pkl"
val_data_pkl="/bigdata/digbose92/Emotic/pkl_files/val_data.pkl"
test_data_pkl="/bigdata/digbose92/Emotic/pkl_files/test_data.pkl"

with open(train_data_pkl,"rb") as f:
    train_data=pickle.load(f)

with open(val_data_pkl,"rb") as f:
    val_data=pickle.load(f)

with open(test_data_pkl,"rb") as f:
    test_data=pickle.load(f)

#print(len(train_data),len(val_data),len(test_data))


base_pre_extracted_folder="/bigdata/digbose92/Emotic/prextracted_bottom_up_features/boxes"
for key in tqdm(list(train_data.keys())):
    sample=train_data[key]
    bbox_list=[]
    for k in sample:
        if('person_' in k):
            bbox=sample[k]['bbox']
            bbox_list.append([int(m) for m in list(bbox)])

    subfolder_name=sample['folder'].split("/")[0]
    dict_set={'bbox':bbox_list}
    overall_filename=os.path.join(base_pre_extracted_folder,subfolder_name,sample['filename'].split(".")[0]+".npz")
    np.savez(overall_filename,**dict_set)
    

for key in tqdm(list(val_data.keys())):
    sample=val_data[key]
    bbox_list=[]
    for k in sample:
        if('person_' in k):
            bbox=sample[k]['bbox']
            bbox_list.append([int(m) for m in list(bbox)])

    subfolder_name=sample['folder'].split("/")[0]
    dict_set={'bbox':bbox_list}
    overall_filename=os.path.join(base_pre_extracted_folder,subfolder_name,sample['filename'].split(".")[0]+".npz")
    np.savez(overall_filename,**dict_set)

for key in tqdm(list(test_data.keys())):
    sample=test_data[key]
    bbox_list=[]
    for k in sample:
        if('person_' in k):
            bbox=sample[k]['bbox']
            bbox_list.append([int(m) for m in list(bbox)])

    subfolder_name=sample['folder'].split("/")[0]
    dict_set={'bbox':bbox_list}
    overall_filename=os.path.join(base_pre_extracted_folder,subfolder_name,sample['filename'].split(".")[0]+".npz")
    np.savez(overall_filename,**dict_set)