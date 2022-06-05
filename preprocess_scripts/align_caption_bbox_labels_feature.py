import os 
import pandas as pd 
import numpy as np
import pickle 
import math
from numpy import linalg as LA
from tqdm import tqdm 
#def generate_aligned_data(base_extracted_feats_folder)

#pkl data
pkl_data_file="/bigdata/digbose92/Emotic/pkl_files/test_data.pkl"
split_option=pkl_data_file.split("/")[-1].split("_")[0]

#caption data 
caption_csv_file="/bigdata/digbose92/Emotic/captions/caption_results_framesdb_emodb_small_mscoco_ade20k.csv"
captions_data=pd.read_csv(caption_csv_file)
caption_image_id=list(captions_data['image_id'])
caption_image_filename=[c.split('/')[-1] for c in caption_image_id]
caption_set=list(captions_data['caption'])

#raw bbox + feature data
base_bbox_feat_folder="/bigdata/digbose92/Emotic/prextracted_bottom_up_features/features/"

with open(pkl_data_file,"rb") as f:
    pkl_data=pickle.load(f)

for key in tqdm(list(pkl_data.keys())):
    sample=pkl_data[key]
    filename=sample['filename']
    folder=sample['folder']
    
    #getting the exact caption for the image 
    caption_index=caption_image_filename.index(filename)
    sample['caption']=caption_set[caption_index]
    
    #find the bbox feature from the pre-extracted bbox features
    person_keys=[key for key in list(sample.keys()) if key.startswith('person')]
    subfold=folder.split("/")[-2]
    subfold_complete_path=os.path.join(base_bbox_feat_folder,subfold)
    feature_dict_path=os.path.join(subfold_complete_path,filename.split(".")[0]+".npz")
    feature_dict=np.load(feature_dict_path)
    feature_list=feature_dict['x']
    feature_bbox=feature_dict['bbox']
    feature_bbox=[[int(bf) for bf in bbox] for bbox in feature_bbox]
    
    for person_key in person_keys:
        bbox_c=sample[person_key]['bbox']
        c_norm=np.array([LA.norm(np.array(bbox_c)-np.array(person_0_bbox)) for person_0_bbox in feature_bbox])
        c_norm_min_id=np.argmin(c_norm)
        feat_c_person=feature_list[c_norm_min_id]
        sample[person_key]['feat']=feat_c_person
    
    pkl_data[key]=sample

dest_option_file="/bigdata/digbose92/Emotic/pkl_files/"+split_option+"_aligned_bbox_caption_data.pkl"
with open(dest_option_file,"wb") as f:
    pickle.dump(pkl_data,f)





