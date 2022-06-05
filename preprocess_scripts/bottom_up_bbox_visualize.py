import os 
import numpy as np 
import cv2
import pickle
from numpy import linalg as LA

image_file="/bigdata/digbose92/Emotic/mscoco/images/COCO_val2014_000000580621.jpg"
npz_feature="/bigdata/digbose92/Emotic/bottom_up_features/features/mscoco/COCO_val2014_000000580621.npz"

image=cv2.imread(image_file)
coco_val_feat=np.load(npz_feature)
x=coco_val_feat['x']
bbox=list(coco_val_feat['bbox'])

for bb in bbox:
    cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0,0,255), 1)

#cv2.imwrite('../samples/sample_bottom_up_feat.jpg',image)
#match the gt bbox with the boxes from the given set of bboxes
train_data_pkl="/bigdata/digbose92/Emotic/pkl_files/train_data.pkl"
val_data_pkl="/bigdata/digbose92/Emotic/pkl_files/val_data.pkl"
test_data_pkl="/bigdata/digbose92/Emotic/pkl_files/test_data.pkl"

with open(train_data_pkl,"rb") as f:
    train_data=pickle.load(f)

with open(val_data_pkl,"rb") as f:
    val_data=pickle.load(f)

with open(test_data_pkl,"rb") as f:
    test_data=pickle.load(f)

train_data_keys=list(train_data.keys())
val_data_keys=list(val_data.keys())
image_key=image_file.split("/")[-1]
#print(image_key)
# print(val_data[image_key])
# print(train_data[image_key])
#print(test_data[image_key])
person_0_bbox=list(test_data[image_key]['person_0']['bbox'])
#print(person_0_bbox)
norm_list=[]
min_norm=1000000
id=0
for id,bb in enumerate(bbox):
    bb_l=list(bb)
    bb_l_c=[int(b) for b in bb_l]
    c_norm=LA.norm(np.array(bb_l_c)-np.array(person_0_bbox))
    if(c_norm<min_norm):
        min_norm=c_norm
        min_id=id
    norm_list.append(c_norm)
    print(bb,person_0_bbox)

#print(norm_list[min_id],bbox[min_id],person_0_bbox)
    #print(bb_l_c)

# if(image_key in val_data_keys):
#     train_data_bbox=val_data[image_key]
#print(train_data_bbox)
#print(x.shape)