import os 
import pickle
from tabnanny import filename_only 
import numpy as np
import pandas as pd
import cv2 

dict_file="/bigdata/digbose92/Emotic/pkl_files/train_data.pkl"

with open(dict_file,"rb") as f:
    dict_tot=pickle.load(f)

#print(dict_tot.keys())
#/bigdata/digbose92/Emotic/emodb_small/images/0nz4q07z6c7rq61a4s.jpg
#/bigdata/digbose92/Emotic/framesdb/images/frame_1lnp4dh222wanywj.jpg
#/bigdata/digbose92/Emotic/framesdb/images/frame_1igrkoa71c7x6skx.jpg
#/bigdata/digbose92/Emotic/framesdb/images/frame_3j4zzsqg15q28408.jpg
#/bigdata/digbose92/Emotic/framesdb/images/frame_04m57s2o9ge152gk.jpg
#/bigdata/digbose92/Emotic/framesdb/images/frame_5co0uifs6hakwn0c.jpg
#/bigdata/digbose92/Emotic/framesdb/images/frame_4dasj160ejy63tyk.jpg
#/bigdata/digbose92/Emotic/framesdb/images/frame_0oix01520t7yrc9t.jpg
#/bigdata/digbose92/Emotic/framesdb/images/frame_7w44g1slj9jutfkf.jpg
#/bigdata/digbose92/Emotic/framesdb/images/c
#/bigdata/digbose92/Emotic/emodb_small/images/4jqhog35xiqj4wi7kv.jpg
#/bigdata/digbose92/Emotic/emodb_small/images/6c41yc6yhfsg2uvzzs.jpg
#/bigdata/digbose92/Emotic/emodb_small/images/1wg1b1ylo6a63gka4a.jpg
#/bigdata/digbose92/Emotic/emodb_small/images/7vhai5b4p5hn13n3wu.jpg
#/bigdata/digbose92/Emotic/framesdb/images/frame_m5dt7e8izrf9b5e9.jpg
key_sample="COCO_train2014_000000160823.jpg"
#"COCO_train2014_000000327702.jpg"
#key_sample="5icy856i7iqgnn14cp.jpg""
#"COCO_train2014_000000057579.jpg"
#"COCO_train2014_000000530610.jpg"
base_folder="/bigdata/digbose92/Emotic"


dict_set=dict_tot[key_sample]
print(dict_set.keys())
folder=dict_set['folder']
sub_folder=os.path.join(base_folder,folder)
filename_ov=os.path.join(sub_folder,key_sample)
print(filename_ov)
image=cv2.imread(filename_ov)
print(image.shape)
for key in list(dict_set.keys()):
    if('person' in key):
        bbox=list(dict_set[key]['bbox'])
        print(bbox)
        disc_emotion=dict_set[key]['disc_emotion']
        print(disc_emotion)
        if(type(disc_emotion) is 'str'):
            disc_emotion_list=[disc_emotion]
        else:
            disc_emotion_list=list(disc_emotion)
        cont_emotion=dict_set[key]['cont_emotion']
        bbox=[int(bb) for bb in bbox]
        #print(bbox,disc_emotion,cont_emotion)
        image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 1)
        #disc_emotion_list=disc_emotion.split(" ")
        if(len(disc_emotion_list)==1):
            image=cv2.putText(image, disc_emotion_list[0], (bbox[0]-10, bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        else:
            disc_emotion_str=" ".join(disc_emotion_list)
            image=cv2.putText(image, disc_emotion_str, (bbox[0]-10, bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    
cv2.imwrite(key_sample,image)
#print(disc_emotion)
# print(dict_set['person_0']['bbox'])
# print(c)