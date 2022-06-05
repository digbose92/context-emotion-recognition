#need to detect face within the person bounding box 
import pickle 
import pandas as pd 
import os 
import numpy as np 
from deepface import DeepFace
from deepface import commons
#from commons import functions 
import cv2

pkl_file="/bigdata/digbose92/Emotic/pkl_files/train_data.pkl"

with open(pkl_file,"rb") as f:
    data=pickle.load(f)
print(data.keys())
base_folder="/bigdata/digbose92/Emotic"
sample_key='COCO_train2014_000000017938.jpg'
print(data[sample_key])
filename=os.path.join(base_folder,data[sample_key]['folder'],sample_key)
print(filename)
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
#check detection from deepface

#print(folder)
#face = functions.preprocess_face(img_path = filename, target_size = (224, 224), detector_backend = backends[4], align=False, return_region=True)
#keep align = False if you don't want rotaion
face, face_region=DeepFace.detectFace(img_path = filename, target_size = (224, 224), detector_backend = backends[4], align=False)
print(face_region)
# embedding = DeepFace.represent(img_path = filename, detector_backend = backends[4], model_name = 'Facenet')

#APIs are working
#need to get the bbox directly and check if it lies within the vertical box 
# if it lies there then extract embedding for that person inside the box 
bbox_person_0=list(data[sample_key]['person_0']['bbox'])
bbox_person_1=list(data[sample_key]['person_1']['bbox'])
image=cv2.imread(filename)
image = cv2.rectangle(image, (int(bbox_person_0[0]), int(bbox_person_0[1])), (int(bbox_person_0[2]), int(bbox_person_0[3])), (0,0,255), 1)
image = cv2.rectangle(image, (int(bbox_person_1[0]), int(bbox_person_1[1])), (int(bbox_person_1[2]), int(bbox_person_1[3])), (0,0,255), 1)
image=cv2.rectangle(image, (int(face_region[0]), int(face_region[1])), (int(face_region[0]+face_region[2]), int(face_region[1]+face_region[3])), (0,0,255), 1)




#person detection + bounding box detection in the person box  
#print(embedding)
# detected_face = face * 255
# #cv2.imwrite("out.jpg", detected_face) #opencv reads images as bgr instead of rgb. that's why, this will be blue oriented image.
# cv2.imwrite("out_face_2.jpg", detected_face[:, :, ::-1])
cv2.imwrite('face_sample.jpg',image)


